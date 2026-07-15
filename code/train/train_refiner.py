#!/usr/bin/env python3
"""Train the event-conditioned fusion refiner (v1 U-Net).

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
TRAINING-stage script (pipeline: data prep -> **training** -> member
inference -> fusion -> submission).  The refiner is one of "our own fusion
refiners" in the 30-member ensemble: a lightweight U-Net that, per pixel,
gates between several base interpolations (GIMM-VFI, TimeLens, linear frame
blend, optionally GIMM-refined) conditioned on the event voxel, and adds a
small bounded RGB residual.  Its outputs join the per-frame quadratic-form
fusion (recipe_v15/v16.json) at member-inference time.

Inputs:
  * --shards  pre-baked .npz shards (data-prep stage) holding, per sample:
    anchor frames a0/a1, GT middle, base predictions lin/gimm/tl (+optional
    gimmr), a 16-bin event voxel, and the interpolation phase alpha.
Outputs:
  * <out-dir>/refiner_step{N}.pth checkpoints ({"state_dict", "step"}).

Model I/O:
  Input (32ch): gimm(3) tl(3) lin(3) a0(3) a1(3) vox(16) alpha_map(1)
  Output: 3-way per-pixel softmax gate over {gimm, tl, lin} + RGB residual.
    out = sum_i gate_i * base_i + 0.1 * tanh(residual)
  Loss: L1 to GT. Trained purely on synthetic shards (no validation GT).

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/train_refiner.py \
      --shards data/refiner_shards --out-dir checkpoints/refiner --steps 30000
"""
import argparse
import glob
import io
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ model
class ConvBlock(nn.Module):
    """Two 3x3 conv + GELU layers — the basic U-Net building block."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.body(x)


class RefinerUNet(nn.Module):
    """3-level U-Net, ~6M params. n_bases gating heads + residual."""

    def __init__(self, cin=32, base=48, n_bases=3):
        super().__init__()
        self.n_bases = n_bases
        self.enc0 = ConvBlock(cin, base)
        self.enc1 = ConvBlock(base, base * 2)
        self.enc2 = ConvBlock(base * 2, base * 4)
        self.dec1 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec0 = ConvBlock(base * 2 + base, base)
        self.head_gate = nn.Conv2d(base, n_bases, 3, padding=1)
        self.head_res = nn.Conv2d(base, 3, 3, padding=1)
        # Zero-init both heads so training starts from a plain average of the
        # model bases (a safe, already-strong prediction) with no residual.
        nn.init.zeros_(self.head_res.weight)
        nn.init.zeros_(self.head_res.bias)
        nn.init.zeros_(self.head_gate.weight)
        # bias init: favor model bases equally, lin (last) less
        with torch.no_grad():
            bias = torch.ones(n_bases)
            bias[-1] = 0.0
            self.head_gate.bias.copy_(bias)

    def forward(self, x, bases):
        assert len(bases) == self.n_bases
        # Standard 3-level encoder/decoder with skip connections.
        e0 = self.enc0(x)
        e1 = self.enc1(F.avg_pool2d(e0, 2))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        d1 = self.dec1(torch.cat([F.interpolate(e2, scale_factor=2,
                                                mode="bilinear"), e1], 1))
        d0 = self.dec0(torch.cat([F.interpolate(d1, scale_factor=2,
                                                mode="bilinear"), e0], 1))
        gate = torch.softmax(self.head_gate(d0), dim=1)  # (B,n,H,W)
        # Residual is tanh-bounded to +/-0.1 so it can only fine-correct,
        # never overpower the gated base combination.
        res = 0.1 * torch.tanh(self.head_res(d0))
        out = sum(gate[:, i:i + 1] * bases[i] for i in range(self.n_bases)) + res
        return out, gate


# ------------------------------------------------------------------ data
class ShardDataset(torch.utils.data.Dataset):
    """Random crops from pre-baked shard_*.npz files (see module docstring)."""

    def __init__(self, shard_dir, crop=192, aug=False, event_drop=False):
        self.files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
        assert self.files, shard_dir
        # Probe the first shard: samples per shard and whether the optional
        # 4th base (gimmr = GIMM-refined) is present.
        probe = np.load(self.files[0])
        self.per_shard = len([k for k in probe.files if k.startswith("gt_")])
        self.has_gimmr = any(k.startswith("gimmr_") for k in probe.files)
        self.crop = crop
        self.aug = aug
        self.event_drop = event_drop
        print(f"{len(self.files)} shards x {self.per_shard} "
              f"(gimmr={self.has_gimmr})", flush=True)

    def __len__(self):
        return len(self.files) * self.per_shard

    def __getitem__(self, idx):
        # Fresh per-call RNG so DataLoader workers augment independently.
        rng = random.Random()
        z = np.load(self.files[idx // self.per_shard])
        i = idx % self.per_shard
        get = lambda k: z[f"{k}_{i}"]
        a0, a1, gt = get("a0"), get("a1"), get("gt")
        lin, gimm, tl = get("lin"), get("gimm"), get("tl")
        vox = get("vox").astype(np.float32)
        alpha = float(get("alpha"))

        gimmr = z[f"gimmr_{i}"] if f"gimmr_{i}" in z.files else None

        c = self.crop
        H, W = gt.shape[:2]
        y0 = rng.randrange(0, H - c + 1)
        x0 = rng.randrange(0, W - c + 1)
        sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        a0, a1, gt = a0[sl], a1[sl], gt[sl]
        lin, gimm, tl = lin[sl], gimm[sl], tl[sl]
        if gimmr is not None:
            gimmr = gimmr[sl]
        vox = vox[:, sl[0], sl[1]]

        # Random h/v flips applied identically to every image AND the voxel
        # (spatial dims only; bin/time axis untouched).
        imgs = [a0, a1, gt, lin, gimm, tl] + ([gimmr] if gimmr is not None else [])
        if rng.random() < 0.5:
            imgs = [x[:, ::-1] for x in imgs]
            vox = vox[:, :, ::-1]
        if rng.random() < 0.5:
            imgs = [x[::-1] for x in imgs]
            vox = vox[:, ::-1]
        if gimmr is not None:
            a0, a1, gt, lin, gimm, tl, gimmr = imgs
        else:
            a0, a1, gt, lin, gimm, tl = imgs

        if self.event_drop and rng.random() < 0.5:
            # EventDrop-style voxel aug: simulate sensor dropout / density
            # variation. Voxel only; images/GT untouched.
            mode = rng.randrange(3)
            B_, Hc, Wc = vox.shape
            if mode == 0:  # drop-by-area: zero a random rect (5-25% of area)
                frac = rng.uniform(0.05, 0.25)
                ar = rng.uniform(0.5, 2.0)
                h = min(max(int(round(np.sqrt(frac * Hc * Wc * ar))), 1), Hc)
                w = min(max(int(round(np.sqrt(frac * Hc * Wc / ar))), 1), Wc)
                y = rng.randrange(0, Hc - h + 1)
                x = rng.randrange(0, Wc - w + 1)
                vox = vox.copy()
                vox[:, y:y + h, x:x + w] = 0.0
            elif mode == 1:  # drop-by-time: zero 1-4 of the 16 bins
                vox = vox.copy()
                for b in rng.sample(range(B_), rng.randint(1, 4)):
                    vox[b] = 0.0
            else:  # pixel thinning: per-pixel Bernoulli mask, same across bins
                keep = rng.uniform(0.7, 0.95)
                np_rng = np.random.default_rng(rng.getrandbits(64))
                mask = (np_rng.random((Hc, Wc)) < keep).astype(np.float32)
                vox = vox * mask[None]

        if self.aug:
            # domain-bridging augs (v2 experiment: did NOT beat plain v1):
            # JPEG-compress anchors/bases and rescale voxel magnitude to
            # mimic the real-data domain gap. Kept for reproducibility.
            def jpeg(im):
                buf = io.BytesIO()
                Image.fromarray(np.ascontiguousarray(im)).save(
                    buf, "JPEG", quality=rng.randint(85, 99))
                buf.seek(0)
                return np.asarray(Image.open(buf).convert("RGB"))

            if rng.random() < 0.5:
                a0, a1 = jpeg(a0), jpeg(a1)
                # Rebuild the linear-blend base from the degraded anchors so
                # it stays consistent with them.
                lin = np.clip(np.rint((1 - alpha) * a0.astype(np.float64)
                                      + alpha * a1.astype(np.float64)),
                              0, 255).astype(np.uint8)
            if rng.random() < 0.3:
                gimm = jpeg(gimm)
            if rng.random() < 0.3:
                tl = jpeg(tl)
            vox = vox * rng.uniform(0.3, 1.5)

        to_t = lambda im: torch.from_numpy(
            np.ascontiguousarray(im.transpose(2, 0, 1)).astype(np.float32) / 255.0)
        out = {
            "gimm": to_t(gimm), "tl": to_t(tl), "lin": to_t(lin),
            "a0": to_t(a0), "a1": to_t(a1),
            "vox": torch.from_numpy(np.ascontiguousarray(vox)),
            "alpha": torch.tensor(alpha, dtype=torch.float32),
            "gt": to_t(gt),
        }
        if gimmr is not None:
            out["gimmr"] = to_t(gimmr)
        return out


def build_input(batch):
    """Assemble the network input tensor and the ordered base list.

    Channel layout: gimm(3) [gimmr(3)] tl(3) lin(3) a0(3) a1(3) vox(16)
    alpha_map(1); `bases` order must match the gate-head channel order.
    """
    B, _, H, W = batch["gimm"].shape
    # Broadcast the scalar interpolation phase alpha to a full-res map.
    amap = batch["alpha"][:, None, None, None].expand(B, 1, H, W)
    parts = [batch["gimm"]]
    bases = [batch["gimm"]]
    if "gimmr" in batch:
        parts.append(batch["gimmr"])
        bases.append(batch["gimmr"])
    parts += [batch["tl"], batch["lin"], batch["a0"], batch["a1"],
              batch["vox"], amap]
    bases += [batch["tl"], batch["lin"]]
    return torch.cat(parts, dim=1), tuple(bases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(ROOT, "data", "refiner_shards"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "refiner"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--aug", action="store_true",
                    help="enable v2 domain-bridging augs (did not help)")
    ap.add_argument("--event-drop", action="store_true",
                    help="EventDrop-style voxel aug (area/time drop, thinning)")
    ap.add_argument("--base", type=int, default=48, help="U-Net base channels")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = ShardDataset(args.shards, aug=args.aug, event_drop=args.event_drop)
    # If shards carry the optional gimmr base, widen input and gate to 4 bases.
    n_bases = 4 if ds.has_gimmr else 3
    cin = 32 + (3 if ds.has_gimmr else 0)
    net = RefinerUNet(cin=cin, base=args.base, n_bases=n_bases).cuda()
    print(f"params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"cin={cin} n_bases={n_bases}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True)

    step, t0, l_acc, base_acc = 0, time.time(), 0.0, 0.0
    while step < args.steps:
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            x, bases = build_input(batch)
            out, gate = net(x, bases)
            loss = F.l1_loss(out, batch["gt"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            # Monitoring baseline: L1 of the naive average of the first two
            # bases — the refiner should beat this margin.
            with torch.no_grad():
                base_l1 = F.l1_loss((bases[0] + bases[1]) / 2, batch["gt"])
            l_acc += float(loss); base_acc += float(base_l1)
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{args.steps} l1={l_acc/100:.5f} "
                      f"(avg-base {base_acc/100:.5f}) lr={sched.get_last_lr()[0]:.1e} "
                      f"{(time.time()-t0)/100:.2f}s/it", flush=True)
                l_acc, base_acc, t0 = 0.0, 0.0, time.time()
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"state_dict": net.state_dict(), "step": step},
                           os.path.join(args.out_dir, f"refiner_step{step}.pth"))
                print(f"saved step {step}", flush=True)
            if step >= args.steps:
                break
    print("training done", flush=True)


if __name__ == "__main__":
    main()
