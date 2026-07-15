#!/usr/bin/env python3
"""Fine-tune EMA-E (event-aware EMA-VFI, see emae_arch.py) on SYNTHETIC events.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
This is a TRAINING-stage script (pipeline: data prep -> **training** ->
member inference -> fusion -> submission).  It produces the synthetic-event
EMA-E checkpoints that serve as ensemble members of the final 30-member
blend: EMA-E is EMA-VFI with a zero-initialized event-voxel "graft"
(a small event encoder whose features are injected into the frozen-shape
backbone), so training starts exactly at pretrained EMA-VFI quality and can
only improve as the event branch learns.

Inputs:
  * --hq-root   HQ-EVFI data release (visual_RGB frame streams, ~142 fps)
  * --adobe-root Adobe240 extracted frame folders (240 fps)
  * --init      pretrained EMA-VFI weights (weights/emavfi/ours.pkl)
Outputs:
  * checkpoints/emae_ft/emae_step{N}.pkl  state dicts, saved every
    --save-every steps; downstream member-inference scripts load these.

Training data follows the proven randomized-simulation recipe of
finetune_cbmnet_syn.py: triplets (I0, It=middle, I1) from
  * HQ-EVFI visual_RGB streams (~142 fps), gap g in {8, 10, 12, 16}
  * Adobe240 frame dumps (240 fps), gap g in {12, 16, 20, 24}
with V2V-style ESIM events simulated over the whole gap (SUB=8 linear
sub-steps per source interval; thresholds U[0.2,1.2] with polarity asymmetry,
light noise) splatted into a single 16-bin whole-gap voxel.

t = 0.5 (the model's native fixed-timestep mode), crop 256; augmentations
(channel reversal, temporal swap, v/h flips) are applied to the voxel
consistently (swap => negate + reverse bins; flips => same spatial flip).

Optimizer: AdamW(wd=1e-4), pretrained params lr 2e-5, NEW event-encoder
params lr 2e-4 (separate group), warmup 200 + cosine to 10%; Laplacian loss
on pred + 0.5x per coarse merged prediction; grad clip 1.0.

Usage:
  CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/finetune_emae.py \
      --steps 8000 --batch 8 --save-every 1000 > logs/emae_ft.log 2>&1 &
"""
import argparse
import glob
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)  # avoid OpenCV thread oversubscription inside DataLoader workers

# Repo root = parent of this script's directory; make repo modules importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from emae_arch import build_emae, EV_BINS  # noqa: E402  (also adds EMA-VFI to path)
from evlib.v2v_core_esim import EventEmulator  # noqa: E402

HQ_GAPS = (8, 10, 12, 16)
ADOBE_GAPS = (12, 16, 20, 24)
SUB = 8  # temporal upsampling sub-steps per source interval
# event-emulator threshold range. Re-CALIBRATED 2026-06-13 to match REAL
# EventAid voxel stats (nonzero 12.1%, |max| 5.78): U[0.1,0.7] gives
# 0.138/5.41 vs the old U[0.2,1.2] which was 2x too sparse (0.062/3.03).
THRES_LO = 0.1
THRES_HI = 0.7


def slices_to_voxel(slices, n_bins=EV_BINS):
    """(K,H,W) signed count slices -> (n_bins,H,W) via bilinear temporal splat."""
    K, H, W = slices.shape
    voxel = np.zeros((n_bins, H, W), dtype=np.float32)
    if K == 1:
        voxel[0] = slices[0]
        return voxel
    for k in range(K):
        tn = k / (K - 1) * (n_bins - 1)
        b0 = int(np.floor(tn))
        dt = tn - b0
        voxel[b0] += slices[k] * (1.0 - dt)
        if b0 + 1 < n_bins:
            voxel[b0 + 1] += slices[k] * dt
    return voxel


class HQSeq:
    """HQ-EVFI visual_RGB dir: frames named {idx}_{timestamp}.png."""

    def __init__(self, rgb_dir):
        frames = []
        for p in glob.glob(os.path.join(rgb_dir, "*.png")):
            base = os.path.basename(p)[:-4]
            idx, ts = base.split("_")
            frames.append((int(idx), int(ts), p))
        frames.sort()
        self.frames = frames
        self.by_idx = {f[0]: f for f in frames}


class SynEvDataset(torch.utils.data.Dataset):
    """(img0, img1, gt-middle) + whole-gap 16-bin synthetic-event voxel."""

    def __init__(self, hq_root, adobe_root, crop=256, length=10**6):
        self.sources = []  # (kind, payload, gaps)
        for d in sorted(glob.glob(os.path.join(hq_root, "**", "visual_RGB"),
                                  recursive=True)):
            s = HQSeq(d)
            if len(s.frames) >= max(HQ_GAPS) + 1:
                self.sources.append(("hq", s, HQ_GAPS))
        n_hq = len(self.sources)
        for d in sorted(glob.glob(os.path.join(adobe_root, "*"))):
            if not os.path.isdir(d):
                continue
            frames = sorted(glob.glob(os.path.join(d, "*.jpg")))
            if len(frames) >= max(ADOBE_GAPS) + 1:
                self.sources.append(("adobe", frames, ADOBE_GAPS))
        assert n_hq > 0, hq_root
        print(f"syn-ev dataset: {n_hq} HQ-EVFI + {len(self.sources) - n_hq} "
              f"Adobe240 sequences", flush=True)
        self.crop = crop
        self.length = length

    def __len__(self):
        return self.length

    def _sample_clip(self, rng):
        """Return (g+1, H, W, 3) float32 contiguous-frame clip or None."""
        kind, payload, gaps = self.sources[rng.randrange(len(self.sources))]
        g = rng.choice(gaps)
        if kind == "hq":
            seq = payload
            pos = rng.randrange(0, len(seq.frames))
            i0 = seq.frames[pos][0]
            if any(i0 + k not in seq.by_idx for k in range(g + 1)):
                return None
            f0, fm, f1 = (seq.by_idx[i0], seq.by_idx[i0 + g // 2],
                          seq.by_idx[i0 + g])
            span = f1[1] - f0[1]
            if span <= 0 or abs((fm[1] - f0[1]) - span / 2) > 0.1 * span:
                return None  # middle frame not at temporal midpoint
            paths = [seq.by_idx[i0 + k][2] for k in range(g + 1)]
        else:
            frames = payload
            a = rng.randrange(0, len(frames) - g)
            paths = frames[a:a + g + 1]
        imgs = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                return None
            imgs.append(im)
        if any(im.shape != imgs[0].shape for im in imgs):
            return None
        return np.stack(imgs).astype(np.float32)  # (g+1, H, W, 3) BGR 0..255

    def __getitem__(self, _):
        # Fresh per-call RNG so every DataLoader worker samples independently.
        rng = random.Random()
        c = self.crop
        for _attempt in range(50):
            clip = self._sample_clip(rng)
            if clip is None:
                continue
            n, H, W = clip.shape[:3]
            g = n - 1
            if H < c or W < c:
                continue
            y0 = rng.randrange(0, H - c + 1)
            x0 = rng.randrange(0, W - c + 1)
            clip = clip[:, y0:y0 + c, x0:x0 + c]

            # ---- simulate events over the whole gap (calibrated V2V recipe)
            # ESIM-style emulators need finer temporal resolution than the
            # source frame rate, so linearly upsample the gray clip SUB x
            # before feeding it to the emulator.
            gray = clip.mean(axis=3)  # (g+1, c, c) luminance 0..255
            up = np.empty((g * SUB + 1, c, c), dtype=np.float32)
            for i in range(g):
                for s in range(SUB):
                    w_ = s / SUB
                    up[i * SUB + s] = (1 - w_) * gray[i] + w_ * gray[i + 1]
            up[-1] = gray[-1]
            base_thres = rng.uniform(THRES_LO, THRES_HI)
            asym = rng.uniform(1.0, 1.4)
            emu = EventEmulator(
                pos_thres=base_thres * (asym if rng.random() < 0.5 else 1.0),
                neg_thres=base_thres * (1.0 if rng.random() < 0.5 else asym),
                base_noise_std=rng.uniform(0.0, 0.03),
                hot_pixel_fraction=rng.uniform(0.0, 0.001),
                hot_pixel_std=rng.uniform(0.0, 0.03),
            )
            slices = emu.video_to_voxel(up).astype(np.float32)  # (g*SUB, c, c)
            vox = slices_to_voxel(slices)                       # (16, c, c)

            img0, gt, img1 = clip[0], clip[g // 2], clip[g]

            # ---- augmentations (voxel-consistent)
            if rng.random() < 0.5:  # channel reversal (gray/events unchanged)
                img0, gt, img1 = img0[:, :, ::-1], gt[:, :, ::-1], img1[:, :, ::-1]
            if rng.random() < 0.5:  # temporal swap: reverse + negate bins
                img0, img1 = img1, img0
                vox = -vox[::-1]
            if rng.random() < 0.5:  # vflip
                img0, gt, img1 = img0[::-1], gt[::-1], img1[::-1]
                vox = vox[:, ::-1]
            if rng.random() < 0.5:  # hflip
                img0, gt, img1 = img0[:, ::-1], gt[:, ::-1], img1[:, ::-1]
                vox = vox[:, :, ::-1]

            # Pack (img0, img1, gt) into one 9-channel HWC array so the
            # collate step moves a single contiguous tensor per sample.
            imgs9 = np.concatenate([img0, img1, gt], axis=2)  # (c, c, 9)
            return (torch.from_numpy(
                        np.ascontiguousarray(imgs9.transpose(2, 0, 1))),
                    torch.from_numpy(np.ascontiguousarray(vox)))
        raise RuntimeError("could not sample a valid triplet after 50 tries")


def lr_mult(step, steps, warmup=200):
    """LR schedule multiplier: linear warmup, then cosine decay to 10%."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(steps - warmup, 1)
    return 0.1 + 0.9 * (math.cos(math.pi * t) * 0.5 + 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hq-root", default=os.path.join(ROOT, "data", "data_release"))
    ap.add_argument("--adobe-root",
                    default=os.path.join(ROOT, "data", "corpora", "Adobe240_frames"))
    ap.add_argument("--init", default=os.path.join(ROOT, "weights", "emavfi", "ours.pkl"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "emae_ft"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5, help="pretrained-param lr")
    ap.add_argument("--lr-new", type=float, default=2e-4, help="event-encoder lr")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hq-gaps", type=int, nargs="+", default=None,
                    help="override HQ_GAPS (even values; middle must exist)")
    ap.add_argument("--adobe-gaps", type=int, nargs="+", default=None)
    ap.add_argument("--thres-lo", type=float, default=None,
                    help="event-emulator base-threshold low (calibrated 0.1)")
    ap.add_argument("--thres-hi", type=float, default=None,
                    help="event-emulator base-threshold high (calibrated 0.7)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    global HQ_GAPS, ADOBE_GAPS, THRES_LO, THRES_HI
    if args.hq_gaps:
        HQ_GAPS = tuple(args.hq_gaps)
    if args.adobe_gaps:
        ADOBE_GAPS = tuple(args.adobe_gaps)
    if args.thres_lo is not None:
        THRES_LO = args.thres_lo
    if args.thres_hi is not None:
        THRES_HI = args.thres_hi
    print(f"event thres U[{THRES_LO},{THRES_HI}]", flush=True)
    print(f"gaps: HQ={HQ_GAPS} Adobe={ADOBE_GAPS} seed={args.seed}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    net = build_emae(args.init)
    net.train()
    from model.loss import LapLoss  # EMA-VFI's Laplacian-pyramid loss (import needs its repo on sys.path)
    lap = LapLoss()

    # Two LR groups: freshly-added event encoder trains 10x faster than the
    # pretrained EMA-VFI backbone, which only needs a gentle nudge.
    new_params = [p for n, p in net.named_parameters() if "ev_encoder" in n]
    old_params = [p for n, p in net.named_parameters() if "ev_encoder" not in n]
    groups = [
        {"params": old_params, "lr": args.lr, "base_lr": args.lr},
        {"params": new_params, "lr": args.lr_new, "base_lr": args.lr_new},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    all_params = old_params + new_params
    print(f"params: {sum(p.numel() for p in all_params)/1e6:.2f}M total, "
          f"{sum(p.numel() for p in new_params)/1e3:.1f}K new "
          f"(lr {args.lr}/{args.lr_new}) batch={args.batch} steps={args.steps}",
          flush=True)

    ds = SynEvDataset(args.hq_root, args.adobe_root, crop=args.crop,
                      length=args.steps * args.batch + args.batch)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    step = 0
    loss_acc, t0 = 0.0, time.time()
    for imgs9, vox in loader:
        m = lr_mult(step, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = pg["base_lr"] * m
        imgs9 = imgs9.cuda(non_blocking=True) / 255.0
        vox = vox.cuda(non_blocking=True)
        imgs, gt = imgs9[:, :6], imgs9[:, 6:]  # ch 0-5: I0+I1, ch 6-8: GT middle
        # EMA-E input = 6 frame channels + 16 event-voxel channels.
        flow, mask, merged, pred = net(torch.cat([imgs, vox], 1))
        # Laplacian loss on the final prediction plus 0.5x on each coarse
        # merged (pre-refinement) prediction for deep supervision.
        loss = lap(pred, gt).mean()
        for merge in merged:
            loss = loss + lap(merge, gt).mean() * 0.5
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        loss_acc += float(loss)
        step += 1
        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step}/{args.steps} loss={loss_acc/50:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} {dt/50:.2f}s/it "
                  f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)
            loss_acc, t0 = 0.0, time.time()
        if step % args.save_every == 0 or step == args.steps:
            path = os.path.join(args.out_dir, f"emae_step{step}.pkl")
            torch.save(net.state_dict(), path)
            print(f"saved {path}", flush=True)
        if step >= args.steps:
            break
    print("training done", flush=True)


if __name__ == "__main__":
    main()
