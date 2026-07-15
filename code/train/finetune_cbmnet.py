#!/usr/bin/env python3
"""Fine-tune CBMNet (ours_large) on HQ-EVFI for the EventAid challenge.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
TRAINING-stage script (pipeline: data prep -> **training** -> member
inference -> fusion -> submission).  Produces the fine-tuned CBMNet-Large
checkpoints used as the ensemble's diversity member (held at fixed weight
0.20 in the final per-frame quadratic-form fusion): CBMNet's cross-modal
bidirectional-flow design errs differently from the EMA-VFI/RIFE family,
which is exactly why it earns a fixed seat in the blend.

Inputs:
  * --data-root HQ-EVFI release (frames + per-interval raw event .npz files)
  * --init      BS-ERGB-pretrained CBMNet-Large weights
Outputs:
  * <out-dir>/ft_step{N}.pth checkpoints ({"model_state_dict", "step"})
    loaded by the CBMNet member-inference script.

Data: HQ-EVFI sequences (visual_RGB/{idx}_{ts_ns}.png + RGB-EVS_*/{idx}_{ts}.npz,
npz idx covers frames idx-1 -> idx, keys x,y,t,p with p in {-1,+1}).
Samples mixed skips {1,3,7,15}; events are cropped in event space BEFORE
voxelization (fast). Voxels built with the repo's events_to_voxel_grid /
event_reverse so train == inference preprocessing exactly.

Training follows the repo's joint mode: flownet frozen, multi-scale L1 loss.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python3 scripts/finetune_cbmnet.py \
      --data-root data/data_release --init weights/ours_large_bsergb.pth \
      --out-dir checkpoints/cbmnet_ft --steps 8000 --batch 4
"""
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

# Repo root; put the vendored CBMNet repo (and its tools/) on sys.path so we
# can reuse its exact voxelization + model-manager code.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CBMNET = os.path.join(ROOT, "models", "CBMNet")
sys.path.insert(0, ROOT)
sys.path.insert(0, CBMNET)
sys.path.insert(0, os.path.join(CBMNET, "tools"))

from event_utils import event_reverse, events_to_voxel_grid  # noqa: E402
from models.model_manager import OurModel  # noqa: E402

NUM_BINS = 16  # voxel time bins, must match CBMNet's inference config
# Frame skips to sample: skip k => interpolate across k+1 base intervals,
# mixing easy (1) through hard (15) temporal gaps.
SKIP_CHOICES = (1, 3, 7, 15)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class HQEVFISeq:
    """One HQ-EVFI sequence: indexed RGB frames + per-interval event .npz files."""

    def __init__(self, seq_dir):
        self.dir = seq_dir
        rgb_dir = os.path.join(seq_dir, "visual_RGB")
        ev_dirs = glob.glob(os.path.join(seq_dir, "RGB-EVS*"))
        assert ev_dirs, seq_dir
        self.ev_dir = ev_dirs[0]
        frames = []
        for p in glob.glob(os.path.join(rgb_dir, "*.png")):
            base = os.path.basename(p)[:-4]
            idx, ts = base.split("_")
            frames.append((int(idx), int(ts), p))
        frames.sort()
        self.frames = frames
        self.ev_files = {}
        for p in glob.glob(os.path.join(self.ev_dir, "*.npz")):
            base = os.path.basename(p)[:-4]
            idx, _ = base.split("_")
            self.ev_files[int(idx)] = p
        # keep only frames forming a contiguous run with events available
        self.valid = [
            i for i in range(1, len(frames))
            if frames[i][0] == frames[i - 1][0] + 1 and frames[i][0] in self.ev_files
        ]

    def n_frames(self):
        return len(self.frames)


class HQEVFIDataset(torch.utils.data.Dataset):
    """Random (I0, It, I1) windows + CBMNet-style voxel triplets from HQ-EVFI."""

    def __init__(self, data_root, crop=256, length=100000, skips=SKIP_CHOICES):
        seq_dirs = sorted(
            d for d in glob.glob(os.path.join(data_root, "**", "visual_RGB"),
                                 recursive=True)
        )
        self.seqs = []
        for d in seq_dirs:
            try:
                s = HQEVFISeq(os.path.dirname(d))
                if s.n_frames() >= 20:
                    self.seqs.append(s)
            except AssertionError:
                continue
        assert self.seqs, f"no sequences under {data_root}"
        self.crop = crop
        self.length = length
        self.skips = skips
        print(f"dataset: {len(self.seqs)} sequences", flush=True)

    def __len__(self):
        return self.length

    def _load_events_remapped(self, seq, i0, i1):
        """Events for base intervals i0+1..i1 on a continuous us axis [t(i0), t(i1)].

        Each npz's own t range is linearly mapped onto its frame interval
        (frame ts are ns -> use us).
        """
        chunks = []
        for i in range(i0 + 1, i1 + 1):
            ts_a = seq.frames[i - 1][1] / 1e3
            ts_b = seq.frames[i][1] / 1e3
            z = np.load(seq.ev_files[seq.frames[i][0]])
            t = z["t"].astype(np.float64)
            if t.size == 0:
                continue
            t0, t1 = t.min(), t.max()
            tg = ts_a + (t - t0) / max(t1 - t0, 1e-9) * (ts_b - ts_a)
            ev = np.stack([tg, z["x"].astype(np.float64),
                           z["y"].astype(np.float64),
                           z["p"].astype(np.float64)], axis=1)
            chunks.append(ev)
        if not chunks:
            return np.empty((0, 4))
        ev = np.concatenate(chunks)
        return ev[np.argsort(ev[:, 0], kind="stable")]

    def __getitem__(self, _):
        # Fresh per-call RNG so every DataLoader worker samples independently.
        rng = random.Random()
        for _attempt in range(20):
            seq = rng.choice(self.seqs)
            skip = rng.choice(self.skips)
            step = skip + 1  # window spans `step` base intervals: frames a .. a+step
            n = seq.n_frames()
            if n < step + 1:
                continue
            a = rng.randrange(0, n - step)
            b = a + step
            # all base intervals must have event files and contiguous indices
            ok = all(
                seq.frames[i][0] == seq.frames[i - 1][0] + 1
                and seq.frames[i][0] in seq.ev_files
                for i in range(a + 1, b + 1)
            )
            if not ok:
                continue
            m = rng.randrange(a + 1, b)  # random interior frame = training target

            img0 = np.asarray(Image.open(seq.frames[a][2]).convert("RGB"), np.float32) / 255.0
            img1 = np.asarray(Image.open(seq.frames[b][2]).convert("RGB"), np.float32) / 255.0
            gt = np.asarray(Image.open(seq.frames[m][2]).convert("RGB"), np.float32) / 255.0
            H, W = img0.shape[:2]
            c = self.crop
            if H < c or W < c:
                continue
            y0 = rng.randrange(0, H - c + 1)
            x0 = rng.randrange(0, W - c + 1)

            # Events split at the target frame: [0,t] and [t,1] streams.
            ev_0t = self._load_events_remapped(seq, a, m)
            ev_t1 = self._load_events_remapped(seq, m, b)

            def crop_events(ev):
                # Crop in raw event space BEFORE voxelization (much faster
                # than voxelizing full frames and cropping the grids).
                if ev.shape[0] == 0:
                    return ev
                msk = ((ev[:, 1] >= x0) & (ev[:, 1] < x0 + c)
                       & (ev[:, 2] >= y0) & (ev[:, 2] < y0 + c))
                ev = ev[msk].copy()
                ev[:, 1] -= x0
                ev[:, 2] -= y0
                return ev

            # Build CBMNet's three voxel grids with the repo's own helpers so
            # training preprocessing is bit-identical to inference:
            #   0t = events 0->t forward, t0 = same events time-reversed,
            #   t1 = events t->1 forward.
            ev_0t, ev_t1 = crop_events(ev_0t), crop_events(ev_t1)
            zeros = np.zeros((NUM_BINS, c, c), dtype=np.float32)
            vox_0t = (events_to_voxel_grid(ev_0t.copy(), NUM_BINS, c, c)
                      if ev_0t.shape[0] else zeros)
            vox_t0 = (events_to_voxel_grid(event_reverse(ev_0t.copy()), NUM_BINS, c, c)
                      if ev_0t.shape[0] else zeros)
            vox_t1 = (events_to_voxel_grid(ev_t1.copy(), NUM_BINS, c, c)
                      if ev_t1.shape[0] else zeros)

            img0 = img0[y0:y0 + c, x0:x0 + c]
            img1 = img1[y0:y0 + c, x0:x0 + c]
            gt = gt[y0:y0 + c, x0:x0 + c]

            # random flips (events already voxelized -> flip arrays)
            if rng.random() < 0.5:
                img0, img1, gt = img0[:, ::-1], img1[:, ::-1], gt[:, ::-1]
                vox_0t, vox_t0, vox_t1 = (v[:, :, ::-1] for v in (vox_0t, vox_t0, vox_t1))
            if rng.random() < 0.5:
                img0, img1, gt = img0[::-1], img1[::-1], gt[::-1]
                vox_0t, vox_t0, vox_t1 = (v[:, ::-1] for v in (vox_0t, vox_t0, vox_t1))

            to_t = lambda im: torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1)))
            to_v = lambda v: torch.from_numpy(np.ascontiguousarray(v))
            return {
                "clean_image_first": to_t(img0),
                "clean_image_last": to_t(img1),
                "clean_middle": to_t(gt),
                "voxel_grid_0t": to_v(vox_0t),
                "voxel_grid_t0": to_v(vox_t0),
                "voxel_grid_t1": to_v(vox_t1),
            }
        raise RuntimeError("could not sample a valid window after 20 tries")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "data_release"))
    ap.add_argument("--init", default=os.path.join(ROOT, "weights", "ours_large_bsergb.pth"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "cbmnet_ft"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--unfreeze-flownet", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Minimal stand-in for the CBMNet repo's argparse namespace.
    class NetArgs:
        voxel_num_bins = NUM_BINS
        flow_tb_debug = False
        smoothness_weight = 10.0

    model = OurModel(NetArgs())
    model.initialize("final_models", "ours_large")  # build the ours_large variant
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]  # unwrap training-style checkpoints
    model.load_model(ckpt)
    model.cuda()
    if torch.cuda.device_count() > 1:
        model.use_multi_gpu()  # DataParallel over visible GPUs
    # Repo's "joint" fine-tune mode: flownet frozen by default, only the
    # fusion/synthesis parts train (matches the upstream recipe).
    model.set_mode("joint")
    if not args.unfreeze_flownet:
        model.fix_flownet()
    model.train()

    params = [p for p in model.net.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    print(f"trainable params: {n_param/1e6:.1f}M, lr={args.lr}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.1)

    ds = HQEVFIDataset(args.data_root, crop=args.crop,
                       length=args.steps * args.batch + args.batch)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True)

    step = 0
    t0 = time.time()
    loss_acc = 0.0
    for sample in loader:
        sample = {k: v.cuda(non_blocking=True) for k, v in sample.items()}
        opt.zero_grad()
        # Drive training through the repo's own manager API so the loss
        # (multi-scale L1) matches upstream CBMNet training exactly.
        model.set_train_input(sample)
        model.forward_nets()
        loss = model.get_multi_scale_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        loss_acc += float(loss)
        step += 1
        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step}/{args.steps} loss={loss_acc/50:.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {dt/50:.2f}s/it "
                  f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
            loss_acc = 0.0
            t0 = time.time()
        if step % args.save_every == 0 or step == args.steps:
            # Save the bare module (strip DataParallel wrapper if present).
            net = (model.net.module if hasattr(model.net, "module") else model.net)
            path = os.path.join(args.out_dir, f"ft_step{step}.pth")
            torch.save({"model_state_dict": net.state_dict(), "step": step}, path)
            print(f"saved {path}", flush=True)
        if step >= args.steps:
            break
    print("training done", flush=True)


if __name__ == "__main__":
    main()
