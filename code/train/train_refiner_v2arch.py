#!/usr/bin/env python3
"""Train the upgraded (deformable + temporal-attention) fusion refiner (v2).

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
TRAINING-stage script (pipeline: data prep -> **training** -> member
inference -> fusion -> submission).  Trains RefinerV2 (see
scripts/refiner_v2arch.py), the stronger architecture variant of the fusion
refiner: it consumes the event voxel and anchor frames through a deformable
+ temporal-attention trunk instead of the plain v1 U-Net, then gates the
gimm/tl/lin base interpolations per pixel.  Trained v2 refiners feed the
28-member ensemble as additional members alongside the v1 refiner.

Reuses the v1 shard format (gimm/tl/lin bases) via train_refiner.ShardDataset.
v1 training recipe (no augs, which empirically transferred best),
Charbonnier + 0.2*gradient loss.

Inputs:
  * --shards  pre-baked refiner shards (same .npz format as train_refiner.py)
Outputs:
  * <out-dir>/v2arch_step{N}.pth checkpoints, storing state_dict plus the
    feat/base width hyper-params needed to rebuild the model at inference.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/train_refiner_v2arch.py \
      --shards data/refiner_shards --out-dir checkpoints/refiner_v2arch \
      --steps 15000 --batch 12
"""
import argparse
import os
import sys
import time

import torch

# Repo root; expose repo modules and sibling scripts for import.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from refiner_v2arch import RefinerV2, charbonnier, gradient_l1  # noqa: E402
from train_refiner import ShardDataset  # noqa: E402


def build_input_v2(batch):
    """Split the batch into RefinerV2's two input streams plus the bases.

    Unlike v1 (one concatenated tensor), v2 takes (voxel, anchors) where
    anchors = a0(3) + a1(3) + broadcast alpha map(1); bases order
    (gimm, tl, lin) must match the gate-head channel order.
    """
    B, _, H, W = batch["gimm"].shape
    # Broadcast the scalar interpolation phase alpha to a full-res map.
    amap = batch["alpha"][:, None, None, None].expand(B, 1, H, W)
    anchors = torch.cat([batch["a0"], batch["a1"], amap], dim=1)
    bases = (batch["gimm"], batch["tl"], batch["lin"])
    return (batch["vox"], anchors), bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=os.path.join(ROOT, "data", "refiner_shards"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "refiner_v2arch"))
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--feat", type=int, default=32)
    ap.add_argument("--base", type=int, default=56)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = ShardDataset(args.shards)
    net = RefinerV2(n_bases=3, feat=args.feat, base=args.base).cuda()
    print(f"params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.05)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=True)

    step, t0, l_acc = 0, time.time(), 0.0
    while step < args.steps:
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            x, bases = build_input_v2(batch)
            out, _ = net(x, bases)  # second return value is the gate map (unused here)
            # Charbonnier for robustness + gradient L1 for edge sharpness.
            loss = charbonnier(out, batch["gt"]) + 0.2 * gradient_l1(out, batch["gt"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            l_acc += float(loss)
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{args.steps} loss={l_acc/100:.5f} "
                      f"lr={sched.get_last_lr()[0]:.1e} "
                      f"{(time.time()-t0)/100:.2f}s/it "
                      f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                      flush=True)
                l_acc, t0 = 0.0, time.time()
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"state_dict": net.state_dict(), "step": step,
                            "feat": args.feat, "base": args.base},
                           os.path.join(args.out_dir, f"v2arch_step{step}.pth"))
                print(f"saved step {step}", flush=True)
            if step >= args.steps:
                break
    print("training done", flush=True)


if __name__ == "__main__":
    main()
