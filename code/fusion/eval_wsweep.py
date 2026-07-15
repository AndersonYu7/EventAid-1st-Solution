#!/usr/bin/env python3
"""Select the syn<->ERF scalar blend weight on the HELD-OUT split (paper-clean,
not on test).

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375):
  data prep -> training -> member inference -> [FUSION (this script)] -> submission.
  Fusion-stage weight-selection script for the ERF-X170FPS real-event
  specialist members. It picks the scalar blend weight w between the
  synthetic-trained EMA-E model and the ERF-finetuned specialists on a
  held-out slice of the ERF data — never on the challenge test set — so the
  chosen weight is untainted. The selected w feeds the ERF specialist blend
  that scripts/build_final.py routes into fast/medium-motion test scenes
  (worth about +0.09 dB on the hidden test set).

For each gap, blends pred = w*erf + (1-w)*syn and reports held-out PSNR over
a grid of w. gap8 uses the 7skip specialist, gap16 the 15skip one, matching
how the members map into the competition submission.

Inputs:
  * --root: held-out ERF sequences (processed_images/ + processed_events/ per
    sequence, 5-digit frame indexing).
  * --syn: checkpoint of the synthetic-trained EMA-E model.
  * --erf-g8 / --erf-g16: checkpoints of the ERF specialists for gap 8 / 16.
Outputs:
  * Printed PSNR table over the weight grid WS, per gap and combined.

Usage:
  python3 scripts/eval_wsweep.py --syn ckpt_emae_syn.pth \
      --erf-g8 ckpt_erf_gap8.pth --erf-g16 ckpt_erf_gap16.pth \
      [--root data/erf_holdout] [--crop 512] [--stride 100]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from emae_arch import build_emae                      # noqa: E402
from erf_data import events_to_voxel_crop             # noqa: E402
from run_emae import infer_mid, pad_to_32             # noqa: E402
import cv2                                            # noqa: E402

cv2.setNumThreads(1)
# Grid of candidate ERF fractions (w = 1.0 means pure ERF specialist output).
WS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]


def load(path):
    """Build an EMA-E network on GPU and load a checkpoint (handles both raw
    state dicts and {"model": ...} wrappers)."""
    net = build_emae().cuda().eval()
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    net.load_state_dict(sd, strict=False)
    return net


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/erf_holdout"))
    ap.add_argument("--syn", required=True)
    ap.add_argument("--erf-g8", required=True, help="7skip specialist (gap8)")
    ap.add_argument("--erf-g16", required=True, help="15skip specialist (gap16)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=100)
    args = ap.parse_args()

    syn = load(args.syn)
    erf = {8: load(args.erf_g8), 16: load(args.erf_g16)}
    seqs = sorted(glob.glob(os.path.join(args.root, "**", "processed_images"),
                            recursive=True))
    c = args.crop
    # se[gap][w] = [sum_se, n]
    se = {g: {w: [0.0, 0] for w in WS} for g in (8, 16)}
    for d in seqs:
        base = os.path.dirname(d); evs = os.path.join(base, "processed_events")
        imgs = sorted(glob.glob(os.path.join(d, "*.png")))
        idxs = set(int(os.path.basename(p)[:-4]) for p in imgs)
        if not idxs:
            continue
        lo, hi = min(idxs), max(idxs)
        # Center crop of size --crop keeps GPU memory bounded and avoids
        # sensor-border artifacts.
        H, W = cv2.imread(imgs[0]).shape[:2]
        y0, x0 = (H - c) // 2, (W - c) // 2
        for g in (8, 16):
            # Slide an anchor window of width g through the sequence with
            # --stride spacing; each valid (start, mid, end) triple becomes
            # one held-out evaluation sample.
            a = lo
            while a + g <= hi:
                mid = a + g // 2
                if {a, mid, a + g} <= idxs:
                    i0 = cv2.imread(os.path.join(d, f"{a:05d}.png"))
                    im = cv2.imread(os.path.join(d, f"{mid:05d}.png"))
                    i1 = cv2.imread(os.path.join(d, f"{a+g:05d}.png"))
                    if i0 is not None and im is not None and i1 is not None:
                        img0 = i0[y0:y0+c, x0:x0+c].astype(np.float32)
                        gt = im[y0:y0+c, x0:x0+c].astype(np.float32)
                        img1 = i1[y0:y0+c, x0:x0+c].astype(np.float32)
                        # Gather all event chunks spanning the anchor gap and
                        # voxelize them over the same crop window.
                        npz = [os.path.join(evs, f"{k:05d}.npz")
                               for k in range(a, a + g)
                               if os.path.exists(os.path.join(evs, f"{k:05d}.npz"))]
                        vox = events_to_voxel_crop(npz, y0, x0, c)
                        t0 = torch.from_numpy(img0.transpose(2,0,1)[None]).cuda()/255.
                        t1 = torch.from_numpy(img1.transpose(2,0,1)[None]).cuda()/255.
                        tv = torch.from_numpy(vox[None]).cuda()
                        # Pad spatial dims to multiples of 32 (network stride);
                        # h, w_ remember the original size for un-padding.
                        t0,h,w_ = pad_to_32(t0); t1,_,_ = pad_to_32(t1)
                        tv,_,_ = pad_to_32(tv, mode="constant")
                        # Both models take [frame0, frame1, event voxels]
                        # concatenated on channels; output [3] is the final
                        # full-resolution prediction of the multi-scale head.
                        ps = syn(torch.cat([t0,t1,tv],1))[3][:,:,:h,:w_]
                        pe = erf[g](torch.cat([t0,t1,tv],1))[3][:,:,:h,:w_]
                        ps = ps[0].clamp(0,1).cpu().numpy().transpose(1,2,0)*255
                        pe = pe[0].clamp(0,1).cpu().numpy().transpose(1,2,0)*255
                        # Accumulate squared error for every candidate weight
                        # so a single forward pass scores the whole grid.
                        for w in WS:
                            bl = w*pe + (1-w)*ps
                            se[g][w][0] += ((bl-gt)**2).mean(); se[g][w][1] += 1
                a += args.stride

    print("=== Held-out blend-weight sweep (PSNR; w = ERF fraction) ===")
    print(f"{'w':>6}" + "".join(f"{w:>8.2f}" for w in WS))
    for g in (8, 16):
        row = f"gap{g:<3}"
        for w in WS:
            s,n = se[g][w]
            row += f"{20*np.log10(255/np.sqrt(s/n)):>8.3f}" if n else f"{'-':>8}"
        print(row)
    # combined (both gaps)
    row = "ALL   "
    for w in WS:
        s = sum(se[g][w][0] for g in (8,16)); n = sum(se[g][w][1] for g in (8,16))
        row += f"{20*np.log10(255/np.sqrt(s/n)):>8.3f}"
    print(row)


if __name__ == "__main__":
    main()
