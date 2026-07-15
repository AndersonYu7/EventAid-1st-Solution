#!/usr/bin/env python3
"""Quick per-skip mean PSNR of whatever PNGs are present in a results dir.

Role in the 1st-place EventAid-F pipeline (team yunyu8)
-------------------------------------------------------
Stage: MEMBER INFERENCE and FUSION (fast scoring utility). This is the
workhorse scorer used throughout development to grade any prediction
directory -- a single ensemble member, a partial inference run still in
progress, or a fused blend -- against the validation ground truth. Partial
result dirs are fine: only frames that exist AND have a matching GT image are
scored, so it can be pointed at a run mid-inference. Scores from this tool
guided which members entered the 30-member ensemble and how fusion recipes
were iterated (final full scoring used val_blend_score.py / build_final.py).

Inputs
------
- one or more results dirs laid out as {skip}/{seq}/{idx}.png
- --data DIR : validation dir holding {skip}/{seq}/gt/{idx}_img.jpg
  (default: <repo>/challenge_data/validation)

Outputs
-------
- stdout: per-skip mean PSNR (dB) with frame counts, optional per-sequence
  breakdown (--per-seq), and the overall per-frame mean per results dir.

Usage
-----
    python3 mini_psnr.py [--data DATA_DIR] [--per-seq] results_dir [...]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Repo root (this file originally lived in <repo>/scripts/), used only for the
# default --data location.
ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "challenge_data" / "validation"))
ap.add_argument("--per-seq", action="store_true")
ap.add_argument("results", nargs="+")
args = ap.parse_args()
DATA = Path(args.data)


def psnr(a, b):
    # 8-bit PSNR in dB; a perfect match is capped at 99 dB instead of inf so
    # means stay finite.
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


for res_dir in args.results:
    res_dir = Path(res_dir)
    print(f"== {res_dir.name}")
    all_vals = []
    # The four official EventAid-F skip levels (skip N = N frames to insert).
    for skip in ("1skip", "3skip", "7skip", "15skip"):
        vals = []
        by_seq = defaultdict(list)
        for png in sorted((res_dir / skip).glob("*/*.png")):
            seq = png.parent.name
            # GT naming convention: prediction "0042.png" <-> "0042_img.jpg".
            gt_path = DATA / skip / seq / "gt" / f"{png.stem}_img.jpg"
            if not gt_path.exists():
                # Skip frames without GT (anchors / not-yet-released frames);
                # this is what makes scoring partial dirs safe.
                continue
            pred = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)
            gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float32)
            v = psnr(pred, gt)
            vals.append(v)
            by_seq[seq].append(v)
        if vals:
            all_vals += vals
            print(f"  {skip:7s}: {np.mean(vals):.3f} dB over {len(vals)} frames")
            if args.per_seq:
                for seq in sorted(by_seq):
                    sv = by_seq[seq]
                    print(f"    {seq:16s}: {np.mean(sv):.3f} dB ({len(sv)})")
    if all_vals:
        # Overall = flat mean over every scored frame (all skips pooled),
        # matching the challenge's per-frame averaging.
        print(f"  mean over {len(all_vals)} frames: {np.mean(all_vals):.3f} dB")
