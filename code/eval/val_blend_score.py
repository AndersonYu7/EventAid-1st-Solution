#!/usr/bin/env python3
"""Blend a candidate ensemble member into the v16 fusion output and score it on val.

Role in the 1st-place EventAid-F pipeline (team yunyu8)
-------------------------------------------------------
Stage: FUSION (validation-side member admission test). After a new member has
been run through inference, this script answers the decisive question "does
this member add value on the validation set?" before it is admitted into the
per-frame quadratic-form fusion recipe (recipe_v15/v16.json). It sweeps a
global blend weight w over

    pred = (1 - w) * v16 + w * candidate

and reports per-skip plus overall validation PSNR (per-frame average), using
the exact same blending method as the ERF/test-side blend so val and test
results are directly comparable. The validation set is room1 only (95 frames
across the four skip levels); the w = 0 row must reproduce the official
43.1056 dB score, which doubles as a sanity check of the setup.

Inputs
------
- submission_v16/results/{skip}/{seq}/*.png          current best fused preds
- challenge_data/validation/{skip}/{seq}/gt/*.jpg    ground-truth frames
- --cand DIR : candidate member's results dir in the same {skip}/{seq}/*.png
  layout (frames missing from the candidate silently fall back to plain v16)

Outputs
-------
- stdout table: one row per blend weight w, with PSNR for each skip level
  (1/3/7/15) and the OVERALL per-frame mean.

Usage
-----
    python3 val_blend_score.py --cand /path/to/candidate/results \
        --ws 0,0.1,0.2,0.3
"""
import argparse
import glob
import os
import numpy as np
from PIL import Image

# Fixed project paths: v16 = current best fused submission, GT = challenge
# validation ground truth. The four skip levels are the official EventAid-F
# interpolation gaps (skip N = N intermediate frames to synthesize).
ROOT = "/home/ubuntu-5th/work/EventAid"
V16 = f"{ROOT}/submission_v16/results"
GT = f"{ROOT}/challenge_data/validation"
SKIPS = ("1skip", "3skip", "7skip", "15skip")


def load(p):
    # Load an image as float64 RGB (float64 keeps the blend arithmetic exact).
    return np.asarray(Image.open(p).convert("RGB"), np.float64)


def psnr(a, b):
    # Standard 8-bit PSNR in dB (peak = 255), matching the challenge metric.
    return 20 * np.log10(255 / np.sqrt(((a - b) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="candidate results dir")
    ap.add_argument("--seq", default="room1")
    ap.add_argument("--ws", default="0,0.3,0.5,0.7,1.0")
    args = ap.parse_args()
    WS = [float(x) for x in args.ws.split(",")]

    # per[w][skip] collects one PSNR value per validation frame.
    per = {w: {s: [] for s in SKIPS} for w in WS}
    for skip in SKIPS:
        seqp = f"{V16}/{skip}/{args.seq}"
        gtdir = f"{GT}/{skip}/{args.seq}/gt"
        # Iterate over the v16 predictions; each PNG index must have a matching
        # GT jpg ("{idx}_img.jpg") to be scored.
        for pp in sorted(glob.glob(f"{seqp}/*.png")):
            idx = os.path.basename(pp)[:-4]
            g = f"{gtdir}/{idx}_img.jpg"
            cp = f"{args.cand}/{skip}/{args.seq}/{idx}.png"
            if not os.path.exists(g):
                continue
            gt = load(g)
            v16 = load(pp)
            # Candidate may only cover a subset of frames (e.g. a specialist
            # run on selected scenes); missing frames fall back to pure v16.
            cand = load(cp) if os.path.exists(cp) else None
            for w in WS:
                # Convex blend in float space; w=0 (or no candidate frame)
                # reproduces the v16 baseline exactly.
                pred = (1 - w) * v16 + w * cand if (cand is not None and w > 0) else v16
                per[w][skip].append(psnr(pred, gt))

    # Report: one row per weight, per-skip means plus the flat per-frame mean
    # over all skips (the challenge's OVERALL aggregation).
    print(f"=== val blend (cand={os.path.basename(args.cand)}) per-frame PSNR ===")
    hdr = f"{'w':>6}" + "".join(f"{s:>9}" for s in SKIPS) + f"{'OVERALL':>10}"
    print(hdr)
    for w in WS:
        allf = []
        row = f"{w:>6.2f}"
        for s in SKIPS:
            row += f"{np.mean(per[w][s]):>9.3f}"
            allf += per[w][s]
        row += f"{np.mean(allf):>10.4f}"
        print(row)


if __name__ == "__main__":
    main()
