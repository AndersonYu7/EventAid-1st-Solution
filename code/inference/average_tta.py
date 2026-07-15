#!/usr/bin/env python3
"""Average flip-TTA prediction directories into a single member directory.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375):
  MEMBER-INFERENCE helper. Some member runners (run_timelens.py, run_cbmnet.py,
  run_refid.py) only support a single flip per run (--flip {none,h,v,hv}), so
  their 4-way flip test-time augmentation was produced by running the model
  four times into four sibling directories (base / fh / fv / fhv) and then
  averaging the four uint8 PNGs per frame into the final results_*_tta member
  directory consumed by the fusion recipes (recipe_v15/v16.json).

  Historically this averaging was done with an inline one-off snippet; this
  script re-creates that step as a first-class, reusable tool so the release
  is fully reproducible end to end.

Inputs:
  --dirs  two or more prediction directories with identical layout
          {skip}/{seq}/{idx}.png (e.g. the base and flipped runs)
  --out   output member directory (same layout)

Output:
  For every frame present in the FIRST input dir, the pixel-wise mean of all
  input PNGs (float64 accumulation, rounded once at the end) is written as
  uint8 PNG to --out. Frames missing from any later dir raise an error, so a
  partially-copied run cannot silently produce a biased member.

Usage:
  python3 average_tta.py \
      --dirs results_cbmnet_base results_cbmnet_fh results_cbmnet_fv results_cbmnet_fhv \
      --out results_cbmnet_tta
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="input prediction dirs ({skip}/{seq}/{idx}.png layout)")
    ap.add_argument("--out", required=True, help="output averaged member dir")
    args = ap.parse_args()

    if len(args.dirs) < 2:
        raise SystemExit("need at least two --dirs to average")

    # Enumerate frames from the first dir; all others must contain the same set.
    pngs = sorted(glob.glob(os.path.join(args.dirs[0], "*", "*", "*.png")))
    if not pngs:
        raise SystemExit(f"no PNGs found under {args.dirs[0]}")

    n = 0
    for p in pngs:
        rel = os.path.relpath(p, args.dirs[0])
        acc = np.asarray(Image.open(p), dtype=np.float64)
        for d in args.dirs[1:]:
            q = os.path.join(d, rel)
            if not os.path.exists(q):
                raise SystemExit(f"missing frame in {d}: {rel}")
            acc += np.asarray(Image.open(q), dtype=np.float64)
        avg = np.clip(np.round(acc / len(args.dirs)), 0, 255).astype(np.uint8)
        dst = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(avg).save(dst)
        n += 1
    print(f"averaged {len(args.dirs)} dirs -> {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
