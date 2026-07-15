#!/usr/bin/env python3
"""Create a "model soup" checkpoint by linear interpolation of two state dicts.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375):
  TRAINING-STAGE helper. The ensemble member results_cbmnet_soupuf35_tta was
  produced by CBMNet-Large running a *soup* checkpoint:

      soup = (1 - alpha) * base + alpha * fine_tuned      (alpha = 0.35)

  where `base` is the public CBMNet-Large weight and `fine_tuned` is our
  unfrozen synthetic-event fine-tune of it (checkpoints/cbmnet_ftsyn_uf/
  ftsyn_step8000.pth). Souping keeps the robustness of the public weight
  while pulling it toward the challenge event statistics; alpha = 0.35 was
  selected on the official validation scene (never on test data).

  Historically this mix was done with an inline one-liner; this script
  re-creates that step as a first-class tool. The resulting soup used by the
  final pipeline is shipped as models/cbmnet_soup_uf35.pth (the raw
  fine-tuned checkpoint was deleted in a disk cleanup; see models/MANIFEST.md).

Usage (historical call that produced the shipped soup):
  python3 make_cbmnet_soup.py \
      --base weights/ours_large_weight.pth \
      --ft   checkpoints/cbmnet_ftsyn_uf/ftsyn_step8000.pth \
      --alpha 0.35 \
      --out  checkpoints/cbmnet_soupuf35.pth
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base state-dict checkpoint (.pth)")
    ap.add_argument("--ft", required=True, help="fine-tuned state-dict checkpoint (.pth)")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="fine-tuned fraction: soup = (1-a)*base + a*ft")
    ap.add_argument("--out", required=True, help="output soup checkpoint path")
    args = ap.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    ft = torch.load(args.ft, map_location="cpu", weights_only=False)
    # Unwrap common {'state_dict': ...} / {'model': ...} containers if present.
    for key in ("state_dict", "model"):
        if isinstance(base, dict) and key in base and isinstance(base[key], dict):
            base = base[key]
        if isinstance(ft, dict) and key in ft and isinstance(ft[key], dict):
            ft = ft[key]

    a = args.alpha
    soup, skipped = {}, []
    for k, v in base.items():
        if k in ft and torch.is_tensor(v) and torch.is_tensor(ft[k]) and v.shape == ft[k].shape:
            if v.is_floating_point():
                # Linear interpolation in weight space (the actual souping).
                soup[k] = (1.0 - a) * v.float() + a * ft[k].float()
                soup[k] = soup[k].to(v.dtype)
            else:
                # Integer buffers (counters etc.) cannot be interpolated — keep base.
                soup[k] = v
        else:
            # Key missing in the fine-tune or shape-mismatched — keep base, report.
            soup[k] = v
            skipped.append(k)

    torch.save(soup, args.out)
    print(f"soup = {1-a:.2f}*base + {a:.2f}*ft -> {args.out} "
          f"({len(soup)} tensors, {len(skipped)} kept from base)")
    if skipped:
        print("kept-from-base keys:", skipped[:10], "..." if len(skipped) > 10 else "")


if __name__ == "__main__":
    main()
