#!/usr/bin/env python3
"""Fast ensemble blend-weight optimization via per-frame quadratic forms.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375):
  data prep -> training -> member inference -> [FUSION (this script)] -> submission.
  This is the original, simplest fusion-stage optimizer: it searches linear
  RGB blend weights over a set of ensemble-member prediction directories by
  exhaustive simplex-grid enumeration, evaluated against the validation GT.
  It established the quadratic-form trick (precompute per-frame Gram matrices
  once, then score any weight vector in microseconds) and the position-bucket
  idea that later optimizers (blend_v11.py and successors, culminating in
  recipe_v15.json / recipe_v16.json used by scripts/build_final.py) refined
  with joint YCbCr objectives and coordinate ascent.

Inputs:
  * --dirs: member prediction dirs, each holding
    <skip>/room1/<index:06d>.png validation predictions.
  * challenge_data validation ground truth (via evlib.dataset.load_sequence).
Outputs:
  * Printed per-skip best PSNR and weight vectors (optionally per position
    bucket). This script only reports weights; recipes were written by the
    later optimizers.

For frame f with model predictions P_i and GT G:
  mse_f(w) = w' A_f w - 2 b_f' w + c_f
where A_f[i,j] = mean(P_i * P_j), b_f[i] = mean(P_i * G), c_f = mean(G^2).
Mean PSNR over frames is then evaluated for any w in milliseconds.

Also supports position buckets: frames are grouped by normalized distance
of their timestamp into the anchor gap, and weights are optimized per bucket.

Usage:
  python3 scripts/blend_opt.py --dirs results_rife_tta results_timelens_tta \
      results_cbmnet_bsergb_tta results_tlx_hqevfi [--buckets 3] [--step 0.05]
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evlib.dataset import load_sequence  # noqa: E402

# The four interpolation gaps of the challenge (frames skipped between anchors).
SKIPS = ("1skip", "3skip", "7skip", "15skip")


def load(p):
    """Load an image as float64 RGB array (H, W, 3)."""
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def frame_position(seq, todo):
    """Normalized position in anchor gap, folded to [0, 0.5] (symmetry)."""
    inputs = seq.inputs
    prev = max((f for f in inputs if f.index < todo.index), key=lambda f: f.index)
    nxt = min((f for f in inputs if f.index > todo.index), key=lambda f: f.index)
    a = (todo.timestamp - prev.timestamp) / (nxt.timestamp - prev.timestamp)
    return min(a, 1.0 - a)


def quad_forms(dirs, skip, data_dir="challenge_data"):
    """Precompute per-frame quadratic-form terms (A_f, b_f, c_f) plus folded
    frame positions over the whole validation sequence. After this single
    pass of image I/O, mse_f(w) = w'A_f w - 2 b_f'w + c_f for any weights w."""
    seq = load_sequence(data_dir, "validation", skip, "room1")
    A_list, b_list, c_list, pos_list = [], [], [], []
    n = len(dirs)
    for f in seq.todos:
        G = load(seq.root / "gt" / f"{f.index:06d}_img.jpg").ravel()
        P = np.stack([load(Path(d) / skip / "room1" / f"{f.index:06d}.png").ravel()
                      for d in dirs])
        A_list.append(P @ P.T / P.shape[1])   # Gram matrix of predictions (n,n)
        b_list.append(P @ G / P.shape[1])     # prediction-GT correlations (n,)
        c_list.append(float(G @ G) / P.shape[1])  # GT energy (scalar)
        pos_list.append(frame_position(seq, f))
    return (np.stack(A_list), np.stack(b_list), np.array(c_list),
            np.array(pos_list))


def mean_psnr(w, A, b, c):
    """Mean per-frame PSNR of blend weights w, from precomputed forms only."""
    mse = np.einsum("i,fij,j->f", w, A, w) - 2 * b @ w + c
    mse = np.maximum(mse, 1e-9)
    return float(np.mean(20 * np.log10(255.0) - 10 * np.log10(mse)))


def simplex_grid(n, step):
    """Yield every weight vector on the n-simplex whose entries are multiples
    of `step` and sum to 1 (each combination distributes k = 1/step quanta
    of mass over the n members)."""
    k = round(1.0 / step)
    for combo in itertools.combinations_with_replacement(range(n), k):
        w = np.bincount(np.array(combo), minlength=n) / k
        yield w


def optimize(A, b, c, n, step):
    """Exhaustive search over the simplex grid; returns best (weights, PSNR)."""
    best_w, best_v = None, -1e9
    for w in simplex_grid(n, step):
        v = mean_psnr(w, A, b, c)
        if v > best_v:
            best_w, best_v = w.copy(), v
    return best_w, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--buckets", type=int, default=1,
                    help="number of position buckets over folded position [0,0.5]")
    ap.add_argument("--skips", nargs="+", default=list(SKIPS))
    args = ap.parse_args()

    names = [Path(d).name.replace("results_", "") for d in args.dirs]
    n = len(args.dirs)
    print(f"models: {names}, step={args.step}, buckets={args.buckets}")

    for skip in args.skips:
        A, b, c, pos = quad_forms(args.dirs, skip)
        # 1skip has a single mid-gap target frame, so position bucketing is
        # meaningless there; fall back to one global weight vector.
        if args.buckets == 1 or skip == "1skip":
            w, v = optimize(A, b, c, n, args.step)
            print(f"{skip}: psnr={v:.3f} w={np.round(w, 3).tolist()}")
        else:
            # Split folded positions [0, 0.5] into equal-width buckets and
            # optimize an independent weight vector per bucket.
            edges = np.linspace(0, 0.5 + 1e-9, args.buckets + 1)
            total, cnt = 0.0, 0
            parts = []
            for k in range(args.buckets):
                m = (pos >= edges[k]) & (pos < edges[k + 1])
                if m.sum() == 0:
                    continue
                w, v = optimize(A[m], b[m], c[m], n, args.step)
                parts.append(f"  pos[{edges[k]:.2f},{edges[k+1]:.2f}) "
                             f"({int(m.sum())}f): psnr={v:.3f} w={np.round(w, 3).tolist()}")
                total += v * m.sum()
                cnt += m.sum()
            print(f"{skip}: bucketed mean psnr={total / cnt:.3f}")
            print("\n".join(parts))


if __name__ == "__main__":
    main()
