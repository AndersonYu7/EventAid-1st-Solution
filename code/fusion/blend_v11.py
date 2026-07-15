#!/usr/bin/env python3
"""Blend v11: joint YCbCr position-bucketed ensemble-weight optimization.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375):
  data prep -> training -> member inference -> [FUSION (this script)] -> submission.
  This is the recipe *optimizer* of the fusion stage. Given the per-member
  prediction directories produced by the member-inference stage (28-member
  ensemble: EMA-E grafts, EMA-VFI/RIFE/GIMM-VFI/VFIMamba/TimeLens/TimeLens-XL/
  REFID/CBMNet variants, fusion refiners), it fits per-skip, per-position-bucket
  linear blend weights on the validation split. Successive runs of this
  optimizer produced the recipe lineage v11 -> ... -> recipe_v15.json /
  recipe_v16.json, which scripts/build_final.py consumes to assemble the
  final uint8 PNG submission (val 43.1056 dB / test 31.8835 dB).

Inputs:
  * --base-recipe: previous recipe JSON (defines position buckets and the
    existing member pool per skip; used as warm start).
  * --add SKIP DIR pairs: new member prediction dirs to inject into a skip's
    pool (each dir holds <skip>/room1/<index:06d>.png validation predictions).
  * challenge_data validation ground truth (via evlib.dataset.load_sequence).
Outputs:
  * --out: new recipe JSON mapping each skip to a list of position buckets,
    each with sparse per-member weight dicts "weights_y" (luma) and
    "weights_c" (chroma). Per-bucket validation PSNR deltas are printed.

Improvements over the v10 procedure:
  * EXACT RGB-MSE objective: per-frame joint quadratic forms over the
    concatenated weight vector u = [w_y (n); w_c (n)] including Y<->C cross
    terms through Q = Minv' Minv (v10 optimized Y-MSE and C-MSE separately,
    a proxy that ignores channel mixing in the YCbCr->RGB transform).
  * Warm starts include the v10 per-bucket weights (zero-padded for new
    members), so per-bucket validation PSNR can only improve.
  * Pairwise coordinate ascent with shrinking step sizes after a coarse
    multi-start, instead of a pure simplex grid (intractable for n>8).

Usage:
  python3 scripts/blend_v11.py --base-recipe recipe_v10.json \
      --out recipe_v11.json \
      --add 1skip results_emae_s7000_tta --add 3skip results_emae_s7000_tta \
      --add 7skip results_emae_s7000_tta --add 15skip results_emae_s7000_tta \
      --add 15skip results_emae_s8000_tta
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evlib.dataset import load_sequence  # noqa: E402

# ITU-R BT.601 RGB -> YCbCr transform (full-range, no offsets needed because
# only error differences are measured, and offsets cancel in pred - GT).
YCBCR = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
YCBCR_INV = np.linalg.inv(YCBCR)
Q = YCBCR_INV.T @ YCBCR_INV  # rgb_mse*3 = e_ycc' Q e_ycc summed over channels


def folded_position(seq, todo):
    """Normalized position of a target frame within its anchor gap, folded to
    [0, 0.5]: frames symmetric about the midpoint behave alike, so buckets are
    defined on the folded coordinate."""
    inputs = seq.inputs
    prev = max((f for f in inputs if f.index < todo.index), key=lambda f: f.index)
    nxt = min((f for f in inputs if f.index > todo.index), key=lambda f: f.index)
    a = (todo.timestamp - prev.timestamp) / (nxt.timestamp - prev.timestamp)
    return min(a, 1.0 - a)


def load_ycc(path):
    """Load an image and convert flattened pixels to YCbCr, shape (px, 3)."""
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float64)
    return rgb.reshape(-1, 3) @ YCBCR.T  # (px,3)


def joint_quad_forms(dirs, skip, data_dir="challenge_data"):
    """Per-frame (A (2n,2n), b (2n), c, pos) for exact RGB MSE.

    u = [w_y; w_c]; channel p uses w_y if p==0 else w_c.
    rgb_mse(u) = (u'Au - 2b'u + c)/3
    """
    seq = load_sequence(data_dir, "validation", skip, "room1")
    n = len(dirs)
    blk = [0] + [1] * 2  # channel -> weight block (Y->0, Cb/Cr->1)
    A_l, b_l, c_l, pos_l = [], [], [], []
    for f in seq.todos:
        G = load_ycc(seq.root / "gt" / f"{f.index:06d}_img.jpg")  # (px,3)
        P = np.stack([load_ycc(Path(d) / skip / "room1" / f"{f.index:06d}.png")
                      for d in dirs])  # (n,px,3)
        npx = P.shape[1]
        A = np.zeros((2 * n, 2 * n))
        b = np.zeros(2 * n)
        c = 0.0
        # Accumulate the quadratic form over all YCbCr channel pairs (p, q).
        # Q[p, q] weights how errors in channel p interact with channel q
        # after the YCbCr -> RGB back-transform; the (p, q) Gram blocks land
        # in the (blk[p], blk[q]) sub-block of A because Y uses weight block 0
        # and Cb/Cr share weight block 1.
        for p in range(3):
            for q in range(3):
                qpq = Q[p, q]
                if qpq == 0.0:
                    continue
                Gpq = (P[:, :, p] @ P[:, :, q].T) / npx  # (n,n)
                rp, rq = blk[p] * n, blk[q] * n
                A[rp:rp + n, rq:rq + n] += qpq * Gpq
                gp = (P[:, :, p] @ G[:, q]) / npx  # (n,)
                b[rp:rp + n] += qpq * gp
                c += qpq * float(G[:, p] @ G[:, q]) / npx
        A_l.append(A)
        b_l.append(b)
        c_l.append(c)
        pos_l.append(folded_position(seq, f))
    return np.stack(A_l), np.stack(b_l), np.array(c_l), np.array(pos_l)


def mean_psnr(u, A, b, c):
    """Mean per-frame RGB PSNR for weight vector u, evaluated purely from the
    precomputed quadratic forms (no image I/O) — this is what makes the search
    fast enough for multi-start coordinate ascent."""
    mse = (np.einsum("i,fij,j->f", u, A, u) - 2 * b @ u + c) / 3.0
    mse = np.maximum(mse, 1e-9)
    return float(np.mean(20 * np.log10(255.0) - 10 * np.log10(mse)))


def coord_ascent(u, A, b, c, n, deltas=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005)):
    """Pairwise coordinate ascent on the two probability simplices.

    Each move transfers step-size d of weight mass from coordinate i to j
    within the same block (Y block or C block), so each block's weights keep
    summing to 1. Step sizes shrink coarse -> fine; each level loops until no
    single transfer improves mean PSNR.
    """
    u = u.copy()
    best = mean_psnr(u, A, b, c)
    for d in deltas:
        improved = True
        while improved:
            improved = False
            for blk0 in (0, n):  # move mass only within a block
                for i in range(blk0, blk0 + n):
                    for j in range(blk0, blk0 + n):
                        # skip no-op moves and donors without d mass to give
                        if i == j or u[i] < d - 1e-12:
                            continue
                        u[i] -= d
                        u[j] += d
                        v = mean_psnr(u, A, b, c)
                        if v > best + 1e-7:
                            best = v
                            improved = True
                        else:  # no gain: undo the transfer
                            u[i] += d
                            u[j] -= d
    return u, best


def optimize_bucket(A, b, c, dirs, warm_uy_uc):
    """Multi-start optimization of one position bucket: run coordinate ascent
    from uniform weights, from the warm-start (previous-recipe) weights, and
    from the 3 best single-member solutions; keep the best result."""
    n = len(dirs)
    starts = []
    uni = np.full(n, 1.0 / n)
    starts.append(np.concatenate([uni, uni]))
    for uy, uc in warm_uy_uc:
        starts.append(np.concatenate([uy, uc]))
    # singleton starts on the 3 best single members (by Y-block psnr)
    singles = []
    for i in range(n):
        e = np.zeros(2 * n)
        e[i] = 1.0
        e[n + i] = 1.0
        singles.append((mean_psnr(e, A, b, c), e))
    singles.sort(key=lambda t: -t[0])
    starts.extend(e for _, e in singles[:3])

    best_u, best_v = None, -1e18
    for s in starts:
        u, v = coord_ascent(s, A, b, c, n)
        if v > best_v:
            best_u, best_v = u, v
    return best_u, best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-recipe", default=str(ROOT / "recipe_v10.json"))
    ap.add_argument("--out", default=str(ROOT / "recipe_v11.json"))
    ap.add_argument("--add", nargs=2, action="append", default=[],
                    metavar=("SKIP", "DIR"), help="add member dir to skip pool")
    ap.add_argument("--skips", nargs="+",
                    default=["1skip", "3skip", "7skip", "15skip"])
    args = ap.parse_args()

    base = json.load(open(args.base_recipe))
    recipe = {}
    grand_num, grand_cnt, grand_base_num = 0.0, 0, 0.0
    for skip in args.skips:
        buckets = base[skip]
        # Member pool = every dir referenced anywhere in the base recipe for
        # this skip (older recipes used a single "weights" key before the
        # Y/C split), plus any --add injections.
        pool = sorted({d for bk in buckets
                       for d in (set(bk.get("weights_y", {})) |
                                 set(bk.get("weights_c", {})) |
                                 set(bk.get("weights", {})))})
        for sk, d in args.add:
            if sk == skip and d not in pool:
                pool.append(d)
        n = len(pool)
        print(f"\n=== {skip}: {n} members ===", flush=True)
        A, b, c, pos = joint_quad_forms(pool, skip)
        out_buckets = []
        for bk in buckets:
            # Select the validation frames whose folded position falls in
            # this bucket; optimize weights on those frames only.
            m = (pos >= bk["lo"]) & (pos < bk["hi"])
            if m.sum() == 0:
                continue
            # Warm start from the base recipe (zero for newly added members),
            # renormalized so each block sums to 1.
            wy0 = np.array([bk.get("weights_y", bk.get("weights", {})).get(d, 0.0)
                            for d in pool])
            wc0 = np.array([bk.get("weights_c", bk.get("weights", {})).get(d, 0.0)
                            for d in pool])
            wy0 /= max(wy0.sum(), 1e-9)
            wc0 /= max(wc0.sum(), 1e-9)
            u0 = np.concatenate([wy0, wc0])
            v0 = mean_psnr(u0, A[m], b[m], c[m])
            u, v = optimize_bucket(A[m], b[m], c[m], pool, [(wy0, wc0)])
            print(f"  [{bk['lo']:.3f},{bk['hi']:.3f}) {int(m.sum())}f: "
                  f"v10={v0:.4f} -> v11={v:.4f} (+{v - v0:.4f})", flush=True)
            # Store sparse weights: drop near-zero members to keep the recipe
            # small and the final blend cheap to evaluate.
            wy = {d: round(float(u[i]), 4) for i, d in enumerate(pool) if u[i] > 1e-4}
            wc = {d: round(float(u[n + i]), 4) for i, d in enumerate(pool)
                  if u[n + i] > 1e-4}
            top = sorted(wy.items(), key=lambda t: -t[1])[:4]
            print(f"    Y: {top}", flush=True)
            out_buckets.append({"lo": bk["lo"], "hi": bk["hi"],
                                "weights_y": wy, "weights_c": wc})
            grand_num += v * m.sum()
            grand_base_num += v0 * m.sum()
            grand_cnt += int(m.sum())
        recipe[skip] = out_buckets
    print(f"\nOVERALL (frame-weighted, pre-static-blend): "
          f"v10={grand_base_num / grand_cnt:.4f} -> v11={grand_num / grand_cnt:.4f}")
    json.dump(recipe, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
