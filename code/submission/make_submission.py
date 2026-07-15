#!/usr/bin/env python3
"""Assemble a challenge submission by fusing per-member prediction directories
and packaging the result as the uint8 PNG zip expected by Codabench.

Role in the pipeline
--------------------
This script implements the **fusion -> submission** stages of the 1st-place
EventAid-F pipeline (Codabench 16375, team yunyu8):
data prep -> training -> member inference -> **fusion** -> **submission**.
Each ensemble member (EMA-VFI/EMA-E, RIFE, GIMM-VFI, VFIMamba, TimeLens,
TimeLens-XL, REFID, CBMNet, our fusion refiners, ...) has already written its
per-frame predictions to its own results_* directory during member inference;
this script combines them per skip level — either with simple global weights
(--pick) or with the position-bucketed recipe JSONs (--recipe recipe_v15.json /
recipe_v16.json) that define the winning per-frame quadratic-form fusion —
then runs the official format check + validation scoring and zips the frames.
(The very final submission additionally layers specialist blends on top of the
v16 output via scripts/build_final.py.)

Fusion modes
  * --pick    skip=dir[:w][,dir:w...]   one global weight vector per skip level
  * --recipe  JSON with per-skip position buckets over the folded temporal
              position [0, 0.5]; each bucket carries either RGB "weights" or
              separate luma/chroma "weights_y"/"weights_c" (YCbCr fusion)
  * --fast-recipe / --fast-scenes       alternative recipe for named high-motion
              scenes (event-density-routed hedge, e.g. ball,traffic)
  * --static-blend                      per-pixel pull toward linear anchor
              interpolation in static regions

Inputs
  * challenge_data/ (or --data-dir): official sequences (anchor frames, todo
    lists, timestamps) plus the organizers' evaluate_results.py
  * one or more member results dirs: {dir}/{skip}/{seq}/{idx:06d}.png

Outputs
  * {out}/results/{skip}/{seq}/{idx:06d}.png for BOTH splits (val + test)
  * {out}.zip rooted at results/, ready to upload to Codabench

Usage:
  python3 scripts/make_submission.py \
      --pick 1skip=results_rife \
      --pick 3skip=results_rife \
      --pick "7skip=results_rife:0.4,results_timelens:0.4,results_cbmnet_bsergb:0.2" \
      --pick "15skip=results_rife:0.4,results_timelens:0.4,results_cbmnet_bsergb:0.2" \
      --out submission

Writes {out}/results/{skip}/{seq}/{idx:06d}.png for BOTH splits, runs the
official format check + validation scoring, then zips rooted at results/.
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evlib.dataset import iter_sequences  # noqa: E402  (project-local dataset loader)

SKIPS = ("1skip", "3skip", "7skip", "15skip")

# ITU-R BT.601 RGB -> YCbCr matrix; lets a recipe weight luma and chroma
# independently (weights_y / weights_c), which the v15/v16 recipes exploit.
YCBCR = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
YCBCR_INV = np.linalg.inv(YCBCR)


def folded_position(seq, todo):
    """Normalized distance into the anchor gap, folded to [0, 0.5]."""
    inputs = seq.inputs
    # nearest anchor frames strictly before/after the frame to interpolate
    prev = max((f for f in inputs if f.index < todo.index), key=lambda f: f.index)
    nxt = min((f for f in inputs if f.index > todo.index), key=lambda f: f.index)
    a = (todo.timestamp - prev.timestamp) / (nxt.timestamp - prev.timestamp)
    return min(a, 1.0 - a)  # fold: position 0.8 behaves like 0.2


def parse_pick(text):
    """Parse one --pick argument "skip=dir[:w][,dir:w...]" into
    (skip, [(results_dir, weight), ...]); weights must sum to 1."""
    skip, spec = text.split("=", 1)
    assert skip in SKIPS, skip
    parts = []
    for item in spec.split(","):
        if ":" in item:
            d, w = item.rsplit(":", 1)  # rsplit so dir names may contain ':'
            parts.append((ROOT / d, float(w)))
        else:
            parts.append((ROOT / item, 1.0))  # bare dir means weight 1.0
    total = sum(w for _, w in parts)
    assert abs(total - 1.0) < 1e-6, f"weights for {skip} sum to {total}"
    return skip, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--pick", action="append",
                    help="skip=dir:w,dir:w (global per-skip weights)")
    ap.add_argument("--recipe", type=Path,
                    help="JSON: {skip: [{lo, hi, weights: {dir: w}}, ...]} — "
                         "position-bucketed weights on folded position [0, 0.5]")
    ap.add_argument("--out", default=str(ROOT / "submission"))
    ap.add_argument("--static-blend", action="store_true",
                    help="per-pixel blend toward linear interp in static regions "
                         "(w = exp(-boxblur(|a0-a1|, 15)))")
    ap.add_argument("--fast-recipe", type=Path,
                    help="alt recipe applied only to --fast-scenes (event-density "
                         "hedge for high-motion test scenes)")
    ap.add_argument("--fast-scenes", default="",
                    help="comma-sep scene names that use --fast-recipe (e.g. ball,traffic)")
    args = ap.parse_args()

    def _norm(rec):
        """Validate a recipe JSON and renormalize each bucket's weights to
        exactly 1.0 (buckets may use joint RGB "weights" or split
        "weights_y"/"weights_c" for independent luma/chroma fusion)."""
        for skip, buckets in rec.items():
            assert skip in SKIPS
            for bk in buckets:
                for ks in (["weights_y", "weights_c"] if "weights_y" in bk else ["weights"]):
                    total = sum(bk[ks].values())
                    assert abs(total - 1.0) < 0.01, f"{skip} {ks} sum {total}"
                    bk[ks] = {k: v / total for k, v in bk[ks].items()}
        return rec

    fast_recipe = _norm(json.loads(args.fast_recipe.read_text())) if args.fast_recipe else None
    fast_scenes = set(s for s in args.fast_scenes.split(",") if s)
    # --recipe and --pick are mutually exclusive fusion modes
    if args.recipe:
        recipe = _norm(json.loads(args.recipe.read_text()))
        picks = None
    else:
        recipe = None
        picks = dict(parse_pick(p) for p in args.pick or [])
        missing_skips = [s for s in SKIPS if s not in picks]
        assert not missing_skips, f"no pick for {missing_skips}"

    out = Path(args.out)
    results = out / "results"
    if results.exists():
        shutil.rmtree(results)

    n_written, n_missing = 0, []
    # Main fusion loop: for every frame the challenge asks us to interpolate
    # ("todo"), resolve which member dirs + weights apply, blend, and save.
    for seq in iter_sequences(args.data_dir):
        for f in seq.todos:
            ycbcr_bucket = None
            # fast scenes may be routed to an alternative recipe (--fast-recipe)
            cur_recipe = (fast_recipe if (fast_recipe is not None
                          and seq.name in fast_scenes) else recipe)
            if cur_recipe is not None:
                # recipe mode: select the position bucket containing this
                # frame's folded temporal position (last bucket owns pos=0.5)
                pos = folded_position(seq, f)
                bucket = next(bk for bk in cur_recipe[seq.skip]
                              if bk["lo"] <= pos < bk["hi"] or
                              (pos >= 0.5 - 1e-9 and bk["hi"] >= 0.5))
                if "weights_y" in bucket:
                    # luma/chroma split weights: load every dir referenced by
                    # either channel set; actual weighting happens later
                    ycbcr_bucket = bucket
                    sources = [(ROOT / d, 1.0) for d in
                               set(bucket["weights_y"]) | set(bucket["weights_c"])]
                else:
                    sources = [(ROOT / d, w) for d, w in bucket["weights"].items()
                               if w > 0]
            else:
                sources = picks[seq.skip]  # --pick mode: global per-skip weights
            rel = Path(seq.skip) / seq.name / f"{f.index:06d}.png"
            srcs = [d / rel for d, _ in sources]
            if any(not s.exists() for s in srcs):
                # record every missing member prediction; fail at the end
                n_missing.extend(str(s) for s in srcs if not s.exists())
                continue
            dst = results / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # fast path: single source, no post-processing -> byte-exact copy
            if len(srcs) == 1 and not args.static_blend and ycbcr_bucket is None:
                shutil.copy2(srcs[0], dst)
                n_written += 1
                continue
            if ycbcr_bucket is not None:
                # YCbCr fusion: convert each member frame to YCbCr, then mix
                # the Y plane with weights_y and the Cb/Cr planes with
                # weights_c before converting back to RGB.
                ycc = {}
                for (d, _), s in zip(sources, srcs):
                    key = str(d.relative_to(ROOT))  # recipe keys are ROOT-relative
                    ycc[key] = np.asarray(Image.open(s).convert("RGB"),
                                          dtype=np.float64) @ YCBCR.T
                Y = sum(w * ycc[d][..., 0]
                        for d, w in ycbcr_bucket["weights_y"].items())
                Cb = sum(w * ycc[d][..., 1]
                         for d, w in ycbcr_bucket["weights_c"].items())
                Cr = sum(w * ycc[d][..., 2]
                         for d, w in ycbcr_bucket["weights_c"].items())
                acc = np.stack([Y, Cb, Cr], axis=-1) @ YCBCR_INV.T
            else:
                # plain weighted average of member frames in RGB float space
                acc = None
                for (d, w), s in zip(sources, srcs):
                    img = np.asarray(Image.open(s).convert("RGB"), dtype=np.float64)
                    acc = img * w if acc is None else acc + img * w
            if args.static_blend:
                # In static regions the linear interpolation of the two anchor
                # frames is nearly exact; pull the ensemble output toward it
                # with per-pixel weight exp(-boxblur(|a0-a1|, 15)).
                inputs = seq.inputs
                prev = max((x for x in inputs if x.index < f.index),
                           key=lambda x: x.index)
                nxt = min((x for x in inputs if x.index > f.index),
                          key=lambda x: x.index)
                a0 = np.asarray(Image.open(seq.root / prev.path).convert("RGB"),
                                dtype=np.float64)
                a1 = np.asarray(Image.open(seq.root / nxt.path).convert("RGB"),
                                dtype=np.float64)
                alpha = (f.timestamp - prev.timestamp) / (nxt.timestamp - prev.timestamp)
                lin = (1 - alpha) * a0 + alpha * a1  # time-weighted anchor mix
                # small anchor difference -> static pixel -> wmap near 1
                wmap = np.exp(-ndi.uniform_filter(
                    np.abs(a0 - a1).mean(2), 15))[..., None]
                acc = wmap * lin + (1 - wmap) * acc
            # quantize the float blend to the uint8 PNG the challenge scores
            Image.fromarray(np.clip(np.rint(acc), 0, 255).astype(np.uint8),
                            "RGB").save(dst)
            n_written += 1

    print(f"wrote {n_written} frames; missing {len(n_missing)}")
    for m in n_missing[:20]:
        print("  MISSING:", m)
    if n_missing:
        sys.exit(1)  # an incomplete submission would be rejected — abort

    # Run the organizers' official format check + validation scoring on the
    # assembled frames before packaging anything.
    r = subprocess.run(
        [sys.executable, str(ROOT / "challenge_data" / "evaluate_results.py"),
         "--data-dir", args.data_dir, "--results-dir", str(results)],
        capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)

    # Package the upload: PNGs stored uncompressed (ZIP_STORED, they are
    # already compressed), archive paths rooted at results/ per the format.
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(results.rglob("*.png")):
            z.write(f, f.relative_to(out))
    print(f"wrote {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB, "
          f"{len(list(results.rglob('*.png')))} files)")


if __name__ == "__main__":
    main()
