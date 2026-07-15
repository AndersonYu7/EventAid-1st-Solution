#!/usr/bin/env python3
"""Build the FINAL 1st-place submission for the EventAid-F event-aided video
frame interpolation challenge (Codabench 16375, EBMV @ ECCV 2026, team yunyu8).

Role in the pipeline
--------------------
This is the very last stage of the pipeline
(data prep -> training -> member inference -> fusion -> **submission**):
it takes the already-fused v16 ensemble output (produced by
scripts/make_submission.py with recipe_v16.json from the 28-member ensemble)
and layers two final, "paper-clean" specialist blends on top before packaging
the uint8 PNG zip that was uploaded to Codabench
(official scores: validation 43.1056 dB / 0.9802 SSIM, test 31.8835 dB / 0.9088).

Final submission = v16 ensemble + two paper-clean gains:

  * VAL  (room1 7/15skip): per-position-bucket blend of v16 with the REAL-event
    member (realevs step2000), weights chosen on room1 via blend_opt.
    NOTE: this branch is DISABLED below (ROOM1_W = {}) because the float gain
    does not survive uint8 PNG quantization; room1 stays pure v16.
  * TEST (ball/traffic 7/15skip): blend of v16 with the ERF-X170FPS fast-motion
    specialist (v1 @7skip, v2 @15skip) at held-out-selected w=0.85, extended to
    a 3-way fusion with the CBMNet-Large diversity member at fixed weight 0.20.
  * TEST (7 medium-motion scenes): per-pixel flow-adaptive blend toward the ERF
    specialist where Farneback optical-flow magnitude is high.

Everything else stays v16. The script then runs the official validation
evaluator and zips the result.

Inputs (all under the project root, see path constants below)
  * submission_v16/results/            fused v16 ensemble PNGs (base prediction)
  * results_test_v1_8way/, results_test_v2_8way/   8-way TTA ERF specialist PNGs
  * results_cbmL_erf_test/             CBMNet-Large-ERF member PNGs
  * challenge_data/                    official frames/events + evaluator script

Outputs
  * submission_final/results/{skip}/{seq}/{idx}.png   final uint8 frames
  * submission_final.zip                              the file uploaded to Codabench

Usage
-----
    python3 scripts/build_final.py

(No CLI arguments; all inputs/weights are frozen as constants below so the
winning submission is exactly reproducible.)
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/ubuntu-5th/work/EventAid")
sys.path.insert(0, str(ROOT))
from evlib.dataset import load_sequence  # noqa: E402  (project-local dataset loader)

V16 = ROOT / "submission_v16/results"   # base prediction: fused v16 ensemble PNGs
OUT = ROOT / "submission_final"         # output directory for the final submission
RES = OUT / "results"                   # final frames land here, then get zipped

# --- VAL: room1 realevs per-bucket weights (realevs fraction) from blend_opt ---
# buckets over folded position [0,0.17),[0.17,0.33),[0.33,0.5]
# NOTE: the realevs val gain (+0.021 in float) is BELOW the 8-bit quantization
# floor (realevs-v16 differ ~0.3 grey levels; 30% blend ~0.09 -> rounds away in
# uint8 PNG). It is NOT a real submittable gain, so room1 stays pure v16.
ROOM1_REALEVS = "results_val_realevs2k"
ROOM1_W = {}  # disabled: gain doesn't survive uint8 quantization
BUCKET_EDGES = [0.0, 1/6, 1/3, 0.5 + 1e-9]
# --- TEST: per-pixel flow-adaptive ERF blend on ALL scenes (unified method) ---
# ball/traffic and the 7 medium scenes all use the same rule: ERF where Farneback
# flow is fast, synthetic where slow. Set USE_PIXEL_FAST=False to revert ball/
# traffic to whole-frame w=0.85.
USE_PIXEL_FAST = False  # whole-frame ERF on ball/traffic beats per-pixel (all-fast)
# 8-way self-ensemble ERF predictions (all 9 scenes in each dir)
ERF = {"7skip": "results_test_v1_8way", "15skip": "results_test_v2_8way"}
ERF_W = 0.85
FAST = set() if USE_PIXEL_FAST else {"ball", "traffic"}
MEDIUM = {"building", "wall", "sculpture", "room2", "blocks", "umbrella", "playball"}
if USE_PIXEL_FAST:
    MEDIUM = MEDIUM | {"ball", "traffic"}
ERF_MED = {"7skip": "results_test_v1_8way", "15skip": "results_test_v2_8way"}
ERF_MED_BT = {"7skip": "results_test_v1_8way", "15skip": "results_test_v2_8way"}
FLOW_K, FLOW_S = 4.0, 2.0
# CBMNet-Large-ERF (CVPR23, ERF-X170FPS weights): a decorrelated event-VFI member.
# Blended into ball/traffic at a fixed principled weight (NOT per-scene test-tuned).
CBM_DIR = "results_cbmL_erf_test"
CBM_W = 0.30
CBM_SCENES = {"ball", "traffic"}
# Joint 3-way fusion for ball/traffic: (v16, ERF-EMA-E specialist, CBMNet).
# LEGAL weights (no test GT): CBMNet gets a FIXED conservative diversity weight 0.20
# (standard "small weight for an independent-architecture member" — NOT tuned on
# test); the remaining 0.80 splits ERF/base by the held-out-selected 0.85/0.15 ratio
# -> (base 0.12, ERF 0.68, CBMNet 0.20). Fully reproducible; see report limitation.
THREEWAY = (0.12, 0.68, 0.20)
import cv2  # noqa: E402
cv2.setNumThreads(4)


def load(p):
    """Load a PNG as float64 RGB (float precision needed for weighted blends)."""
    return np.asarray(Image.open(p).convert("RGB"), np.float64)


def folded_pos(seq, f):
    """Normalized temporal position of frame `f` inside its anchor gap,
    folded to [0, 0.5] (symmetric around the gap midpoint), used to pick a
    position bucket for the per-bucket blend weights."""
    inp = seq.inputs
    # nearest input (anchor) frames strictly before/after the target frame
    prev = max((x for x in inp if x.index < f.index), key=lambda x: x.index)
    nxt = min((x for x in inp if x.index > f.index), key=lambda x: x.index)
    a = (f.timestamp - prev.timestamp) / (nxt.timestamp - prev.timestamp)
    return min(a, 1 - a)  # fold: position 0.8 behaves like 0.2


def save(arr, dst):
    """Round + clip a float image to uint8 and save as the submission PNG."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB").save(dst)


def main():
    if RES.exists():
        shutil.rmtree(RES)  # always rebuild from scratch
    # position lookup for room1 (per split it's validation)
    n = 0
    # Walk every skip level and both splits; each sequence decides which of the
    # three specialist branches (room1 / fast / medium) applies, else pure v16.
    for skip in ("1skip", "3skip", "7skip", "15skip"):
        for split in ("validation", "test"):
            base = ROOT / "challenge_data" / split / skip
            if not base.exists():
                continue
            for seqp in sorted(p for p in base.iterdir() if p.is_dir()):
                seq = load_sequence(ROOT / "challenge_data", split, skip, seqp.name)
                # branch flags for this (scene, skip) pair
                room1 = (seqp.name == "room1" and skip in ROOM1_W)
                fast = (seqp.name in FAST and skip in ERF)
                medium = (seqp.name in MEDIUM and skip in ERF_MED)
                # anchors sorted by index, needed for the flow-adaptive branch
                inp_sorted = sorted(seq.inputs, key=lambda f: f.index) if medium else None
                pos_map = {}
                if room1:
                    # precompute folded temporal position for each target frame
                    for f in seq.todos:
                        pos_map[f.index] = folded_pos(seq, f)
                # iterate over the already-fused v16 frames for this sequence
                for v16p in sorted(glob.glob(str(V16 / skip / seqp.name / "*.png"))):
                    idx = os.path.basename(v16p)[:-4]  # zero-padded frame index
                    dst = RES / skip / seqp.name / f"{idx}.png"
                    base_img = load(v16p)
                    if room1:
                        # VAL branch (currently disabled via ROOM1_W = {}):
                        # blend v16 with the real-event member, weight chosen
                        # by the frame's folded-position bucket.
                        cp = ROOT / ROOM1_REALEVS / skip / "room1" / f"{idx}.png"
                        if cp.exists():
                            p = pos_map.get(int(idx), 0.0)
                            bk = next(k for k in range(3)
                                      if BUCKET_EDGES[k] <= p < BUCKET_EDGES[k + 1])
                            w = ROOM1_W[skip][bk]
                            base_img = (1 - w) * base_img + w * load(cp)
                    elif fast:
                        # TEST fast-motion branch (ball/traffic @7/15skip):
                        # joint 3-way fusion of (v16 base, ERF specialist,
                        # CBMNet-Large) at fixed weights THREEWAY; fall back to
                        # the 2-way 0.15/0.85 blend if CBMNet output is absent.
                        cp = ROOT / ERF[skip] / skip / seqp.name / f"{idx}.png"
                        cb = ROOT / CBM_DIR / skip / seqp.name / f"{idx}.png"
                        if cp.exists() and cb.exists():       # joint 3-way fusion
                            base_img = (THREEWAY[0] * base_img + THREEWAY[1] * load(cp)
                                        + THREEWAY[2] * load(cb))
                        elif cp.exists():
                            base_img = (1 - ERF_W) * base_img + ERF_W * load(cp)
                    elif medium:
                        # TEST medium-motion branch: per-pixel flow-adaptive
                        # blend — lean on the ERF specialist only where motion
                        # between the two anchor frames is fast.
                        src = ERF_MED_BT if seqp.name in ("ball", "traffic") else ERF_MED
                        cp = ROOT / src[skip] / skip / seqp.name / f"{idx}.png"
                        if cp.exists():
                            erf = load(cp)
                            # anchor frames bracketing this target index
                            prev = max((x for x in inp_sorted if x.index < int(idx)),
                                       key=lambda x: x.index)
                            nxt = min((x for x in inp_sorted if x.index > int(idx)),
                                      key=lambda x: x.index)
                            a0 = seq.load_frame(prev).astype(np.uint8)
                            a1 = seq.load_frame(nxt).astype(np.uint8)
                            # dense Farneback optical flow between the anchors
                            fl = cv2.calcOpticalFlowFarneback(
                                cv2.cvtColor(a0, cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor(a1, cv2.COLOR_RGB2GRAY),
                                None, 0.5, 3, 25, 3, 5, 1.2, 0)
                            mag = np.linalg.norm(fl, axis=2)
                            # sigmoid gate: flow magnitude -> per-pixel ERF weight
                            # (midpoint FLOW_K px, softness FLOW_S px)
                            w = (1 / (1 + np.exp(-(mag - FLOW_K) / FLOW_S)))[..., None]
                            base_img = w * erf + (1 - w) * base_img
                    # (CBMNet is now folded into the ball/traffic 3-way fusion above)
                    save(base_img, dst)
                    n += 1
    print(f"wrote {n} frames -> {RES}")

    # Sanity check 1: run the challenge's official validation evaluator on the
    # assembled frames (this is the exact script Codabench runs on val).
    r = subprocess.run(
        [sys.executable, str(ROOT / "challenge_data/evaluate_results.py"),
         "--data-dir", str(ROOT / "challenge_data"), "--results-dir", str(RES)],
        capture_output=True, text=True)
    print("=== OFFICIAL VALIDATION ===")
    print(r.stdout[-800:])
    if r.returncode != 0:
        print(r.stderr[-500:])

    # Package the Codabench upload: PNGs stored uncompressed (ZIP_STORED),
    # archive paths rooted at results/ as the submission format requires.
    zp = OUT.with_suffix(".zip")
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(RES.rglob("*.png")):
            z.write(f, f.relative_to(OUT))
    print(f"wrote {zp} ({zp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
