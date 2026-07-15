#!/usr/bin/env python3
"""Hierarchical (dyadic) TimeLens interpolation for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. This hierarchical TimeLens variant is a
  separate member of the 28-member ensemble, complementary to the direct
  runner (run_timelens.py) at large skips. Pipeline: data prep -> training
  -> member inference (THIS script) -> per-frame quadratic-form fusion
  (recipe_v15/v16.json) -> submission assembly (scripts/build_final.py).

Method:
  Instead of synthesizing every TODO frame directly from the two real anchors
  (motion over up to 16 base intervals), recursively bisect: predict the middle
  frame first, then use it as a pseudo-anchor for the quarter positions, etc.
  Each stage only warps over half the previous temporal span. --anchor-dir
  substitutes materialized ensemble frames as pseudo-anchors (anchored
  bisection); --tta averages 4 full hierarchical flip passes.

Inputs:
  --data-dir     challenge data root; --checkpoint TimeLens weights (.bin)
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame,
  consumed by the fusion stage.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_timelens_hier.py \
      --splits validation --skips 15skip --output-dir results_timelens_hier
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "rpg_timelens"))

from evlib.dataset import SKIPS, iter_sequences, save_result_png  # noqa: E402

from timelens import attention_average_network  # noqa: E402
from timelens.common import pytorch_tools, transformers  # noqa: E402
from timelens.common.event import EventSequence  # noqa: E402

FLIP_DIMS = {"none": (), "h": (1,), "v": (0,)}


def load_network(checkpoint_file: str):
    network = attention_average_network.AttentionAverage()
    network.from_legacy_checkpoint(checkpoint_file)
    network.cuda()
    network.eval()
    return network


def make_event_sequence(seq, t0, t1, flip="none"):
    # Challenge events [t,x,y,p(0/1)] -> TimeLens EventSequence [x,y,t,p(-1/1)],
    # with coordinates mirrored to match the TTA flip of the images.
    ev = seq.load_events(t0, t1)
    feats = np.empty((ev.shape[0], 4), dtype=np.float64)
    if ev.shape[0]:
        feats[:, 0] = ev[:, 1]            # x
        feats[:, 1] = ev[:, 2]            # y
        feats[:, 2] = ev[:, 0]            # t (microseconds)
        feats[:, 3] = ev[:, 3] * 2.0 - 1  # p {0,1} -> {-1,+1}
        if "h" in flip:
            feats[:, 0] = seq.width - 1 - feats[:, 0]
        if "v" in flip:
            feats[:, 1] = seq.height - 1 - feats[:, 1]
    return EventSequence(feats, seq.height, seq.width, start_time=t0, end_time=t1)


@torch.no_grad()
def synth(network, transform_list, seq, left_img, right_img, t_left, t_right,
          t_target, flip):
    """One TimeLens forward at exact t_target between (possibly pseudo) anchors."""
    full = make_event_sequence(seq, t_left, t_right, flip)
    left_ev, right_ev = full.split_in_two(float(t_target))
    weight = (t_target - t_left) / (t_right - t_left)
    example = {
        "before": {"rgb_image": left_img, "events": left_ev},
        "middle": {"weight": float(weight)},
        "after": {"rgb_image": right_img, "events": right_ev},
    }
    example = transformers.apply_transforms(example, transform_list)
    example = transformers.collate([example])
    example = pytorch_tools.move_tensors_to_cuda(example)
    frame, _ = network.run_fast(example)
    img = torch.clamp(frame.squeeze(0), 0, 1).cpu().numpy()
    return np.transpose(img, (1, 2, 0))


def hier(network, tl, seq, left_img, right_img, t_left, t_right, lo, hi,
         frames_by_idx, results, flip, anchors=None):
    """Recursively fill TODO frames with indices in (lo, hi).

    anchors: optional {index: unflipped PIL image} of externally materialized
    frames (e.g. current best ensemble); when present they replace the model's
    own midpoint output as the pseudo-anchor for deeper levels. Outputs stay
    the model's own predictions.
    """
    if hi - lo < 2:
        return
    mid = (lo + hi) // 2
    t_mid = frames_by_idx[mid]
    # Predict the midpoint of this (sub-)interval and record it.
    arr = synth(network, tl, seq, left_img, right_img, t_left, t_right, t_mid, flip)
    results[mid] = results.get(mid, []) + [arr]
    # Pick the pseudo-anchor for the two half-intervals: an externally
    # materialized ensemble frame when available, else our own prediction.
    if anchors and mid in anchors:
        mid_img = flip_image(anchors[mid], flip)
    else:
        mid_img = Image.fromarray(np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8), "RGB")
    hier(network, tl, seq, left_img, mid_img, t_left, t_mid, lo, mid,
         frames_by_idx, results, flip, anchors)
    hier(network, tl, seq, mid_img, right_img, t_mid, t_right, mid, hi,
         frames_by_idx, results, flip, anchors)


def flip_image(img, flip):
    # Apply the TTA spatial flip to an input PIL image.
    if "h" in flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if "v" in flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def unflip_array(arr, flip):
    # Invert the TTA flip on the network output (flips are involutions).
    if "h" in flip:
        arr = arr[:, ::-1]
    if "v" in flip:
        arr = arr[::-1, :]
    return np.ascontiguousarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--skips", nargs="+", default=["7skip", "15skip"])
    parser.add_argument("--output-dir", default=str(ROOT / "results_timelens_hier"))
    parser.add_argument("--checkpoint",
                        default=str(ROOT / "weights" / "timelens_checkpoint.bin"))
    parser.add_argument("--tta", action="store_true",
                        help="average h/v/hv flips (4 hierarchical passes)")
    parser.add_argument("--anchor-dir", default=None,
                        help="materialized ensemble root ({skip}/{seq}/{idx}.png) "
                             "used as pseudo-anchors for deeper levels")
    args = parser.parse_args()

    pytorch_tools.set_fastest_cuda_mode()
    tl = transformers.initialize_transformers(number_of_bins_in_voxel_grid=5)
    network = load_network(args.checkpoint)
    print(f"loaded {args.checkpoint}", flush=True)

    flips = ["none", "h", "v", "hv"] if args.tta else ["none"]
    total = 0
    t0_all = time.time()
    for seq in iter_sequences(args.data_dir, splits=args.splits, skips=args.skips):
        t_seq = time.time()
        inputs = seq.inputs
        ts_by_idx = {f.index: f.timestamp for f in seq.frames}
        todo_idx = {f.index for f in seq.todos}
        n = 0
        results_accum = {}  # index -> list of per-flip predictions (unflipped)
        for flip in flips:
            for a, b in zip(inputs[:-1], inputs[1:]):
                gap = [i for i in range(a.index + 1, b.index) if i in todo_idx]
                if not gap:
                    continue  # no TODO frames between this anchor pair
                left = flip_image(Image.open(seq.root / a.path).convert("RGB"), flip)
                right = flip_image(Image.open(seq.root / b.path).convert("RGB"), flip)
                anchors = None
                if args.anchor_dir:
                    anchors = {}
                    for i in range(a.index + 1, b.index):
                        p = (Path(args.anchor_dir) / seq.skip / seq.name
                             / f"{i:06d}.png")
                        if p.exists():
                            anchors[i] = Image.open(p).convert("RGB")
                results = {}
                hier(network, tl, seq, left, right,
                     a.timestamp, b.timestamp, a.index, b.index,
                     ts_by_idx, results, flip, anchors)
                for idx, arrs in results.items():
                    arr = unflip_array(arrs[0], flip)
                    results_accum.setdefault(idx, []).append(arr)
        # Average the flip variants (TTA) and write only requested frames.
        for idx, arrs in sorted(results_accum.items()):
            if idx not in todo_idx:
                continue
            avg = np.mean(arrs, axis=0)
            save_result_png(args.output_dir, seq.skip, seq.name, idx, avg)
            n += 1
        total += n
        print(f"[{seq.split}/{seq.skip}/{seq.name}] {n} frames "
              f"in {time.time() - t_seq:.1f}s", flush=True)
    print(f"DONE: {total} frames in {time.time() - t0_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
