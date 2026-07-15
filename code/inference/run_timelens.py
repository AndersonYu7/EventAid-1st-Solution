#!/usr/bin/env python3
"""TimeLens (CVPR'21, uzh-rpg/rpg_timelens) runner for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. TimeLens is an event-based member of the
  30-member ensemble; its outputs are also base inputs to our trained
  fusion refiners (run_refiner*.py). Pipeline: data prep -> training ->
  member inference (THIS script) -> per-frame quadratic-form fusion
  (recipe_v15/v16.json) -> submission assembly (scripts/build_final.py).

Method:
  Drives the AttentionAverage network directly per TODO frame (instead of the
  repo CLI), so each frame is synthesized at its EXACT timestamp:
    - events in [t_left, t_right) are split at t_target
    - middle weight = (t_target - t_left) / (t_right - t_left)
  Spatial flips (--flip) and temporal reversal (--treverse) provide TTA
  variants that are averaged externally by the fusion stage. Optional
  classical event denoising (--denoise hp / hp+baf).

Inputs:
  --data-dir     challenge data root (frames + raw events + timestamps)
  --checkpoint   weights/timelens_checkpoint.bin (official release weights)
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_timelens.py \
      --splits validation --skips 1skip 3skip 7skip 15skip \
      --output-dir results_timelens
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
from evlib.denoise import EventDenoiser  # noqa: E402

from timelens import attention_average_network  # noqa: E402
from timelens.common import pytorch_tools, transformers  # noqa: E402
from timelens.common.event import EventSequence  # noqa: E402


def load_network(checkpoint_file: str):
    network = attention_average_network.AttentionAverage()
    network.from_legacy_checkpoint(checkpoint_file)
    network.cuda()
    network.eval()
    return network


def make_event_sequence(seq, t0: float, t1: float, flip: str = "none",
                        treverse: bool = False, denoiser=None) -> EventSequence:
    """Challenge events [t,x,y,p(0/1)] -> TimeLens EventSequence [x,y,t,p(-1/1)].

    treverse: play the interval backwards — t -> t0+t1-t, polarity negated.
    """
    ev = seq.load_events(t0, t1)
    if denoiser is not None:
        ev = denoiser(ev)
    feats = np.empty((ev.shape[0], 4), dtype=np.float64)
    if ev.shape[0]:
        feats[:, 0] = ev[:, 1]            # x (plain integer pixels)
        feats[:, 1] = ev[:, 2]            # y
        feats[:, 2] = ev[:, 0]            # t (microseconds)
        feats[:, 3] = ev[:, 3] * 2.0 - 1  # p {0,1} -> {-1,+1}
        if "h" in flip:
            feats[:, 0] = seq.width - 1 - feats[:, 0]
        if "v" in flip:
            feats[:, 1] = seq.height - 1 - feats[:, 1]
        if treverse:
            feats[:, 2] = t0 + t1 - feats[:, 2]
            feats[:, 3] = -feats[:, 3]
            feats = feats[np.argsort(feats[:, 2], kind="stable")]
    return EventSequence(feats, seq.height, seq.width, start_time=t0, end_time=t1)


def flip_image(img: Image.Image, flip: str) -> Image.Image:
    # Apply the TTA spatial flip to an input PIL image.
    if "h" in flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if "v" in flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def unflip_array(arr: np.ndarray, flip: str) -> np.ndarray:
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
    parser.add_argument("--skips", nargs="+", default=list(SKIPS))
    parser.add_argument("--output-dir", default=str(ROOT / "results_timelens"))
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "weights" / "timelens_checkpoint.bin")
    )
    parser.add_argument("--flip", default="none", choices=["none", "h", "v", "hv"],
                        help="spatially flip inputs (images+events); outputs are unflipped")
    parser.add_argument("--treverse", action="store_true",
                        help="time-reverse the problem (swap anchors, reverse "
                             "events, weight -> 1-weight)")
    parser.add_argument("--denoise", default="none", choices=["none", "hp", "hp+baf"],
                        help="classical event denoising: hot-pixel removal (hp), "
                             "optionally + background activity filter (hp+baf)")
    parser.add_argument("--denoise-tau", type=float, default=5000.0,
                        help="BAF neighbor-support window in microseconds")
    args = parser.parse_args()

    pytorch_tools.set_fastest_cuda_mode()
    transform_list = transformers.initialize_transformers(number_of_bins_in_voxel_grid=5)
    network = load_network(args.checkpoint)
    print(f"Loaded checkpoint {args.checkpoint}", flush=True)

    total_frames = 0
    t_global = time.time()
    for seq in iter_sequences(args.data_dir, splits=args.splits, skips=args.skips):
        print(f"== {seq.split}/{seq.skip}/{seq.name} ({seq.width}x{seq.height}) ==", flush=True)
        inputs = seq.inputs
        todos = seq.todos
        idx_to_input = {f.index: f for f in inputs}
        denoiser = None
        if args.denoise != "none":
            # Fit the classical denoiser (hot-pixel map / background-activity
            # filter) on ALL events of the sequence once, then apply per gap.
            ev_all = seq.load_events(seq.event_files[0].t_start,
                                     seq.event_files[-1].t_end + 1)
            denoiser = EventDenoiser(seq.width, seq.height, mode=args.denoise,
                                     tau=args.denoise_tau).fit(ev_all)
        t_seq = time.time()
        n_seq = 0
        for li in range(len(inputs) - 1):
            left, right = inputs[li], inputs[li + 1]
            gap_todos = [f for f in todos if left.index < f.index < right.index]
            if not gap_todos:
                continue
            left_img = flip_image(Image.open(seq.root / left.path).convert("RGB"), args.flip)
            right_img = flip_image(Image.open(seq.root / right.path).convert("RGB"), args.flip)
            full_events = make_event_sequence(seq, left.timestamp, right.timestamp,
                                              args.flip, args.treverse, denoiser)
            if args.treverse:
                # Temporal-reversal TTA: the whole problem is mirrored in
                # time, so the anchors swap too.
                left_img, right_img = right_img, left_img
            for todo in gap_todos:
                # Attention-average middle weight = normalized target time.
                weight = (todo.timestamp - left.timestamp) / (
                    right.timestamp - left.timestamp
                )
                split_ts = float(todo.timestamp)
                if args.treverse:
                    # Mirror the target time and weight into the reversed axis.
                    weight = 1.0 - weight
                    split_ts = float(left.timestamp + right.timestamp - todo.timestamp)
                left_events, right_events = full_events.split_in_two(split_ts)
                # Assemble the example exactly as the TimeLens repo expects
                # (before/middle/after dict), then run its transform pipeline
                # (voxelization etc.) and one forward pass.
                example = {
                    "before": {"rgb_image": left_img, "events": left_events},
                    "middle": {"weight": float(weight)},
                    "after": {"rgb_image": right_img, "events": right_events},
                }
                example = transformers.apply_transforms(example, transform_list)
                example = transformers.collate([example])
                example = pytorch_tools.move_tensors_to_cuda(example)
                with torch.no_grad():
                    frame, _ = network.run_fast(example)
                img = torch.clamp(frame.squeeze(0), 0, 1).cpu().numpy()
                img = np.transpose(img, (1, 2, 0))  # (H,W,3)
                img = unflip_array(img, args.flip)
                assert img.shape[0] == seq.height and img.shape[1] == seq.width, img.shape
                save_result_png(args.output_dir, seq.skip, seq.name, todo.index, img)
                n_seq += 1
        total_frames += n_seq
        dt = time.time() - t_seq
        print(
            f"  wrote {n_seq} frames in {dt:.1f}s ({dt / max(n_seq, 1):.2f}s/frame)",
            flush=True,
        )
        if denoiser is not None:
            print(f"  denoise: {denoiser.stats()}", flush=True)
    print(
        f"DONE: {total_frames} frames in {time.time() - t_global:.1f}s "
        f"-> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
