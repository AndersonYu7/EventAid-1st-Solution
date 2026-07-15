#!/usr/bin/env python3
"""CBMNet (CVPR'23) inference runner for the EventAid Frame Interpolation Challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. CBMNet-Large is the dedicated DIVERSITY
  member of the 30-member ensemble, blended at a fixed weight of 0.20 in
  the final recipe. Pipeline: data prep -> training -> member inference
  (THIS script) -> per-frame quadratic-form fusion (recipe_v15/v16.json)
  -> submission assembly (scripts/build_final.py). Outputs per TODO frame
  are uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png.

Method — for each [TODO] frame at time t between anchors (t0, t1) it builds three
16-bin voxel grids exactly the way CBMNet's tools/preprocess_events.py does
for BS-ERGB (minus the /32 coordinate scaling and hardcoded resolution):
  0t : events in [t0, t) voxelized forward
  t0 : the same events time-reversed with flipped polarity
  t1 : events in [t, t1) voxelized forward
and runs one whole-frame forward pass (zero-pad to multiple of 64 inside
model.set_test_input, cropped back by forward_joint_test). Falls back to the
repo's overlapping tiled path on CUDA OOM.

Example:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_cbmnet.py \
      --ckpt weights/ours_large_bsergb.pth --model-name ours_large \
      --splits validation --skips 1skip 3skip 7skip 15skip \
      --output-dir results_cbmnet_bsergb
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CBMNET = os.path.join(ROOT, "models", "CBMNet")
sys.path.insert(0, ROOT)
sys.path.insert(0, CBMNET)
sys.path.insert(0, os.path.join(CBMNET, "tools"))

from evlib.dataset import SKIPS, load_sequence, save_result_png  # noqa: E402
from event_utils import event_reverse, events_to_voxel_grid  # noqa: E402
from models.model_manager import OurModel  # noqa: E402

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

NUM_BINS = 16


def normalize_voxel(v, mode, target):
    """Rescale voxel magnitude to a fixed robust scale (returns a new array)."""
    if mode == "none":
        return v
    if mode == "robust98":
        nz = v[v != 0]
        if nz.size == 0:
            return v
        scale = np.percentile(np.abs(nz), 98)
    elif mode == "unitstd":
        scale = v.std()
    else:
        raise ValueError(mode)
    return (v / (scale + 1e-9) * target).astype(np.float32)


def make_voxels(seq, t0, t, t1):
    """Build (0t, t0, t1) voxel grids, float32 (16, H, W).

    0t: events from left anchor to target, forward in time;
    t0: same events time-reversed with flipped polarity (target -> left);
    t1: events from target to right anchor, forward. These are the three
    inputs CBMNet's cross-bidirectional-motion estimator expects.
    """
    H, W = seq.height, seq.width
    ev_0t = seq.load_events(t0, t)
    ev_t1 = seq.load_events(t, t1)
    zeros = np.zeros((NUM_BINS, H, W), dtype=np.float32)
    if ev_0t.shape[0] > 0:
        vox_0t = events_to_voxel_grid(ev_0t.copy(), NUM_BINS, W, H)
        vox_t0 = events_to_voxel_grid(event_reverse(ev_0t.copy()), NUM_BINS, W, H)
    else:
        vox_0t, vox_t0 = zeros, zeros
    if ev_t1.shape[0] > 0:
        vox_t1 = events_to_voxel_grid(ev_t1.copy(), NUM_BINS, W, H)
    else:
        vox_t1 = zeros
    return vox_0t, vox_t0, vox_t1


def forward_whole(model, sample):
    # Whole-frame path: model.set_test_input zero-pads to a multiple of 64
    # and forward_joint_test crops back internally.
    model.set_test_input(sample)
    model.forward_joint_test()
    return model.test_outputs["interp_out"]


def forward_tiled(model, sample, h_patch=640, w_patch=896, h_ov=305, w_ov=352):
    """Repo's overlapping-tile path from run_samples.py."""
    frame1 = sample["clean_image_first"]
    B, C, H, W = frame1.shape
    h_patch = min(h_patch, H)
    w_patch = min(w_patch, W)
    h_stride = h_patch - h_ov
    w_stride = w_patch - w_ov
    h_idx_list = list(range(0, H - h_patch, h_stride)) + [max(0, H - h_patch)]
    w_idx_list = list(range(0, W - w_patch, w_stride)) + [max(0, W - w_patch)]
    # E accumulates weighted patch outputs, W_ the weights; final = E / W_.
    E = torch.zeros(B, C, H, W, device=frame1.device)
    W_ = torch.zeros_like(E)
    keys = ["clean_image_first", "clean_image_last",
            "voxel_grid_0t", "voxel_grid_t1", "voxel_grid_t0"]
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            _sample = {k: sample[k][..., h_idx:h_idx + h_patch, w_idx:w_idx + w_patch]
                       for k in keys}
            model.set_test_input(_sample)
            model.forward_joint_test()
            out_patch = model.test_outputs["interp_out"]
            out_mask = torch.ones_like(out_patch)
            # Zero out half of each overlap band so neighboring tiles stitch
            # without seams (interior tiles trim on all overlapped sides).
            if h_idx < h_idx_list[-1]:
                out_patch[..., -h_ov // 2:, :] *= 0
                out_mask[..., -h_ov // 2:, :] *= 0
            if w_idx < w_idx_list[-1]:
                out_patch[..., -w_ov // 2:] *= 0
                out_mask[..., -w_ov // 2:] *= 0
            if h_idx > h_idx_list[0]:
                out_patch[..., :h_ov // 2, :] *= 0
                out_mask[..., :h_ov // 2, :] *= 0
            if w_idx > w_idx_list[0]:
                out_patch[..., :w_ov // 2] *= 0
                out_mask[..., :w_ov // 2] *= 0
            E[:, :, h_idx:h_idx + h_patch, w_idx:w_idx + w_patch].add_(out_patch)
            W_[:, :, h_idx:h_idx + h_patch, w_idx:w_idx + w_patch].add_(out_mask)
    return E.div_(W_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "challenge_data"))
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--skips", nargs="+", default=list(SKIPS))
    ap.add_argument("--seqs", nargs="+", default=None,
                    help="only process these sequence names (default: all)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-name", default="ours_large", choices=["ours", "ours_large"])
    ap.add_argument("--force-tiled", action="store_true")
    ap.add_argument("--flip", default="none", choices=["none", "h", "v", "hv"],
                    help="spatially flip inputs (frames+voxels); outputs are unflipped")
    ap.add_argument("--vox-norm", default="none", choices=["none", "robust98", "unitstd"],
                    help="rescale each voxel grid to a fixed robust magnitude")
    ap.add_argument("--vox-target", type=float, default=1.0,
                    help="target scale for --vox-norm (98th pct of |nonzero| or std)")
    args = ap.parse_args()
    # Tensor dims to flip for TTA: -2 = height (v), -1 = width (h).
    flip_dims = []
    if "v" in args.flip:
        flip_dims.append(-2)
    if "h" in args.flip:
        flip_dims.append(-1)

    # Minimal stand-in for the repo's argparse namespace expected by OurModel.
    class NetArgs:
        voxel_num_bins = NUM_BINS
        flow_tb_debug = False
        smoothness_weight = 10.0

    model = OurModel(NetArgs())
    model.initialize("final_models", args.model_name)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_model(ckpt)
    model.cuda()
    model.eval()
    model.set_mode("joint")
    print(f"loaded {args.ckpt} into {args.model_name}", flush=True)

    use_tiled = args.force_tiled
    n_done = 0
    t_start_all = time.time()
    with torch.no_grad():
        for split in args.splits:
            for skip in args.skips:
                skip_dir = os.path.join(args.data_dir, split, skip)
                if not os.path.isdir(skip_dir):
                    continue
                for seq_name in sorted(os.listdir(skip_dir)):
                    if not os.path.isdir(os.path.join(skip_dir, seq_name)):
                        continue
                    if args.seqs is not None and seq_name not in args.seqs:
                        continue
                    seq = load_sequence(args.data_dir, split, skip, seq_name)
                    inputs = seq.inputs
                    # map each todo to its bracketing anchor pair
                    for todo in seq.todos:
                        prev = max((f for f in inputs if f.index < todo.index),
                                   key=lambda f: f.index)
                        nxt = min((f for f in inputs if f.index > todo.index),
                                  key=lambda f: f.index)
                        t0 = time.time()
                        vox_0t, vox_t0, vox_t1 = make_voxels(
                            seq, prev.timestamp, todo.timestamp, nxt.timestamp)
                        if args.vox_norm != "none":
                            vox_0t = normalize_voxel(vox_0t, args.vox_norm, args.vox_target)
                            vox_t0 = normalize_voxel(vox_t0, args.vox_norm, args.vox_target)
                            vox_t1 = normalize_voxel(vox_t1, args.vox_norm, args.vox_target)
                        f0 = torch.from_numpy(seq.load_frame(prev)).permute(2, 0, 1)[None].cuda()
                        f1 = torch.from_numpy(seq.load_frame(nxt)).permute(2, 0, 1)[None].cuda()
                        sample = {
                            "clean_image_first": f0,
                            "clean_image_last": f1,
                            "voxel_grid_0t": torch.from_numpy(vox_0t)[None].cuda(),
                            "voxel_grid_t0": torch.from_numpy(vox_t0)[None].cuda(),
                            "voxel_grid_t1": torch.from_numpy(vox_t1)[None].cuda(),
                        }
                        if flip_dims:
                            sample = {k: torch.flip(v, flip_dims) for k, v in sample.items()}
                        if use_tiled:
                            out = forward_tiled(model, sample)
                        else:
                            # Prefer the whole-frame path; permanently fall
                            # back to overlapping tiles on the first OOM.
                            try:
                                out = forward_whole(model, sample)
                            except torch.cuda.OutOfMemoryError:
                                print("OOM on whole frame; switching to tiled mode", flush=True)
                                torch.cuda.empty_cache()
                                use_tiled = True
                                out = forward_tiled(model, sample)
                        if flip_dims:
                            out = torch.flip(out, flip_dims)
                        img = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                        assert img.shape == (seq.height, seq.width, 3), img.shape
                        save_result_png(args.output_dir, skip, seq_name, todo.index, img)
                        n_done += 1
                        print(f"{split}/{skip}/{seq_name}/{todo.index:06d} "
                              f"{time.time() - t0:.2f}s "
                              f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB",
                              flush=True)
    print(f"done: {n_done} frames in {time.time() - t_start_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
