#!/usr/bin/env python3
"""GIMM-VFI (NeurIPS 2024) image-only VFI runner for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. GIMM-VFI (both -F and -R variants) are
  image-only members of the 28-member ensemble; their outputs also serve as
  base inputs to our trained fusion refiners (run_refiner*.py). Pipeline:
  data prep -> training -> member inference (THIS script) -> per-frame
  quadratic-form fusion (recipe_v15/v16.json) -> submission assembly
  (scripts/build_final.py).

Method:
  Uses native continuous-time interpolation: forward(I0, I1, t) with the EXACT
  timestamp fractions of the TODO frames (no bisection). Variants:
    - f: GIMM-VFI-F (FlowFormer-based, higher PSNR per paper)  [default]
    - r: GIMM-VFI-R (RAFT-based)
  Non-LPIPS checkpoints (gimmvfi_{f,r}_arb.pt) are used since PSNR is the metric.

Inputs:
  --data-dir  challenge data root; weights/gimmvfi/gimmvfi_{f,r}_arb.pt
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame,
  consumed by the fusion stage and the refiner scripts.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_gimmvfi.py \
      --data-dir challenge_data --splits validation --skips 1skip,3skip,7skip,15skip \
      --output-dir results_gimmvfi --ds-scale 0.5
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
GIMMVFI_DIR = ROOT / "models" / "GIMM-VFI"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GIMMVFI_DIR / "src"))

from evlib.dataset import load_sequence, save_result_png  # noqa: E402


def load_model(variant):
    # relative ckpt paths (pretrained_ckpt/...) inside the repo
    os.chdir(GIMMVFI_DIR)
    from models import create_model
    from utils.config import load_config, augment_arch_defaults

    cfg = load_config(str(GIMMVFI_DIR / "configs" / "gimmvfi" / f"gimmvfi_{variant}_arb.yaml"))
    arch = augment_arch_defaults(cfg.arch)
    model, _ = create_model(arch)
    ckpt = torch.load(str(ROOT / "weights" / "gimmvfi" / f"gimmvfi_{variant}_arb.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda().eval()
    return model


def to_tensor(img, device):
    # img: float32 HWC RGB [0,1] -> 1CHW
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to(x, mult):
    # Reflect-pad right/bottom to a multiple of `mult` (64 here, required by
    # the flow backbone); original h, w returned for cropping outputs back.
    _, _, h, w = x.shape
    ph = (h + mult - 1) // mult * mult
    pw = (w + mult - 1) // mult * mult
    return F.pad(x, (0, pw - w, 0, ph - h), mode="reflect"), h, w


FLIPS = ((), (3,), (2,), (2, 3))  # identity, hflip, vflip, hvflip


@torch.no_grad()
def infer_pair(model, img0, img1, fracs, ds_factor):
    """Interpolate at the given exact time fractions. Returns list of 1CHW."""
    # Stack the two anchors along a new time axis: (B, C, 2, H, W).
    xs = torch.cat((img0.unsqueeze(2), img1.unsqueeze(2)), dim=2)
    batch_size = xs.shape[0]
    s_shape = xs.shape[-2:]
    # GIMM-VFI's implicit-neural decoder takes per-timestep coordinate grids
    # plus scalar timesteps; one entry per requested fraction.
    coord_inputs = [
        (model.sample_coord_input(batch_size, s_shape, [f], device=xs.device,
                                  upsample_ratio=ds_factor), None)
        for f in fracs
    ]
    timesteps = [
        f * torch.ones(batch_size, device=xs.device, dtype=torch.float)
        for f in fracs
    ]
    out = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
    return [im.clamp(0, 1) for im in out["imgt_pred"]]


@torch.no_grad()
def infer_pair_tta(model, img0, img1, fracs, ds_factor, tta, tta_time=False):
    # Self-ensemble over spatial flips (--tta) and/or temporal reversal
    # (--tta-time: swap anchors and query at 1-t); average all variants.
    if not tta and not tta_time:
        return infer_pair(model, img0, img1, fracs, ds_factor)
    flips = FLIPS if tta else ((),)
    time_dirs = (False, True) if tta_time else (False,)
    accs, n = None, 0
    for dims in flips:
        a = torch.flip(img0, dims) if dims else img0
        b = torch.flip(img1, dims) if dims else img1
        for rev in time_dirs:
            if rev:
                # Time-reversed pass: anchors swapped, fractions mirrored.
                outs = infer_pair(model, b, a, [1.0 - f for f in fracs], ds_factor)
            else:
                outs = infer_pair(model, a, b, fracs, ds_factor)
            outs = [torch.flip(m, dims) if dims else m for m in outs]
            accs = outs if accs is None else [x + m for x, m in zip(accs, outs)]
            n += 1
    return [x / n for x in accs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--skips", default="1skip,3skip,7skip,15skip")
    ap.add_argument("--output-dir", default=str(ROOT / "results_gimmvfi"))
    ap.add_argument("--variant", default="f", choices=["f", "r"])
    ap.add_argument("--ds-scale", type=float, default=0.5,
                    help="flow downsampling factor (README: 0.5 for high-res)")
    ap.add_argument("--tta-time", action="store_true",
                    help="also average the time-reversed direction (swap "
                         "anchors, t -> 1-t)")
    ap.add_argument("--tta", action="store_true",
                    help="average over h/v/hv flips")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="debug: only first N anchor pairs per sequence")
    ap.add_argument("--chunk", type=int, default=8,
                    help="max timesteps per forward pass")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    splits = args.splits.split(",")
    skips = args.skips.split(",")

    device = torch.device("cuda")
    model = load_model(args.variant)
    ds_factor = args.ds_scale  # always pass a float, as in the official demo

    total_frames = 0
    t_start = time.time()
    for split in splits:
        for skip in skips:
            skip_n = int(skip.replace("skip", ""))
            step = skip_n + 1

            skip_dir = data_dir / split / skip
            for seq_dir in sorted(p for p in skip_dir.iterdir() if p.is_dir()):
                seq = load_sequence(data_dir, split, skip, seq_dir.name)
                inputs = seq.inputs
                ts_by_idx = {f.index: f.timestamp for f in seq.frames}
                todo_idx = {f.index for f in seq.todos}
                t_seq = time.time()
                n_seq = 0
                n_pairs = 0
                for a, b in zip(inputs[:-1], inputs[1:]):
                    if b.index - a.index != step:
                        continue  # non-contiguous anchor pair
                    if args.max_pairs and n_pairs >= args.max_pairs:
                        break
                    n_pairs += 1
                    # Exact time fractions of the TODO frames within the gap,
                    # from the per-frame timestamps in frame_info.txt.
                    idxs = [a.index + k for k in range(1, step)
                            if a.index + k in todo_idx]
                    if not idxs:
                        continue
                    denom = ts_by_idx[b.index] - ts_by_idx[a.index]
                    fracs = [(ts_by_idx[i] - ts_by_idx[a.index]) / denom
                             for i in idxs]
                    img0 = to_tensor(seq.load_frame(a), device)
                    img1 = to_tensor(seq.load_frame(b), device)
                    img0, h, w = pad_to(img0, 64)
                    img1, _, _ = pad_to(img1, 64)
                    # Chunk timesteps to bound VRAM (one forward per chunk).
                    for c0 in range(0, len(idxs), args.chunk):
                        c_idxs = idxs[c0:c0 + args.chunk]
                        c_fracs = fracs[c0:c0 + args.chunk]
                        mids = infer_pair_tta(model, img0, img1, c_fracs,
                                              ds_factor, args.tta,
                                              args.tta_time)
                        for idx, mid in zip(c_idxs, mids):
                            out = mid[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1)
                            save_result_png(out_dir, skip, seq.name, idx,
                                            out.cpu().numpy().astype(np.float32))
                            n_seq += 1
                total_frames += n_seq
                dt = time.time() - t_seq
                print(f"[{split}/{skip}/{seq.name}] {n_seq} frames in {dt:.1f}s "
                      f"({dt / max(n_seq, 1):.2f}s/frame)", flush=True)
    print(f"DONE: {total_frames} frames in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
