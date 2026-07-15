#!/usr/bin/env python3
"""VFIMamba (NeurIPS 2024) image-only VFI runner for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. VFIMamba is an image-only member of the
  30-member ensemble (state-space-model architecture, adds diversity to the
  transformer/flow members). Pipeline: data prep -> training -> member
  inference (THIS script) -> per-frame quadratic-form fusion
  (recipe_v15/v16.json) -> submission assembly (scripts/build_final.py).

Method:
  Uses the full 'VFIMamba' model (F=32, depth [2,2,2,3,3]). Default mode is
  recursive midpoint bisection (t=0.5 only), because VFIMamba's off-center
  timestep generalization proved poor on this data; --mode timestep enables
  native arbitrary-timestep inference at the EXACT timestamp fractions.
  Optional anchored bisection (--anchor-dir) conditions deep recursion
  levels on materialized ensemble frames.

Inputs:
  --data-dir  challenge data root; weights/vfimamba/VFIMamba.pkl
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame,
  consumed by the fusion stage.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_vfimamba.py \
      --data-dir challenge_data --splits validation --skips 1skip,3skip,7skip,15skip \
      --output-dir results_vfimamba --scale 0.5 --local on
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
VFIMAMBA_DIR = ROOT / "models" / "VFIMamba"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VFIMAMBA_DIR))

from evlib.dataset import load_sequence, save_result_png  # noqa: E402


def load_model():
    # Configure the upstream repo's global model config before importing its
    # Trainer (the arch is chosen via module-level globals).
    import config as cfg
    cfg.MODEL_CONFIG['LOGNAME'] = 'VFIMamba'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])
    from Trainer_finetune import Model

    model = Model(-1)
    ckpt = torch.load(str(ROOT / "weights" / "vfimamba" / "VFIMamba.pkl"),
                      map_location="cpu", weights_only=False)

    def convert(param):
        # Strip DataParallel 'module.' prefixes; drop input-size-dependent
        # buffers (attn_mask / HW) that would not match inference resolution.
        return {k.replace("module.", ""): v for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k}

    model.net.load_state_dict(convert(ckpt), strict=True)
    model.eval()
    model.device()
    return model


def to_tensor(img, device):
    # img: float32 HWC RGB [0,1] -> 1CHW
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to(x, mult):
    # Reflect-pad right/bottom to a multiple of `mult` (64 here); original
    # h, w are returned so outputs can be cropped back.
    _, _, h, w = x.shape
    ph = (h + mult - 1) // mult * mult
    pw = (w + mult - 1) // mult * mult
    return F.pad(x, (0, pw - w, 0, ph - h), mode="reflect"), h, w


FLIPS = ((), (3,), (2,), (2, 3))  # identity, hflip, vflip, hvflip


@torch.no_grad()
def infer_t(model, img0, img1, t, local, scale, tta):
    # Single prediction at time fraction t; with --tta average the 4 flip
    # variants (transform inputs, predict, invert the flip on the output).
    if not tta:
        return model.inference(img0, img1, local, TTA=False, timestep=t,
                               scale=scale, fast_TTA=False).clamp(0, 1)
    acc = None
    for dims in FLIPS:
        a = torch.flip(img0, dims) if dims else img0
        b = torch.flip(img1, dims) if dims else img1
        m = model.inference(a, b, local, TTA=False, timestep=t,
                            scale=scale, fast_TTA=False)
        m = torch.flip(m, dims) if dims else m
        acc = m if acc is None else acc + m
    return (acc / len(FLIPS)).clamp(0, 1)


@torch.no_grad()
def bisect(model, img0, img1, depth, local, scale, tta):
    """2^depth - 1 intermediate frames via recursive midpoint (t=0.5 only —
    VFIMamba's off-center timestep generalization is poor)."""
    if depth == 0:
        return []
    mid = infer_t(model, img0, img1, 0.5, local, scale, tta)
    return (bisect(model, img0, mid, depth - 1, local, scale, tta) + [mid]
            + bisect(model, mid, img1, depth - 1, local, scale, tta))


def bisect_anch(model, img0, img1, ia, ib, local, scale, tta, anchors):
    """Index-aware bisection; deeper levels condition on `anchors` (a dict
    {index: padded tensor} of externally materialized frames) instead of the
    model's own midpoint. Every returned frame is still the model's own
    prediction. Returns list of (index, tensor)."""
    if ib - ia < 2:
        return []
    im = (ia + ib) // 2
    mid = infer_t(model, img0, img1, 0.5, local, scale, tta)
    anc = anchors.get(im, mid)
    return (bisect_anch(model, img0, anc, ia, im, local, scale, tta, anchors)
            + [(im, mid)]
            + bisect_anch(model, anc, img1, im, ib, local, scale, tta, anchors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--skips", default="1skip,3skip,7skip,15skip")
    ap.add_argument("--output-dir", default=str(ROOT / "results_vfimamba"))
    ap.add_argument("--scale", type=float, default=0.5,
                    help="flow estimation downscale (0 = full res; README: 0.5 for HD/2K)")
    ap.add_argument("--local", default="on", choices=["on", "off"],
                    help="use the local refinement blocks (SNU-FILM: on for easy/medium)")
    ap.add_argument("--tta", action="store_true",
                    help="average over h/v/hv flips")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="debug: only first N anchor pairs per sequence")
    ap.add_argument("--mode", default="bisect", choices=["bisect", "timestep"],
                    help="bisect = recursive midpoint (default; timestep mode "
                         "collapses off-center)")
    ap.add_argument("--anchor-dir", default=None,
                    help="materialized ensemble root ({skip}/{seq}/{idx}.png); "
                         "bisect deep levels condition on these frames")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    splits = args.splits.split(",")
    skips = args.skips.split(",")
    local = args.local == "on"

    device = torch.device("cuda")
    model = load_model()

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
                    img0 = to_tensor(seq.load_frame(a), device)
                    img1 = to_tensor(seq.load_frame(b), device)
                    img0, h, w = pad_to(img0, 64)
                    img1, _, _ = pad_to(img1, 64)
                    denom = ts_by_idx[b.index] - ts_by_idx[a.index]
                    if args.mode == "bisect":
                        # Recursive t=0.5 bisection: needs step to be a power
                        # of two (depth = log2(step) levels).
                        depth = step.bit_length() - 1
                        assert 1 << depth == step
                        if args.anchor_dir:
                            # Preload materialized ensemble frames as pseudo-
                            # anchors for the deeper bisection levels.
                            anchors = {}
                            for i in range(a.index + 1, b.index):
                                p = (Path(args.anchor_dir) / skip / seq.name
                                     / f"{i:06d}.png")
                                if p.exists():
                                    arr = np.asarray(
                                        Image.open(p).convert("RGB"),
                                        np.float32) / 255.0
                                    t = to_tensor(arr, device)
                                    t, _, _ = pad_to(t, 64)
                                    anchors[i] = t
                            mids = bisect_anch(model, img0, img1, a.index,
                                               b.index, local, args.scale,
                                               args.tta, anchors)
                        else:
                            mids = [(a.index + k, m) for k, m in enumerate(
                                bisect(model, img0, img1, depth, local,
                                       args.scale, args.tta), start=1)]
                        for idx, mid in mids:
                            if idx not in todo_idx:
                                continue
                            out = mid[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1)
                            save_result_png(out_dir, skip, seq.name, idx,
                                            out.cpu().numpy().astype(np.float32))
                            n_seq += 1
                        continue
                    # --mode timestep: direct arbitrary-t inference at the
                    # exact timestamp fraction of each TODO frame.
                    for k in range(1, step):
                        idx = a.index + k
                        if idx not in todo_idx:
                            continue
                        frac = (ts_by_idx[idx] - ts_by_idx[a.index]) / denom
                        mid = infer_t(model, img0, img1, frac, local,
                                      args.scale, args.tta)
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
