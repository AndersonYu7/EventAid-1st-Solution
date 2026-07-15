#!/usr/bin/env python3
"""RIFE HDv3 image-only VFI runner for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. RIFE is a fast image-only member of the
  30-member ensemble, adding flow-based diversity to the transformer
  members. Pipeline: data prep -> training -> member inference (THIS
  script) -> per-frame quadratic-form fusion (recipe_v15/v16.json) ->
  submission assembly (scripts/build_final.py).

Method:
  Pure recursive midpoint interpolation: skip+1 is a power of two (2/4/8/16),
  so depth = log2(skip+1) bisections produce frames at exactly the needed
  time fractions (timestamps are uniform to ~1us). Optional anchored
  bisection (--anchor-dir) conditions deep recursion levels on materialized
  ensemble frames instead of the model's own midpoints.

Inputs:
  --data-dir  challenge data root; weights are the stock RIFE HDv3
              train_log checkpoint under models/RIFE/train_log.
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame,
  consumed by the fusion stage.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_rife.py \
      --data-dir challenge_data --splits validation --skips 1skip,3skip,7skip,15skip \
      --output-dir results_rife
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RIFE_DIR = ROOT / "models" / "RIFE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RIFE_DIR))

from evlib.dataset import load_sequence, save_result_png  # noqa: E402


def load_model():
    # train_log.RIFE_HDv3 imports model.loss / model.warplayer -> needs cwd/path at repo
    os.chdir(RIFE_DIR)
    from train_log.RIFE_HDv3 import Model
    model = Model()
    model.load_model("train_log", -1)
    model.eval()
    model.device()
    return model


def to_tensor(img, device):
    # img: float32 HWC RGB [0,1] -> 1CHW
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to_32(x):
    # Zero-pad right/bottom to multiples of 32 (RIFE pyramid requirement);
    # original h, w are returned for cropping outputs back.
    _, _, h, w = x.shape
    ph = (h + 31) // 32 * 32
    pw = (w + 31) // 32 * 32
    return F.pad(x, (0, pw - w, 0, ph - h)), h, w


FLIPS = ((), (3,), (2,), (2, 3))  # identity, hflip, vflip, hvflip


@torch.no_grad()
def infer_mid(model, img0, img1, tta):
    # Midpoint (t=0.5) prediction; with --tta average the 4 flip variants
    # (transform inputs, predict, invert the flip on the output).
    if not tta:
        return model.inference(img0, img1)
    acc = None
    for dims in FLIPS:
        a = torch.flip(img0, dims) if dims else img0
        b = torch.flip(img1, dims) if dims else img1
        m = model.inference(a, b)
        m = torch.flip(m, dims) if dims else m
        acc = m if acc is None else acc + m
    return acc / len(FLIPS)


@torch.no_grad()
def bisect(model, img0, img1, depth, tta=False):
    """Return list of 2^depth - 1 intermediate frames (tensors) in time order."""
    if depth == 0:
        return []
    mid = infer_mid(model, img0, img1, tta)
    return (bisect(model, img0, mid, depth - 1, tta) + [mid]
            + bisect(model, mid, img1, depth - 1, tta))


@torch.no_grad()
def bisect_anch(model, img0, img1, ia, ib, tta, anchors):
    """Index-aware bisection; deeper levels condition on `anchors`
    ({index: padded tensor} of materialized ensemble frames) instead of the
    model's own midpoint. Returned frames stay the model's own predictions."""
    if ib - ia < 2:
        return []
    im = (ia + ib) // 2
    mid = infer_mid(model, img0, img1, tta)
    anc = anchors.get(im, mid)
    return (bisect_anch(model, img0, anc, ia, im, tta, anchors)
            + [(im, mid)]
            + bisect_anch(model, anc, img1, im, ib, tta, anchors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--skips", default="1skip,3skip,7skip,15skip")
    ap.add_argument("--output-dir", default=str(ROOT / "results_rife"))
    ap.add_argument("--tta", action="store_true",
                    help="average over h/v/hv flips at every midpoint inference")
    ap.add_argument("--anchor-dir", default=None,
                    help="materialized ensemble root ({skip}/{seq}/{idx}.png); "
                         "bisect deep levels condition on these (abs path safest)")
    args = ap.parse_args()
    anchor_root = Path(args.anchor_dir).resolve() if args.anchor_dir else None

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    splits = args.splits.split(",")
    skips = args.skips.split(",")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()

    total_frames = 0
    t_start = time.time()
    for split in splits:
        for skip in skips:
            # "Nskip" = N missing frames -> anchor stride step = N+1 must be
            # a power of two for pure bisection.
            skip_n = int(skip.replace("skip", ""))
            step = skip_n + 1
            depth = step.bit_length() - 1
            assert 1 << depth == step, f"skip+1={step} is not a power of 2"

            skip_dir = data_dir / split / skip
            for seq_dir in sorted(p for p in skip_dir.iterdir() if p.is_dir()):
                seq = load_sequence(data_dir, split, skip, seq_dir.name)
                inputs = seq.inputs
                todo_idx = {f.index for f in seq.todos}
                t_seq = time.time()
                n_seq = 0
                for a, b in zip(inputs[:-1], inputs[1:]):
                    if b.index - a.index != step:
                        continue  # non-contiguous anchor pair
                    img0 = to_tensor(seq.load_frame(a), device)
                    img1 = to_tensor(seq.load_frame(b), device)
                    img0, h, w = pad_to_32(img0)
                    img1, _, _ = pad_to_32(img1)
                    if anchor_root is not None:
                        # Preload materialized ensemble frames as pseudo-
                        # anchors for the deeper bisection levels.
                        anchors = {}
                        for i in range(a.index + 1, b.index):
                            p = anchor_root / skip / seq.name / f"{i:06d}.png"
                            if p.exists():
                                arr = np.asarray(Image.open(p).convert("RGB"),
                                                 np.float32) / 255.0
                                t = to_tensor(arr, device)
                                t, _, _ = pad_to_32(t)
                                anchors[i] = t
                        mids = bisect_anch(model, img0, img1, a.index, b.index,
                                           args.tta, anchors)
                    else:
                        mids = [(a.index + k, m) for k, m in enumerate(
                            bisect(model, img0, img1, depth, tta=args.tta),
                            start=1)]
                    for idx, mid in mids:
                        if idx not in todo_idx:
                            continue  # only save challenge-requested frames
                        # Crop padding away, back to HWC float [0,1].
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
