#!/usr/bin/env python3
"""EMA-VFI (CVPR 2023) image-only VFI runner for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. EMA-VFI variants (pretrained 'ours' and
  our EventAid-finetuned soups 'ours_ft_*') are image-only members of the
  30-member ensemble. Pipeline: data prep -> training -> member inference
  (THIS script) -> per-frame quadratic-form fusion (recipe_v15/v16.json)
  -> submission assembly (scripts/build_final.py).

Method:
  Uses the 'ours' fixed-timestep checkpoint (trained on Vimeo90K septuplets,
  NOT on EventAid). Pure recursive midpoint interpolation: skip+1 is a power
  of two (2/4/8/16), so depth = log2(skip+1) bisections produce frames at
  exactly the needed time fractions.

Inputs:
  --data-dir  challenge data root; --variant selects weights/emavfi/{v}.pkl
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame,
  consumed by the fusion stage.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_emavfi.py \
      --data-dir challenge_data --splits validation --skips 1skip,3skip,7skip,15skip \
      --output-dir results_emavfi
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
EMAVFI_DIR = ROOT / "models" / "EMA-VFI"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EMAVFI_DIR))

from evlib.dataset import load_sequence, save_result_png  # noqa: E402


def load_model(variant="ours"):
    # Configure the upstream EMA-VFI repo's global model config before
    # importing its Trainer: small arch for 'ours_small*', full arch otherwise.
    import config as cfg
    if variant.startswith("ours_small"):
        cfg.MODEL_CONFIG['LOGNAME'] = variant
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=16, depth=[2, 2, 2, 2, 2])
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = variant
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
    from Trainer import Model

    model = Model(-1)

    def convert(param):
        # Strip DataParallel 'module.' prefixes and drop buffers that depend
        # on training-time input size (attn_mask / HW).
        return {k.replace("module.", ""): v for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k}

    ckpt = ROOT / "weights" / "emavfi" / f"{variant}.pkl"
    model.net.load_state_dict(convert(torch.load(ckpt, map_location="cuda")))
    model.eval()
    model.device()
    return model


def to_tensor(img, device):
    # img: float32 HWC RGB [0,1] -> 1CHW
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to_32(x):
    # Pad right/bottom to multiples of 32 (network pyramid requirement);
    # original h, w are returned so outputs can be cropped back.
    _, _, h, w = x.shape
    ph = (h + 31) // 32 * 32
    pw = (w + 31) // 32 * 32
    return F.pad(x, (0, pw - w, 0, ph - h), mode="replicate"), h, w


FLIPS = ((), (3,), (2,), (2, 3))  # identity, hflip, vflip, hvflip


@torch.no_grad()
def infer_mid(model, img0, img1, tta):
    # Midpoint (t=0.5) prediction; with --tta average the 4 flip variants
    # (transform inputs, predict, invert the flip on the output).
    if not tta:
        return model.inference(img0, img1, TTA=False)
    acc = None
    for dims in FLIPS:
        a = torch.flip(img0, dims) if dims else img0
        b = torch.flip(img1, dims) if dims else img1
        m = model.inference(a, b, TTA=False)
        m = torch.flip(m, dims) if dims else m
        acc = m if acc is None else acc + m
    return acc / len(FLIPS)


@torch.no_grad()
def bisect(model, img0, img1, depth, tta=False):
    """Return list of 2^depth - 1 intermediate frames (tensors) in time order."""
    if depth == 0:
        return []
    mid = infer_mid(model, img0, img1, tta).clamp(0, 1)
    return (bisect(model, img0, mid, depth - 1, tta) + [mid]
            + bisect(model, mid, img1, depth - 1, tta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--skips", default="1skip,3skip,7skip,15skip")
    ap.add_argument("--output-dir", default=str(ROOT / "results_emavfi"))
    ap.add_argument("--variant", default="ours",
                    help="weights/emavfi/{variant}.pkl; ours_small* -> small arch, "
                         "anything else -> full arch (incl. ours_ft_* soups)")
    ap.add_argument("--tta", action="store_true",
                    help="average over h/v/hv flips at every midpoint inference")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    splits = args.splits.split(",")
    skips = args.skips.split(",")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.variant)

    total_frames = 0
    t_start = time.time()
    for split in splits:
        for skip in skips:
            # "Nskip" = N missing frames between anchors -> anchor stride
            # step = N+1 must be a power of two for pure bisection.
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
                    mids = bisect(model, img0, img1, depth, tta=args.tta)
                    for k, mid in enumerate(mids, start=1):
                        idx = a.index + k
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
