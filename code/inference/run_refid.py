#!/usr/bin/env python3
"""REFID (CVPR'23) zero-shot inference runner for the EventAid VFI Challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. REFID is an event-based member of the
  28-member ensemble, run zero-shot with HighREV-finetuned weights (no
  EventAid training). Pipeline: data prep -> training -> member inference
  (THIS script) -> per-frame quadratic-form fusion (recipe_v15/v16.json)
  -> submission assembly (scripts/build_final.py). Outputs per TODO frame
  are uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png.

Method — sharp-frame-interpolation mode (TwoSharpImageEventRecurrent +
FinalBidirectionAttenfusion). For each pair of consecutive [input] anchors
with n [TODO] frames in between it replicates GoProSharpEventRecurrentDataset:
  - one (n+1)-bin bilinear voxel grid (polarity +-1) over events in [ta, tb)
    (bins anchored to the anchor frame timestamps),
  - recurrent voxel tensor (n, 2, H, W) of overlapping bin pairs [i, i+2),
  - NO voxel normalisation (the repo's voxel_norm call in the dataset rebinds
    the loop variable, i.e. training/test both used unnormalised voxels),
  - lq = the two RGB anchors stacked (1, 2, 3, H, W),
and gets all n intermediate frames from one forward pass (the model is
recurrent over t, so any n works with any checkpoint).

Inputs are reflect-padded to a multiple of 8 (num_encoders=3) and outputs
cropped back (944x624 is already divisible by 8, so this is a no-op there).

Default checkpoints: HighREV-finetuned sharp-VFI weights from the official
GitHub release, selected per skip by validation PSNR (HighREV-7skip for
3skip; HighREV-15skip for 1/7/15skip — GoPro-only weights were ~2-5 dB
worse). Override with --ckpt (all skips) or --ckpt7/--ckpt15.

Example:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_refid.py \
      --splits validation --skips 1skip 3skip 7skip 15skip \
      --output-dir results_refid --flip none
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFID = os.path.join(ROOT, "models", "REFID")
sys.path.insert(0, ROOT)
sys.path.insert(0, REFID)

from evlib.dataset import SKIPS, events_to_voxel, load_sequence, save_result_png  # noqa: E402
from basicsr.models.archs.XXNet_final_attenfusion_arch import FinalBidirectionAttenfusion  # noqa: E402

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

WEIGHTS_DIR = os.path.join(ROOT, "weights", "refid")
DEFAULT_CKPT7 = os.path.join(WEIGHTS_DIR, "REFID-HighREV-7skip.pth")
DEFAULT_CKPT15 = os.path.join(WEIGHTS_DIR, "REFID-HighREV-15skip.pth")


def build_model(ckpt_path, device):
    model = FinalBidirectionAttenfusion(
        img_chn=6, ev_chn=2, num_encoders=3, base_num_channels=32,
        num_block=1, num_residual_blocks=2)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("params", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def pad_to_multiple(x, mult=8):
    """Reflect-pad NCHW (or N T C H W flattened) tensor on bottom/right."""
    h, w = x.shape[-2:]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph == 0 and pw == 0:
        return x, h, w
    shape = x.shape
    x = x.reshape(-1, 1, h, w)
    x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x.reshape(*shape[:-2], h + ph, w + pw), h, w


def flip_tensor(x, flip):
    if "h" in flip:
        x = torch.flip(x, [-1])
    if "v" in flip:
        x = torch.flip(x, [-2])
    return x


@torch.no_grad()
def interp_pair(model, device, img_a, img_b, events, ta, tb, n, H, W, flip):
    """Return list of n float32 RGB [0,1] HWC frames between img_a and img_b."""
    # voxel: (n+1) bins over [ta, tb], polarity +-1, no normalisation.
    # The recurrent model consumes overlapping bin PAIRS [i, i+2): one 2ch
    # slice per intermediate frame, i.e. shape (n, 2, H, W).
    vox = events_to_voxel(events, n + 1, H, W, ta, tb)  # (n+1, H, W)
    sub = np.stack([vox[i:i + 2] for i in range(n)], axis=0)  # (n, 2, H, W)
    voxel = torch.from_numpy(sub).unsqueeze(0).to(device)  # 1,n,2,H,W

    lq = torch.from_numpy(np.stack([img_a, img_b], 0).transpose(0, 3, 1, 2).copy())
    lq = lq.unsqueeze(0).to(device)  # 1,2,3,H,W

    if flip != "none":
        # TTA: flip frames and voxels together; the output is unflipped below.
        lq = flip_tensor(lq, flip)
        voxel = flip_tensor(voxel, flip)

    # Pad to a multiple of 8 (num_encoders=3 -> 3 stride-2 levels), run one
    # recurrent forward that yields all n intermediate frames, then crop.
    lq, h0, w0 = pad_to_multiple(lq.reshape(1, 6, H, W))
    lq = lq.reshape(1, 2, 3, *lq.shape[-2:])
    voxel, _, _ = pad_to_multiple(voxel)

    out = model(x=lq, event=voxel)  # 1,n,3,H',W'
    out = out[..., :h0, :w0]
    if flip != "none":
        out = flip_tensor(out, flip)
    out = out.clamp(0, 1).squeeze(0).permute(0, 2, 3, 1).float().cpu().numpy()
    return [out[i] for i in range(n)]


def run_sequence(model, device, seq, out_dir, flip, limit=None):
    inputs = seq.inputs
    todos = {f.index: f for f in seq.todos}
    n_done, t_total = 0, 0.0
    for a, b in zip(inputs[:-1], inputs[1:]):
        between = sorted((f for f in seq.todos if a.index < f.index < b.index),
                         key=lambda f: f.timestamp)
        if not between:
            continue
        if limit is not None and n_done >= limit:
            break
        img_a = seq.load_frame(a)
        img_b = seq.load_frame(b)
        events = seq.load_events(a.timestamp, b.timestamp)
        t0 = time.time()
        preds = interp_pair(model, device, img_a, img_b, events,
                            a.timestamp, b.timestamp, len(between),
                            seq.height, seq.width, flip)
        t_total += time.time() - t0
        for frame, pred in zip(between, preds):
            save_result_png(out_dir, seq.skip, seq.name, frame.index, pred)
            n_done += 1
    return n_done, t_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "challenge_data"))
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--skips", nargs="+", default=list(SKIPS))
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "results_refid"))
    ap.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none")
    ap.add_argument("--ckpt", default=None, help="single checkpoint for all skips")
    ap.add_argument("--ckpt7", default=DEFAULT_CKPT7, help="checkpoint for 3skip")
    ap.add_argument("--ckpt15", default=DEFAULT_CKPT15,
                    help="checkpoint for 1/7/15skip")
    ap.add_argument("--limit", type=int, default=None,
                    help="max TODO frames per sequence (debug)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    def get_model(skip):
        # Per-skip checkpoint routing (selected by validation PSNR):
        # HighREV-7skip weights for 3skip, HighREV-15skip for the rest.
        # Models are cached so each checkpoint is loaded at most once.
        ckpt = args.ckpt or (args.ckpt7 if skip == "3skip" else args.ckpt15)
        if ckpt not in models:
            print(f"loading {ckpt}", flush=True)
            models[ckpt] = build_model(ckpt, device)
        return models[ckpt]

    grand_n, grand_t = 0, 0.0
    for split in args.splits:
        for skip in args.skips:
            d = os.path.join(args.data_dir, split, skip)
            if not os.path.isdir(d):
                continue
            model = get_model(skip)
            for name in sorted(os.listdir(d)):
                if not os.path.isdir(os.path.join(d, name)):
                    continue
                seq = load_sequence(args.data_dir, split, skip, name)
                n, t = run_sequence(model, device, seq, args.output_dir,
                                    args.flip, args.limit)
                grand_n += n
                grand_t += t
                print(f"{split}/{skip}/{name}: {n} frames, "
                      f"{t:.1f}s ({t / max(n, 1):.2f}s/frame)", flush=True)
    print(f"TOTAL: {grand_n} frames, {grand_t:.1f}s GPU "
          f"({grand_t / max(grand_n, 1):.2f}s/frame)", flush=True)


if __name__ == "__main__":
    main()
