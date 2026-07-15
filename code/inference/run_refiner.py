#!/usr/bin/env python3
"""Apply the trained fusion refiner (RefinerUNet) to real challenge data.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script for one of our own learned fusion
  refiners. The refiner is trained (scripts/train_refiner.py) to combine
  precomputed base-member outputs (GIMM-VFI, TimeLens, linear blend) with
  event evidence into a better frame; its outputs are themselves members
  of the 30-member ensemble. Pipeline: data prep -> training -> member
  inference (THIS script, run AFTER the base members it consumes) ->
  per-frame quadratic-form fusion (recipe_v15/v16.json) -> submission
  assembly (scripts/build_final.py).

Method:
  Per TODO frame, builds the same 32ch input as training:
    gimm (from --gimm-dir PNGs), tl (--tl-dir), lin (computed), a0, a1,
    16-bin signed voxel over the anchor gap (evlib.events_to_voxel),
    alpha map. The network gates/combines the base predictions.
  4-base checkpoints (n_bases=4) additionally consume GIMM-VFI-R PNGs
  from --gimmr-dir; the base count is inferred from the checkpoint.

Inputs:
  --ckpt      trained refiner checkpoint (.pth)
  --gimm-dir / --gimmr-dir / --tl-dir  base-member result directories
              (each laid out as {skip}/{seq}/{index:06d}.png)
Outputs:
  refined uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_refiner.py \
      --ckpt checkpoints/refiner/refiner_step30000.pth \
      --gimm-dir results_gimmvfi_tta --tl-dir results_timelens_tta \
      --splits validation --skips 7skip 15skip --output-dir results_refined
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
sys.path.insert(0, str(ROOT / "scripts"))

from evlib.dataset import SKIPS, events_to_voxel, iter_sequences, save_result_png  # noqa: E402
from train_refiner import RefinerUNet  # noqa: E402


def load01(p):
    # Load a PNG as float32 HWC RGB in [0,1].
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def to_t(img, device):
    # float32 HWC -> 1CHW tensor on the target device.
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gimm-dir", default=str(ROOT / "results_gimmvfi_tta"))
    ap.add_argument("--gimmr-dir", default=str(ROOT / "results_gimmvfi_r_tta"))
    ap.add_argument("--tl-dir", default=str(ROOT / "results_timelens_tta"))
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--skips", nargs="+", default=list(SKIPS))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--vox-scale", type=float, default=1.0,
                    help="multiply real voxels (train/real density mismatch knob)")
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Infer the architecture hyperparameters (number of fused bases, input
    # channels, base width) directly from checkpoint tensor shapes so one
    # script serves all refiner variants.
    n_bases = ck["state_dict"]["head_gate.bias"].shape[0]
    cin = ck["state_dict"]["enc0.body.0.weight"].shape[1]
    base = ck["state_dict"]["enc0.body.0.weight"].shape[0]
    net = RefinerUNet(cin=cin, base=base, n_bases=n_bases).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    use_gimmr = n_bases == 4  # 4-base variant also fuses GIMM-VFI-R
    print(f"loaded {args.ckpt} (step {ck.get('step')}, n_bases={n_bases})", flush=True)

    n_total = 0
    t0_all = time.time()
    with torch.no_grad():
        for seq in iter_sequences(args.data_dir, splits=args.splits, skips=args.skips):
            inputs = seq.inputs
            t_seq = time.time()
            n = 0
            for left, right in zip(inputs[:-1], inputs[1:]):
                gap = [f for f in seq.todos if left.index < f.index < right.index]
                if not gap:
                    continue  # no TODO frames between this anchor pair
                # Anchor frames and the shared 16-bin signed event voxel of
                # the whole gap (loaded once, reused for every TODO inside).
                a0 = seq.load_frame(left)
                a1 = seq.load_frame(right)
                ev = seq.load_events(left.timestamp, right.timestamp)
                vox = events_to_voxel(ev, 16, seq.height, seq.width,
                                      left.timestamp, right.timestamp) * args.vox_scale
                vox_t = torch.from_numpy(vox)[None].to(device)
                a0_t, a1_t = to_t(a0, device), to_t(a1, device)
                for f in gap:
                    # alpha = normalized target time within the gap; also fed
                    # to the net as a constant spatial map.
                    alpha = (f.timestamp - left.timestamp) / (right.timestamp - left.timestamp)
                    rel = Path(f"{seq.skip}/{seq.name}/{f.index:06d}.png")
                    # Base predictions: precomputed member PNGs + linear blend.
                    gimm = to_t(load01(Path(args.gimm_dir) / rel), device)
                    tl = to_t(load01(Path(args.tl_dir) / rel), device)
                    lin = (1 - alpha) * a0_t + alpha * a1_t
                    amap = torch.full((1, 1, seq.height, seq.width), alpha,
                                      device=device)
                    if use_gimmr:
                        gimmr = to_t(load01(Path(args.gimmr_dir) / rel), device)
                        x = torch.cat([gimm, gimmr, tl, lin, a0_t, a1_t,
                                       vox_t, amap], dim=1)
                        bases = (gimm, gimmr, tl, lin)
                    else:
                        x = torch.cat([gimm, tl, lin, a0_t, a1_t, vox_t, amap],
                                      dim=1)
                        bases = (gimm, tl, lin)
                    # The net predicts per-base gates + a residual over the
                    # gated combination of `bases`; second output is unused.
                    out, _ = net(x, bases)
                    img = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    save_result_png(args.output_dir, seq.skip, seq.name, f.index, img)
                    n += 1
            n_total += n
            print(f"[{seq.split}/{seq.skip}/{seq.name}] {n} frames "
                  f"in {time.time() - t_seq:.1f}s", flush=True)
    print(f"DONE: {n_total} in {time.time() - t0_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
