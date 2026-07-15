#!/usr/bin/env python3
"""Apply the trained v2arch fusion refiner (RefinerV2) to challenge data (full res).

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script for the v2-architecture variant of our own
  lightweight fusion refiners (refiner_v2arch.py; the 4.86M-param design).
  Like run_refiner.py it fuses precomputed base members (GIMM-VFI,
  TimeLens, linear anchor blend) with event evidence, but RefinerV2 takes
  the event voxel and anchor context as a separate conditioning input
  instead of one flat channel stack. Its refined outputs are members of
  the 30-member ensemble. Pipeline: data prep -> training (scripts/
  refiner_v2arch.py trainer) -> member inference (THIS script, run AFTER
  the base members it consumes) -> per-frame quadratic-form fusion
  (recipe_v15/v16.json) -> submission assembly (scripts/build_final.py).

Inputs:
  --ckpt      trained RefinerV2 checkpoint (.pth; stores feat/base widths)
  --gimm-dir / --tl-dir  base-member result directories laid out as
              {skip}/{seq}/{index:06d}.png
Outputs:
  refined uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_refiner_v2arch.py \
      --ckpt checkpoints/refiner_v2arch/v2arch_step2500.pth \
      --splits validation test --skips 7skip 15skip \
      --output-dir results_refined_v2a
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
from refiner_v2arch import RefinerV2  # noqa: E402


def load01(p):
    # Load a PNG as float32 HWC RGB in [0,1].
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def to_t(img, device):
    # float32 HWC -> 1CHW tensor on the target device.
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(device)


def pad4(x):
    # Replicate-pad right/bottom to multiples of 4 (RefinerV2 has two
    # stride-2 levels); original h, w returned for cropping outputs back.
    _, _, h, w = x.shape
    ph = (h + 3) // 4 * 4
    pw = (w + 3) // 4 * 4
    if ph == h and pw == w:
        return x, h, w
    return torch.nn.functional.pad(x, (0, pw - w, 0, ph - h), mode="replicate"), h, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gimm-dir", default=str(ROOT / "results_gimmf_ds10_tta"))
    ap.add_argument("--tl-dir", default=str(ROOT / "results_timelens_tta"))
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--skips", nargs="+", default=list(SKIPS))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Width hyperparameters are stored inside the checkpoint dict.
    net = RefinerV2(n_bases=3, feat=ck["feat"], base=ck["base"]).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    print(f"loaded {args.ckpt} (step {ck.get('step')})", flush=True)

    n_total = 0
    t0_all = time.time()
    with torch.no_grad():
        for seq in iter_sequences(args.data_dir, splits=args.splits, skips=args.skips):
            t_seq = time.time()
            inputs = seq.inputs
            n = 0
            for left, right in zip(inputs[:-1], inputs[1:]):
                gap = [f for f in seq.todos if left.index < f.index < right.index]
                if not gap:
                    continue  # no TODO frames between this anchor pair
                # Anchor frames and the shared 16-bin signed event voxel of
                # the whole gap (built once, reused for every TODO inside).
                a0 = to_t(load01(seq.root / left.path), device)
                a1 = to_t(load01(seq.root / right.path), device)
                ev = seq.load_events(left.timestamp, right.timestamp)
                vox = torch.from_numpy(events_to_voxel(
                    ev, 16, seq.height, seq.width,
                    left.timestamp, right.timestamp))[None].to(device)
                for f in gap:
                    # alpha = normalized target time within the gap.
                    alpha = (f.timestamp - left.timestamp) / (right.timestamp - left.timestamp)
                    rel = Path(f"{seq.skip}/{seq.name}/{f.index:06d}.png")
                    # Base predictions: precomputed member PNGs + linear blend.
                    gimm = to_t(load01(Path(args.gimm_dir) / rel), device)
                    tl = to_t(load01(Path(args.tl_dir) / rel), device)
                    lin = (1 - alpha) * a0 + alpha * a1
                    amap = torch.full((1, 1, seq.height, seq.width), alpha,
                                      device=device)
                    # RefinerV2 conditioning input: anchors + alpha map (7ch)
                    # and the event voxel, passed separately from the bases.
                    anchors = torch.cat([a0, a1, amap], dim=1)
                    vx, h, w = pad4(vox)
                    an, _, _ = pad4(anchors)
                    bs = [pad4(b)[0] for b in (gimm, tl, lin)]
                    # Net outputs gated base combination + residual; the
                    # second return value (gate maps) is unused here.
                    out, _ = net((vx, an), tuple(bs))
                    img = out[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    save_result_png(args.output_dir, seq.skip, seq.name, f.index, img)
                    n += 1
            n_total += n
            print(f"[{seq.split}/{seq.skip}/{seq.name}] {n} in {time.time()-t_seq:.1f}s",
                  flush=True)
    print(f"DONE: {n_total} in {time.time()-t0_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
