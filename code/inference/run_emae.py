#!/usr/bin/env python3
"""EMA-E (event-aware EMA-VFI, emae_arch.py) runner for the EventAid challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  This is a MEMBER-INFERENCE stage script. EMA-E is our zero-initialized
  event graft on EMA-VFI and the strongest single member of the 28-member
  ensemble. The pipeline is: data prep -> training (train scripts produce the
  EMA-E checkpoints) -> member inference (THIS script, one results dir per
  member/variant) -> per-frame quadratic-form fusion (recipe_v15/v16.json)
  -> uint8 PNG submission assembly (scripts/build_final.py).

Method:
  Same recursive midpoint bisection as run_emavfi.py, but every midpoint
  inference additionally feeds the 16-bin whole-gap event voxel built from the
  challenge events between the two (possibly intermediate) anchor times:
  events_to_voxel(ev, 16, H, W, t_a, t_b). Timestamps of intermediate frames
  come from frame_info.txt, so sub-interval voxels are exact. Optional
  "anchored bisection" (--anchor-dir) conditions deep recursion levels on
  materialized frames from the current best ensemble instead of the model's
  own midpoints, reducing error accumulation at 7skip/15skip.

Inputs:
  --data-dir  challenge data root ({split}/{skip}/{seq}/ with PNG frames,
              event files, and frame_info.txt timestamps)
  --ckpt      EMA-E checkpoint (.pkl); --film / --b4 select variant archs
Outputs:
  float32-derived uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png,
  one per TODO (to-be-interpolated) frame, consumed by the fusion stage.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_emae.py \
      --ckpt checkpoints/emae_ft/emae_step1000.pkl \
      --splits validation --skips 7skip,15skip --output-dir results_emae/s1000
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from emae_arch import build_emae, EV_BINS  # noqa: E402
from evlib.dataset import load_sequence, events_to_voxel, save_result_png  # noqa: E402


def to_tensor(img, device):
    # img: float32 HWC RGB in [0,1] -> 1CHW tensor on the target device
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_to_32(x, mode="replicate"):
    # Pad right/bottom so H and W are multiples of 32 (EMA-VFI pyramid
    # requirement); returns the padded tensor plus the original h, w for
    # cropping the output back to the exact challenge resolution.
    _, _, h, w = x.shape
    ph = (h + 31) // 32 * 32
    pw = (w + 31) // 32 * 32
    return F.pad(x, (0, pw - w, 0, ph - h), mode=mode), h, w


FLIPS = ((), (3,), (2,), (2, 3))  # identity, hflip, vflip, hvflip
DIHEDRAL = os.environ.get("EMAE_DIHEDRAL") == "1"  # add 4 transpose variants
# Temporal-reversal self-ensemble (ours): for the SYMMETRIC midpoint, swapping the
# two anchors and reversing event time+polarity (vox -> -flip(vox, time)) yields an
# equivalent estimate. Averaging it with the forward pass is an event-specific TTA
# axis that the spatial flips do not cover. EMAE_TREV=1 enables it.
TREV = os.environ.get("EMAE_TREV") == "1"


def _t(x, dims, tr):
    # Apply one dihedral transform: optional transpose (H<->W) then flips.
    if tr:
        x = x.transpose(2, 3)
    return torch.flip(x, dims) if dims else x


@torch.no_grad()
def infer_mid(net, img0, img1, vox, tta):
    """Predict the temporal midpoint frame; optionally self-ensemble.

    With --tta, averages over 4 spatial flips (x2 transposes if
    EMAE_DIHEDRAL=1 for the full 8-way dihedral group, x2 temporal reversal
    if EMAE_TREV=1). net(...)[3] is the merged RGB prediction of EMA-E.
    """
    if not tta:
        return net(torch.cat([img0, img1, vox], 1))[3]
    acc, n = None, 0
    transposes = (False, True) if DIHEDRAL else (False,)
    for trev in ((False, True) if TREV else (False,)):
        if trev:                       # swap anchors, reverse event time + polarity
            s0, s1, sv = img1, img0, -torch.flip(vox, [1])
        else:
            s0, s1, sv = img0, img1, vox
        for tr in transposes:
            for dims in FLIPS:
                # Transform inputs, run the net, then invert the transform on
                # the output (flips/transposes are involutions).
                a = _t(s0, dims, tr); b = _t(s1, dims, tr); v = _t(sv, dims, tr)
                m = net(torch.cat([a, b, v], 1))[3]
                m = torch.flip(m, dims) if dims else m
                if tr:
                    m = m.transpose(2, 3)
                acc = m if acc is None else acc + m
                n += 1
    return acc / n


@torch.no_grad()
def bisect(net, seq, ts, img0, img1, ia, ib, depth, device, tta=False,
           anchors=None):
    """Return list of (index, tensor) for the 2^depth - 1 midpoints.

    anchors: optional {index: padded tensor} of externally materialized frames
    (e.g. the current best ensemble). When present, deeper recursion levels
    are conditioned on these instead of the model's own midpoint outputs —
    every returned frame is still this model's own prediction.
    """
    if depth == 0:
        return []
    im = (ia + ib) // 2
    # Build the event voxel for exactly this (sub-)interval; timestamps of
    # intermediate frames come from frame_info.txt so bins are exact.
    ev = seq.load_events(ts[ia], ts[ib])
    vox = events_to_voxel(ev, EV_BINS, seq.height, seq.width, ts[ia], ts[ib])
    vox = torch.from_numpy(vox).unsqueeze(0).to(device)
    vox, _, _ = pad_to_32(vox, mode="constant")  # zeros: no events off-frame
    mid = infer_mid(net, img0, img1, vox, tta).clamp(0, 1)
    # Anchored bisection: recurse on the externally materialized ensemble
    # frame when available (more reliable pseudo-anchor), but always RETURN
    # this model's own prediction `mid`.
    anc = anchors.get(im, mid) if anchors else mid
    return (bisect(net, seq, ts, img0, anc, ia, im, depth - 1, device, tta,
                   anchors)
            + [(im, mid)]
            + bisect(net, seq, ts, anc, img1, im, ib, depth - 1, device, tta,
                     anchors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "challenge_data"))
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--skips", default="7skip,15skip")
    ap.add_argument("--output-dir", default=str(ROOT / "results_emae"))
    ap.add_argument("--ckpt",
                    default=str(ROOT / "weights" / "emavfi" / "ours.pkl"),
                    help="pretrained ours.pkl (step-0 sanity) or EMA-E ckpt")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--film", action="store_true", help="load a FiLM-EMA-E checkpoint")
    ap.add_argument("--b4", action="store_true", help="load a B4 (event->flow) checkpoint")
    ap.add_argument("--seqs", default="",
                    help="comma-sep scene names to restrict to (default: all)")
    ap.add_argument("--anchor-dir", default=None,
                    help="materialized ensemble root ({skip}/{seq}/{idx}.png); "
                         "deep bisection levels anchor on these frames")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    device = torch.device("cuda")
    if args.film:
        from emae_film_arch import build_emae_film
        net = build_emae_film(args.ckpt).eval()
    elif args.b4:
        from emae_b4_arch import build_emae_b4
        net = build_emae_b4(args.ckpt).eval()
    else:
        net = build_emae(args.ckpt).eval()

    total_frames = 0
    t_start = time.time()
    for split in args.splits.split(","):
        for skip in args.skips.split(","):
            # "Nskip" means N missing frames between anchors, i.e. anchor
            # stride step = N+1; bisection needs step to be a power of two.
            skip_n = int(skip.replace("skip", ""))
            step = skip_n + 1
            depth = step.bit_length() - 1
            assert 1 << depth == step, f"skip+1={step} is not a power of 2"

            only = set(s for s in args.seqs.split(",") if s)
            skip_dir = data_dir / split / skip
            for seq_dir in sorted(p for p in skip_dir.iterdir() if p.is_dir()):
                if only and seq_dir.name not in only:
                    continue
                seq = load_sequence(data_dir, split, skip, seq_dir.name)
                ts = {f.index: f.timestamp for f in seq.frames}
                inputs = seq.inputs
                todo_idx = {f.index for f in seq.todos}
                t_seq = time.time()
                n_seq = 0
                for a, b in zip(inputs[:-1], inputs[1:]):
                    if b.index - a.index != step:
                        continue  # non-contiguous anchor pair, nothing to fill
                    img0 = to_tensor(seq.load_frame(a), device)
                    img1 = to_tensor(seq.load_frame(b), device)
                    img0, h, w = pad_to_32(img0)
                    img1, _, _ = pad_to_32(img1)
                    # Preload materialized ensemble frames (if any) to serve
                    # as pseudo-anchors for the deeper bisection levels.
                    anchors = None
                    if args.anchor_dir:
                        anchors = {}
                        for i in range(a.index + 1, b.index):
                            p = (Path(args.anchor_dir) / skip / seq.name
                                 / f"{i:06d}.png")
                            if p.exists():
                                img = np.asarray(
                                    Image.open(p).convert("RGB"),
                                    dtype=np.float32) / 255.0
                                t = to_tensor(img, device)
                                t, _, _ = pad_to_32(t)
                                anchors[i] = t
                    mids = bisect(net, seq, ts, img0, img1, a.index, b.index,
                                  depth, device, tta=args.tta, anchors=anchors)
                    for idx, mid in mids:
                        if idx not in todo_idx:
                            continue  # only save frames the challenge asks for
                        # Crop padding away and convert back to HWC [0,1].
                        out = mid[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1)
                        save_result_png(out_dir, skip, seq.name, idx,
                                        out.cpu().numpy().astype(np.float32))
                        n_seq += 1
                total_frames += n_seq
                dt = time.time() - t_seq
                print(f"[{split}/{skip}/{seq.name}] {n_seq} frames in {dt:.1f}s "
                      f"({dt / max(n_seq, 1):.2f}s/frame) "
                      f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                      flush=True)
    print(f"DONE: {total_frames} frames in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
