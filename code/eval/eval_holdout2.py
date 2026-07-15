#!/usr/bin/env python3
"""Held-out ERF fast-motion benchmark with EXTENDED evaluation criteria.

Role in the 1st-place EventAid-F pipeline (team yunyu8)
-------------------------------------------------------
Stage: MEMBER INFERENCE / model selection (feeds the FUSION stage). The final
ensemble routes ERF-X170FPS real-event specialist checkpoints (EMA-E, i.e.
zero-init event grafts on EMA-VFI) into fast/medium-motion scenes by event
density. This script is the independent benchmark that justified that design:
it evaluates specialist checkpoints on a held-out ERF split that was NEVER
used for training, so every number here is judged locally, independent of the
competition test server (no submission budget spent).

On that split it adds, beyond plain PSNR:
  - SSIM (skimage) and LPIPS (AlexNet, perceptual) alongside PSNR
  - MOTION-STRATIFIED PSNR: pixels are bucketed by Farneback optical-flow
    magnitude between the two anchor frames (static / medium / fast), which
    shows exactly where a specialist's gain lives
  - a FLOW-ADAPTIVE BLEND pseudo-model: per-pixel weight
    w = sigmoid((|flow| - K) / S), pred = w * erf + (1 - w) * syn -- the
    prototype of the synthetic/real-event routing used in the final fusion
  - configurable interpolation gaps (e.g. add gap 20 for extreme motion)

Inputs
------
- --root DIR : held-out ERF data; each sequence dir contains
  processed_images/{idx:05d}.png and processed_events/{idx:05d}.npz
- --ckpts label=path [...] : EMA-E (or EMA-E-B4 event->flow) checkpoints
- --blend SYN ERF : labels of two loaded checkpoints to flow-adaptive blend
- --gaps / --crop / --stride / --tta : evaluation protocol knobs

Outputs
-------
- stdout table: per model (and the optional "blend" pseudo-model) PSNR, SSIM,
  LPIPS, plus motion-stratified PSNR for static/medium/fast pixels.

Usage
-----
    python3 eval_holdout2.py --ckpts syn=ckpt_syn.pth erf=ckpt_erf.pth \
        --blend syn erf --gaps 8 16 --tta
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make the repo root and scripts/ importable for the project-local modules
# below (architecture builders, event voxelization, inference helpers).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from emae_arch import build_emae                      # noqa: E402
from erf_data import events_to_voxel_crop             # noqa: E402
from run_emae import infer_mid, pad_to_32             # noqa: E402
import cv2                                            # noqa: E402
from skimage.metrics import structural_similarity as ssim_fn  # noqa: E402
import lpips as lpips_mod                             # noqa: E402

# Keep OpenCV single-threaded: Farneback flow otherwise fights torch for cores.
cv2.setNumThreads(1)

# flow-magnitude bucket edges (pixels) for motion-stratified PSNR
MBUCKETS = [(0, 2, "static"), (2, 8, "medium"), (8, 1e9, "fast")]
# blend params: per-pixel ERF weight = sigmoid((|flow|-K)/S)
BLEND_K, BLEND_S = 5.0, 3.0


# Load an EMA-E checkpoint, auto-detecting the B4 (event->flow) variant.
def load_ckpt(path, device):
    sd = torch.load(path, map_location="cpu")
    # Training checkpoints wrap weights under "model"; raw state dicts don't.
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    if any("ev_flow" in k for k in sd):          # B4 (event->flow) checkpoint
        from emae_b4_arch import build_emae_b4
        net = build_emae_b4().to(device).eval()
    else:
        net = build_emae().to(device).eval()
    net.load_state_dict(sd, strict=False)        # attn_mask/HW are recomputed buffers
    return net


# Yield start indices of [a, a+gap] anchor windows, stepping by stride.
def windows(lo, hi, gap, stride):
    a = lo
    while a + gap <= hi:
        yield a
        a += stride


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/erf_holdout"))
    ap.add_argument("--ckpts", nargs="+", required=True, help="label=path")
    ap.add_argument("--blend", nargs=2, metavar=("SYN", "ERF"),
                    help="labels of two loaded ckpts to flow-adaptive blend")
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    device = "cuda"

    # Load every requested checkpoint under its user-given label.
    nets = {}
    for spec in args.ckpts:
        lab, p = spec.split("=", 1)
        nets[lab] = load_ckpt(p, device)
    lpips_net = lpips_mod.LPIPS(net="alex").to(device).eval()

    # The flow-adaptive blend is scored as an extra pseudo-model ("blend").
    labels = list(nets)
    if args.blend:
        labels.append("blend")

    # Discover held-out sequences by their processed_images/ folder.
    seqs = sorted(glob.glob(os.path.join(args.root, "**", "processed_images"),
                            recursive=True))
    c = args.crop
    # accumulators: per label -> dict(se=sum, n=count, ssim=sum, lpips=sum,
    #   mbuckets={name:[se,npix]})
    def newacc():
        return dict(se=0.0, n=0, ssim=0.0, lp=0.0,
                    mb={nm: [0.0, 0] for _, _, nm in MBUCKETS})
    agg = {lab: newacc() for lab in labels}

    def score(lab, pred, gt, flowmag):
        # Accumulate all metrics for one predicted crop into agg[lab].
        a = agg[lab]
        a["se"] += ((pred - gt) ** 2).mean()
        a["n"] += 1
        a["ssim"] += ssim_fn(gt, pred, channel_axis=2, data_range=255)
        # LPIPS expects NCHW tensors in [-1, 1].
        pt = torch.from_numpy(pred.transpose(2, 0, 1)[None]).to(device) / 127.5 - 1
        gt_t = torch.from_numpy(gt.transpose(2, 0, 1)[None]).to(device) / 127.5 - 1
        a["lp"] += float(lpips_net(pt, gt_t))
        d = ((pred - gt) ** 2).mean(2)  # per-pixel SE (mean over channels)
        # Route each pixel's squared error into its motion bucket so PSNR can
        # later be reported separately for static/medium/fast regions.
        for lo, hi, nm in MBUCKETS:
            m = (flowmag >= lo) & (flowmag < hi)
            if m.any():
                a["mb"][nm][0] += d[m].sum()
                a["mb"][nm][1] += int(m.sum())

    for d in seqs:
        base = os.path.dirname(d)
        evs = os.path.join(base, "processed_events")
        imgs = sorted(glob.glob(os.path.join(d, "*.png")))
        idxs = set(int(os.path.basename(p)[:-4]) for p in imgs)
        if not idxs:
            continue
        lo, hi = min(idxs), max(idxs)
        # All frames are evaluated on a fixed center crop of size c x c.
        H, W = cv2.imread(imgs[0]).shape[:2]
        y0, x0 = (H - c) // 2, (W - c) // 2
        for g in args.gaps:
            for a in windows(lo, hi, g, args.stride):
                # Anchors are frames a and a+g; the model must reconstruct the
                # true middle frame (mid), which serves as ground truth.
                mid = a + g // 2
                if not ({a, mid, a + g} <= idxs):
                    continue
                i0 = cv2.imread(os.path.join(d, f"{a:05d}.png"))
                im = cv2.imread(os.path.join(d, f"{mid:05d}.png"))
                i1 = cv2.imread(os.path.join(d, f"{a+g:05d}.png"))
                if i0 is None or im is None or i1 is None:
                    continue
                img0 = i0[y0:y0+c, x0:x0+c].astype(np.float32)
                gt = im[y0:y0+c, x0:x0+c].astype(np.float32)
                img1 = i1[y0:y0+c, x0:x0+c].astype(np.float32)
                # flow magnitude between anchors (proxy for local motion)
                g0 = cv2.cvtColor(img0.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 25,
                                                  3, 5, 1.2, 0)
                flowmag = np.linalg.norm(fl, axis=2)
                # Gather the event packets covering the whole [a, a+g) window
                # and rasterize them into the model's voxel-grid input, cropped
                # to the same center window as the frames.
                npz = [os.path.join(evs, f"{k:05d}.npz")
                       for k in range(a, a + g)
                       if os.path.exists(os.path.join(evs, f"{k:05d}.npz"))]
                vox = events_to_voxel_crop(npz, y0, x0, c)
                # To NCHW tensors in [0, 1]; pad_to_32 satisfies the network's
                # stride-32 constraint (h, w remember the original size so the
                # prediction can be un-padded after inference).
                t0 = torch.from_numpy(img0.transpose(2, 0, 1)[None]).to(device) / 255.
                t1 = torch.from_numpy(img1.transpose(2, 0, 1)[None]).to(device) / 255.
                tv = torch.from_numpy(vox[None]).to(device)
                t0, h, w = pad_to_32(t0); t1, _, _ = pad_to_32(t1)
                tv, _, _ = pad_to_32(tv, mode="constant")
                preds = {}
                for lab, net in nets.items():
                    # Predict the middle frame (optionally flip-TTA averaged),
                    # crop the padding off, and score in 0-255 float space.
                    pr = infer_mid(net, t0, t1, tv, args.tta)[:, :, :h, :w]
                    preds[lab] = (pr[0].clamp(0, 1).cpu().numpy()
                                  .transpose(1, 2, 0) * 255.)
                    score(lab, preds[lab], gt, flowmag)
                if args.blend:
                    # Flow-adaptive blend: fast-moving pixels trust the ERF
                    # (real-event) specialist, static pixels the synthetic one.
                    syn, erf = args.blend
                    wgt = 1 / (1 + np.exp(-(flowmag - BLEND_K) / BLEND_S))
                    wgt = wgt[..., None]
                    bl = wgt * preds[erf] + (1 - wgt) * preds[syn]
                    score("blend", bl, gt, flowmag)

    # Report. Global PSNR comes from the pooled MSE over all crops; the
    # motion-stratified PSNRs come from each bucket's pooled per-pixel SE.
    print(f"=== Held-out ERF EXTENDED (crop {c}, {len(seqs)} seqs, "
          f"gaps {args.gaps}) ===")
    print(f"{'model':>10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} | "
          f"{'static':>8} {'medium':>8} {'fast':>8}  (motion-PSNR)")
    for lab in labels:
        a = agg[lab]
        psnr = 20 * np.log10(255 / np.sqrt(a["se"] / a["n"]))
        ss = a["ssim"] / a["n"]
        lp = a["lp"] / a["n"]
        mp = []
        for _, _, nm in MBUCKETS:
            s, n = a["mb"][nm]
            mp.append(20 * np.log10(255 / np.sqrt(s / n)) if n else float("nan"))
        print(f"{lab:>10} {psnr:8.3f} {ss:8.4f} {lp:8.4f} | "
              f"{mp[0]:8.3f} {mp[1]:8.3f} {mp[2]:8.3f}")


if __name__ == "__main__":
    main()
