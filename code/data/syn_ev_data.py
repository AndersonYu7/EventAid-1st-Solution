#!/usr/bin/env python3
"""Synthetic-event training dataset for the event-grafted VFI members.

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  data prep -> TRAINING (this file) -> member inference -> fusion -> submission.
  This is the training-data source for the EMA-E and VFIMamba-E ensemble
  members: zero-initialized event grafts on EMA-VFI / VFIMamba that consume a
  16-bin event voxel next to the two anchor RGB frames. It is a *standalone*
  module (no model-architecture imports) so that it can be shared by both
  training scripts without triggering the EMA-VFI vs. VFIMamba `model`/`config`
  top-level module-name collision. The sampling logic is identical to
  finetune_emae.SynEvDataset, including the 2026-06-13 event thresholds
  U[0.1, 0.7] calibrated against the real EventAid sensor statistics
  (see scripts/analyze_event_stats.py).

Inputs:
  - HQ-EVFI RGB sequences: <hq_root>/**/visual_RGB/*.png, files named
    "<index>_<timestamp_us>.png".
  - Adobe240 sequences: <adobe_root>/<seq>/*.jpg (240fps video frames).
  Events are NOT read from disk; they are simulated on the fly from the RGB
  frames with the ESIM-style EventEmulator (evlib/v2v_core_esim.py) after 8x
  linear temporal upsampling of the grayscale clip.

Output per sample (matches what finetune scripts expect):
  - imgs9: float32 tensor (9, crop, crop) = channel-concat of
    [anchor0 BGR, anchor1 BGR, ground-truth middle BGR], values in [0, 255].
  - vox:   float32 tensor (EV_BINS, crop, crop) signed event voxel spanning
    the full anchor gap (temporal-bilinear binning of simulator slices).

Usage example (as a library; consumed by the EMA-E finetuning loop):
  from syn_ev_data import SynEvDataset
  ds = SynEvDataset(hq_root="data/HQ-EVFI", adobe_root="data/adobe240/frames",
                    crop=256)
  imgs9, vox = ds[0]
"""
import glob
import math
import os
import random
import sys

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evlib.v2v_core_esim import EventEmulator  # noqa: E402

EV_BINS = int(os.environ.get("EMAE_EVBINS", 16))  # 32 -> finer temporal voxel (B2)
HQ_GAPS = (8, 10, 12, 16)      # anchor gaps (in source frames) for HQ-EVFI clips
ADOBE_GAPS = (12, 16, 20, 24)  # larger gaps for the higher-fps Adobe240 clips
SUB = 8          # temporal substeps per source interval fed to the emulator
THRES_LO = 0.1   # calibrated to real EventAid (nonzero 12%, |max| 5.8)
THRES_HI = 0.7


def slices_to_voxel(slices, n_bins=EV_BINS):
    """Rebin K signed event-count slices into an n_bins temporal voxel.

    Each slice k sits at normalized time k/(K-1); its counts are split
    bilinearly between the two nearest output bins (standard EVFI voxelization).
    """
    K, H, W = slices.shape
    voxel = np.zeros((n_bins, H, W), dtype=np.float32)
    if K == 1:
        voxel[0] = slices[0]
        return voxel
    for k in range(K):
        tn = k / (K - 1) * (n_bins - 1)   # fractional target-bin position
        b0 = int(np.floor(tn)); dt = tn - b0
        voxel[b0] += slices[k] * (1.0 - dt)
        if b0 + 1 < n_bins:
            voxel[b0 + 1] += slices[k] * dt
    return voxel


class HQSeq:
    """Index of one HQ-EVFI RGB directory; filenames are '<idx>_<ts_us>.png'."""

    def __init__(self, rgb_dir):
        frames = []
        for p in glob.glob(os.path.join(rgb_dir, "*.png")):
            idx, ts = os.path.basename(p)[:-4].split("_")
            frames.append((int(idx), int(ts), p))
        frames.sort()
        self.frames = frames
        self.by_idx = {f[0]: f for f in frames}


class SynEvDataset(torch.utils.data.Dataset):
    """Infinite random-crop dataset: (anchor0, anchor1, GT middle) triplets
    plus a simulated event voxel over the anchor gap."""

    def __init__(self, hq_root, adobe_root, crop=256, length=10**6):
        # Enumerate usable source sequences from both corpora up front.
        self.sources = []
        for d in sorted(glob.glob(os.path.join(hq_root, "**", "visual_RGB"),
                                  recursive=True)):
            s = HQSeq(d)
            if len(s.frames) >= max(HQ_GAPS) + 1:
                self.sources.append(("hq", s, HQ_GAPS))
        n_hq = len(self.sources)
        for d in sorted(glob.glob(os.path.join(adobe_root, "*"))):
            if not os.path.isdir(d):
                continue
            frames = sorted(glob.glob(os.path.join(d, "*.jpg")))
            if len(frames) >= max(ADOBE_GAPS) + 1:
                self.sources.append(("adobe", frames, ADOBE_GAPS))
        assert n_hq > 0, hq_root
        print(f"syn-ev dataset: {n_hq} HQ-EVFI + {len(self.sources)-n_hq} Adobe240",
              flush=True)
        self.crop = crop
        self.length = length

    def __len__(self):
        return self.length

    def _sample_clip(self, rng):
        """Draw one contiguous (gap+1)-frame clip; return float32 (g+1,H,W,3)
        BGR stack, or None if the draw is unusable (caller retries)."""
        kind, payload, gaps = self.sources[rng.randrange(len(self.sources))]
        g = rng.choice(gaps)
        if kind == "hq":
            seq = payload
            pos = rng.randrange(0, len(seq.frames))
            i0 = seq.frames[pos][0]
            # Require every frame index in [i0, i0+g] to exist (no dropped frames).
            if any(i0 + k not in seq.by_idx for k in range(g + 1)):
                return None
            f0, fm, f1 = (seq.by_idx[i0], seq.by_idx[i0 + g // 2], seq.by_idx[i0 + g])
            # Reject clips whose middle frame is not temporally centered
            # (>10% off), so gt truly corresponds to alpha = 0.5.
            span = f1[1] - f0[1]
            if span <= 0 or abs((fm[1] - f0[1]) - span / 2) > 0.1 * span:
                return None
            paths = [seq.by_idx[i0 + k][2] for k in range(g + 1)]
        else:
            frames = payload
            a = rng.randrange(0, len(frames) - g)
            paths = frames[a:a + g + 1]
        imgs = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                return None
            imgs.append(im)
        if any(im.shape != imgs[0].shape for im in imgs):
            return None
        return np.stack(imgs).astype(np.float32)

    def __getitem__(self, _):
        # Fresh unseeded RNG per call -> workers do not repeat each other.
        rng = random.Random()
        c = self.crop
        for _attempt in range(50):
            clip = self._sample_clip(rng)
            if clip is None:
                continue
            n, H, W = clip.shape[:3]
            g = n - 1
            if H < c or W < c:
                continue
            # Random spatial crop shared by frames and simulated events.
            y0 = rng.randrange(0, H - c + 1); x0 = rng.randrange(0, W - c + 1)
            clip = clip[:, y0:y0 + c, x0:x0 + c]
            gray = clip.mean(axis=3)
            # Linear temporal upsampling (SUB substeps per source interval):
            # gives the emulator a smoother intensity trajectory to threshold.
            up = np.empty((g * SUB + 1, c, c), dtype=np.float32)
            for i in range(g):
                for s in range(SUB):
                    w_ = s / SUB
                    up[i * SUB + s] = (1 - w_) * gray[i] + w_ * gray[i + 1]
            up[-1] = gray[-1]
            # Randomized simulator parameters: contrast threshold in the
            # EventAid-calibrated range, optional pos/neg asymmetry, plus
            # mild shot/hot-pixel noise for robustness.
            base_thres = rng.uniform(THRES_LO, THRES_HI)
            asym = rng.uniform(1.0, 1.4)
            emu = EventEmulator(
                pos_thres=base_thres * (asym if rng.random() < 0.5 else 1.0),
                neg_thres=base_thres * (1.0 if rng.random() < 0.5 else asym),
                base_noise_std=rng.uniform(0.0, 0.03),
                hot_pixel_fraction=rng.uniform(0.0, 0.001),
                hot_pixel_std=rng.uniform(0.0, 0.03))
            slices = emu.video_to_voxel(up).astype(np.float32)
            vox = slices_to_voxel(slices)
            img0, gt, img1 = clip[0], clip[g // 2], clip[g]
            # Voxel-consistent augmentations (each with prob 0.5):
            # channel swap (BGR<->RGB) -- events are grayscale, voxel unchanged.
            if rng.random() < 0.5:
                img0, gt, img1 = img0[:, :, ::-1], gt[:, :, ::-1], img1[:, :, ::-1]
            # time reversal: swap anchors, flip voxel bin order AND polarity.
            if rng.random() < 0.5:
                img0, img1 = img1, img0
                vox = -vox[::-1]
            # vertical flip (images + voxel rows).
            if rng.random() < 0.5:
                img0, gt, img1 = img0[::-1], gt[::-1], img1[::-1]
                vox = vox[:, ::-1]
            # horizontal flip (images + voxel columns).
            if rng.random() < 0.5:
                img0, gt, img1 = img0[:, ::-1], gt[:, ::-1], img1[:, ::-1]
                vox = vox[:, :, ::-1]
            imgs9 = np.concatenate([img0, img1, gt], axis=2)
            return (torch.from_numpy(np.ascontiguousarray(imgs9.transpose(2, 0, 1))),
                    torch.from_numpy(np.ascontiguousarray(vox)))
        raise RuntimeError("could not sample a valid triplet after 50 tries")


def lr_mult(step, steps, warmup=200):
    """LR schedule shared by the finetune scripts: linear warmup for `warmup`
    steps, then cosine decay from 1.0 down to a 0.1 floor."""
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(steps - warmup, 1)
    return 0.1 + 0.9 * (math.cos(math.pi * t) * 0.5 + 0.5)
