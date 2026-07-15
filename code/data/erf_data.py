#!/usr/bin/env python3
"""ERF-X170FPS REAL-event training dataset (no event simulation).

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  data prep -> TRAINING (this file) -> member inference -> fusion -> submission.
  Training-data source for the "real-event specialist" EMA-E variants of the
  30-member ensemble. Those specialists are routed at inference time (by event
  density) to fast/medium-motion scenes, where real-event training transfers
  best. This is the data-side bet of the solution: instead of SIMULATING
  events from RGB (which suffers a sim-to-real gap), train EMA-E on ERF's REAL
  sensor events, whose statistics (voxel |max| ~5.9 vs EventAid 5.78, measured
  with scripts/analyze_event_stats.py) closely match the challenge camera.

Each sample mirrors SynEvDataset's output exactly
(imgs9 = channel-concat[img0, img1, gt], 16-bin signed voxel), so the
finetune_emae training loop runs unchanged on either dataset.

Inputs -- ERF-X170FPS layout (per extracted sequence dir):
  processed_images/NNNNN.png   frames @170fps, 975x1440
  processed_events/NNNNN.npz   events in [frame N, frame N+1): keys x,y,p,t
  Event coordinates are stored at 128x sub-pixel resolution -> divide by 128
  to get image-pixel coordinates.

Outputs per sample:
  - imgs9: float32 tensor (9, crop, crop), [anchor0, anchor1, GT middle] BGR
    stacked on channels, values in [0, 255].
  - vox:   float32 tensor (EV_BINS, crop, crop), signed real-event voxel over
    the full anchor gap, cropped to the same window as the images.

Usage example (as a library; consumed by the EMA-E finetuning loop):
  from erf_data import ERFDataset
  ds = ERFDataset(root="data/ERF_X170FPS", crop=256)
  imgs9, vox = ds[0]
"""
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
EV_BINS = int(os.environ.get("EMAE_EVBINS", 16))  # 32 -> finer temporal voxel (B2)
GAPS = (8, 12, 16)          # large-motion regime (challenge needs 7/15skip)
SUBPIX = 128.0              # ERF event coords are 128x sub-pixel


def events_to_voxel_crop(npz_files, y0, x0, c, nbins=EV_BINS):
    """Accumulate ERF events over the gap, keep crop [y0:y0+c, x0:x0+c],
    build an (nbins,c,c) temporal-bilinear voxel (polarity +/-1)."""
    # Concatenate all per-interval npz chunks covering the anchor gap.
    xs, ys, ps, ts = [], [], [], []
    for f in npz_files:
        d = np.load(f)
        if d["x"].size == 0:
            continue
        xs.append(d["x"].astype(np.float64) / SUBPIX)  # sub-pixel -> pixel
        ys.append(d["y"].astype(np.float64) / SUBPIX)
        ps.append(np.where(d["p"] > 0, 1.0, -1.0))     # polarity -> {+1,-1}
        ts.append(d["t"].astype(np.float64))
    vox = np.zeros((nbins, c, c), np.float32)
    if not xs:
        return vox
    x = np.concatenate(xs); y = np.concatenate(ys)
    p = np.concatenate(ps); t = np.concatenate(ts)
    # Keep only events inside the spatial crop window.
    m = (x >= x0) & (x < x0 + c) & (y >= y0) & (y < y0 + c)
    if not m.any():
        return vox
    x, y, p, t = x[m] - x0, y[m] - y0, p[m], t[m]
    # Normalize timestamps to [0, nbins-1] and split each event bilinearly
    # between its two nearest temporal bins (standard EVFI voxelization).
    t0, t1 = t.min(), t.max()
    tn = (t - t0) / (t1 - t0) * (nbins - 1) if t1 > t0 else np.zeros_like(t)
    xi = np.clip(x.astype(int), 0, c - 1); yi = np.clip(y.astype(int), 0, c - 1)
    b0 = np.floor(tn).astype(int); dt = tn - b0
    np.add.at(vox, (b0, yi, xi), (p * (1 - dt)).astype(np.float32))
    mm = b0 + 1 < nbins  # upper-bin share, only where it stays in range
    np.add.at(vox, (b0[mm] + 1, yi[mm], xi[mm]), (p * dt)[mm].astype(np.float32))
    return vox


class ERFDataset(torch.utils.data.Dataset):
    """Anchor pair + GT middle + REAL-event voxel from ERF sequences."""

    def __init__(self, root, crop=256, length=10**6, gaps=GAPS):
        # Index every extracted ERF sequence that has both frames and events
        # and is long enough for the largest anchor gap.
        self.seqs = []
        for d in sorted(glob.glob(os.path.join(root, "**", "processed_images"),
                                  recursive=True)):
            base = os.path.dirname(d)
            imgs = sorted(glob.glob(os.path.join(d, "*.png")))
            evs = os.path.join(base, "processed_events")
            if len(imgs) >= max(gaps) + 1 and os.path.isdir(evs):
                idxs = [int(os.path.basename(p)[:-4]) for p in imgs]
                self.seqs.append((base, set(idxs), min(idxs), max(idxs)))
        assert self.seqs, root
        print(f"ERF dataset: {len(self.seqs)} sequences", flush=True)
        self.crop = crop; self.length = length; self.gaps = gaps

    def __len__(self):
        return self.length

    def _load_img(self, base, idx):
        p = os.path.join(base, "processed_images", f"{idx:05d}.png")
        im = cv2.imread(p)
        return im

    def __getitem__(self, _):
        # Fresh unseeded RNG per call -> dataloader workers do not repeat.
        rng = random.Random()
        c = self.crop
        for _try in range(50):
            base, idxs, lo, hi = rng.choice(self.seqs)
            g = rng.choice(self.gaps)
            a = rng.randint(lo, hi - g)
            mid = a + g // 2
            # All three frames (anchors + middle GT) must exist on disk.
            if not ({a, mid, a + g} <= idxs):
                continue
            i0 = self._load_img(base, a); im = self._load_img(base, mid)
            i1 = self._load_img(base, a + g)
            if i0 is None or im is None or i1 is None:
                continue
            H, W = i0.shape[:2]
            if H < c or W < c:
                continue
            y0 = rng.randrange(0, H - c + 1); x0 = rng.randrange(0, W - c + 1)
            img0 = i0[y0:y0 + c, x0:x0 + c].astype(np.float32)
            gt = im[y0:y0 + c, x0:x0 + c].astype(np.float32)
            img1 = i1[y0:y0 + c, x0:x0 + c].astype(np.float32)
            # Event chunk k covers [frame k, frame k+1) -> take chunks a..a+g-1.
            npz = [os.path.join(base, "processed_events", f"{k:05d}.npz")
                   for k in range(a, a + g) if k in idxs
                   and os.path.exists(os.path.join(base, "processed_events",
                                                    f"{k:05d}.npz"))]
            vox = events_to_voxel_crop(npz, y0, x0, c)
            # Augmentations (voxel-consistent), mirroring SynEvDataset:
            # BGR<->RGB swap; time reversal (swap anchors, flip bins+polarity);
            # vertical flip; horizontal flip -- each with prob 0.5.
            if rng.random() < 0.5:
                img0, gt, img1 = img0[:, :, ::-1], gt[:, :, ::-1], img1[:, :, ::-1]
            if rng.random() < 0.5:
                img0, img1 = img1, img0; vox = -vox[::-1]
            if rng.random() < 0.5:
                img0, gt, img1 = img0[::-1], gt[::-1], img1[::-1]; vox = vox[:, ::-1]
            if rng.random() < 0.5:
                img0, gt, img1 = img0[:, ::-1], gt[:, ::-1], img1[:, ::-1]
                vox = vox[:, :, ::-1]
            imgs9 = np.concatenate([img0, img1, gt], axis=2)
            return (torch.from_numpy(np.ascontiguousarray(imgs9.transpose(2, 0, 1))),
                    torch.from_numpy(np.ascontiguousarray(vox)))
        raise RuntimeError("ERF sampling failed")
