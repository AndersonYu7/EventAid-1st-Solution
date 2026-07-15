#!/usr/bin/env python3
"""REAL EventAid-camera events from data_release (RGB-EVS), the val-domain bet.

The challenge corpus data_release ships, next to each visual_RGB frame, a
RGB-EVS/<idx>_<ts>.npz holding the REAL events for that frame interval
(keys x,y,p,t in pixel coords, p in {-1,+1}). SynEvDataset ignored these and
SIMULATED events from the RGB instead -> a sim-real gap on room1 (validation),
which is the SAME EventAid camera. This dataset trains EMA-E on the REAL events,
exactly the move that made ERF a fast-scene specialist, but in the val domain.

Sample = (imgs9 = cat[img0,img1,gt], 16-bin real-event voxel), identical to
ERFDataset/SynEvDataset so the trainer loop is unchanged.
"""
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
EV_BINS = int(os.environ.get("EMAE_EVBINS", 16))
GAPS = (4, 8, 12, 16)


def _idx(path):
    return int(os.path.basename(path).split("_")[0])


def events_to_voxel_crop(npz_files, y0, x0, c, nbins=EV_BINS):
    """Accumulate real events over the gap, crop [y0:y0+c, x0:x0+c],
    build (nbins,c,c) temporal-bilinear voxel (polarity +/-1)."""
    xs, ys, ps, ts = [], [], [], []
    for f in npz_files:
        d = np.load(f)
        if d["x"].size == 0:
            continue
        xs.append(d["x"].astype(np.float64))
        ys.append(d["y"].astype(np.float64))
        ps.append(np.where(d["p"] > 0, 1.0, -1.0))
        ts.append(d["t"].astype(np.float64))
    vox = np.zeros((nbins, c, c), np.float32)
    if not xs:
        return vox
    x = np.concatenate(xs); y = np.concatenate(ys)
    p = np.concatenate(ps); t = np.concatenate(ts)
    m = (x >= x0) & (x < x0 + c) & (y >= y0) & (y < y0 + c)
    if not m.any():
        return vox
    x, y, p, t = x[m] - x0, y[m] - y0, p[m], t[m]
    t0, t1 = t.min(), t.max()
    tn = (t - t0) / (t1 - t0) * (nbins - 1) if t1 > t0 else np.zeros_like(t)
    xi = np.clip(x.astype(int), 0, c - 1); yi = np.clip(y.astype(int), 0, c - 1)
    b0 = np.floor(tn).astype(int); dt = tn - b0
    np.add.at(vox, (b0, yi, xi), (p * (1 - dt)).astype(np.float32))
    mm = b0 + 1 < nbins
    np.add.at(vox, (b0[mm] + 1, yi[mm], xi[mm]), (p * dt)[mm].astype(np.float32))
    return vox


class MixedDataset(torch.utils.data.Dataset):
    """Sample REAL EventAid events with prob p_real, else SIMULATED (diverse
    content). Real -> sensor realism; synthetic -> content diversity. Lets the
    model learn real-event handling without overfitting the limited 'hand'
    content, so it can train longer than pure-real (which decays)."""

    def __init__(self, real_ds, syn_ds, p_real=0.5, length=10**6):
        self.real, self.syn, self.p, self.length = real_ds, syn_ds, p_real, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.real[i] if random.Random().random() < self.p else self.syn[i]


class RealEVSDataset(torch.utils.data.Dataset):
    def __init__(self, root, crop=256, length=10**6, gaps=GAPS):
        self.seqs = []
        for d in sorted(glob.glob(os.path.join(root, "**", "visual_RGB"),
                                  recursive=True)):
            base = os.path.dirname(d)
            evs = os.path.join(base, "RGB-EVS")
            if not os.path.isdir(evs):
                continue
            imgs = {_idx(p): p for p in glob.glob(os.path.join(d, "*.png"))}
            evdict = {_idx(p): p for p in glob.glob(os.path.join(evs, "*.npz"))}
            common = sorted(set(imgs) & set(evdict))
            if len(common) >= max(gaps) + 1:
                self.seqs.append((imgs, evdict, common))
        assert self.seqs, root
        print(f"RealEVS dataset: {len(self.seqs)} sessions", flush=True)
        self.crop = crop; self.length = length; self.gaps = gaps

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = random.Random()
        c = self.crop
        for _try in range(50):
            imgs, evdict, common = rng.choice(self.seqs)
            g = rng.choice(self.gaps)
            pos = rng.randint(0, len(common) - g - 1)
            ia, im_, ib = common[pos], common[pos + g // 2], common[pos + g]
            i0 = cv2.imread(imgs[ia]); im = cv2.imread(imgs[im_]); i1 = cv2.imread(imgs[ib])
            if i0 is None or im is None or i1 is None:
                continue
            H, W = i0.shape[:2]
            if im.shape[:2] != (H, W) or i1.shape[:2] != (H, W) or H < c or W < c:
                continue
            y0 = rng.randrange(0, H - c + 1); x0 = rng.randrange(0, W - c + 1)
            img0 = i0[y0:y0+c, x0:x0+c].astype(np.float32)
            gt = im[y0:y0+c, x0:x0+c].astype(np.float32)
            img1 = i1[y0:y0+c, x0:x0+c].astype(np.float32)
            # events in (ia, ib]: npz for frames pos+1 .. pos+g
            npz = [evdict[common[pos + k]] for k in range(1, g + 1)]
            vox = events_to_voxel_crop(npz, y0, x0, c)
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
        raise RuntimeError("RealEVS sampling failed")
