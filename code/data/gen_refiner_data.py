#!/usr/bin/env python3
"""Generate training data for the event-conditioned fusion refiner.

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  DATA PREP (this file) -> training -> member inference -> fusion -> submission.
  The learned fusion refiners in the 30-member ensemble learn to
  combine several base-interpolator outputs plus an event voxel into a better
  middle frame. This script manufactures their supervised training set: for
  random (anchor pair, target time) draws it runs the frozen base models
  (GIMM-VFI-F, GIMM-VFI-R, TimeLens) and stores their predictions alongside
  the ground-truth frame and a simulated event voxel.

Each sample (256x256 crop from HQ-EVFI RGB sequences, synthetic events via the
calibrated V2V-style simulator in evlib/v2v_core_esim.py):
  a0, a1     anchor frames           uint8 (256,256,3)
  gt         target middle frame     uint8
  lin        linear interpolation    uint8
  gimm/gimmr GIMM-VFI-F / -R output at the exact target alpha
  tl         TimeLens output (synthetic events fed as an EventSequence)
  vox        16-bin signed voxel over the whole anchor gap, float16
  alpha      target time fraction; skip = number of skipped frames (1/3/7/15)

Inputs:  --data-root HQ-EVFI release (dirs containing visual_RGB/*.png),
         frozen weights under weights/ (gimmvfi_{f,r}_arb.pt,
         timelens_checkpoint.bin).
Outputs: --out directory of sharded npz files, 100 samples per shard
         (keys "<field>_<i>" flattened per shard).

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/gen_refiner_data.py \
      --out data/refiner_shards --n-samples 10000
"""
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models", "rpg_timelens"))
sys.path.insert(0, os.path.join(ROOT, "models", "GIMM-VFI", "src"))

from evlib.v2v_core_esim import EventEmulator  # noqa: E402

SKIP_CHOICES = (1, 3, 7, 15)  # matches the challenge skip settings
CROP = 256
SUB = 8  # temporal substeps per source interval


def slices_to_voxel(slices, n_bins=16):
    """Rebin K signed event-count slices into an n_bins temporal voxel via
    bilinear splitting in time (same formulation as syn_ev_data)."""
    K, H, W = slices.shape
    voxel = np.zeros((n_bins, H, W), dtype=np.float32)
    if K == 1:
        voxel[0] = slices[0]
        return voxel
    for k in range(K):
        tn = k / (K - 1) * (n_bins - 1)
        b0 = int(np.floor(tn))
        dt = tn - b0
        voxel[b0] += slices[k] * (1.0 - dt)
        if b0 + 1 < n_bins:
            voxel[b0 + 1] += slices[k] * dt
    return voxel


def slices_to_events(slices, t0_us, dt_us, rng):
    """Signed count slices (K,H,W) -> discrete events (N,4) [x,y,t_us,p±1].

    TimeLens consumes an explicit event list, not a voxel, so each per-slice
    count of magnitude c is expanded into c individual events with timestamps
    drawn uniformly inside that slice's time interval.
    """
    xs, ys, ts, ps = [], [], [], []
    K = slices.shape[0]
    for k in range(K):
        sl = slices[k]
        nz = np.nonzero(sl)
        if nz[0].size == 0:
            continue
        counts = sl[nz]
        for sign in (1, -1):  # handle positive and negative polarity separately
            m = (counts * sign) > 0
            if not m.any():
                continue
            yy, xx = nz[0][m], nz[1][m]
            cc = np.abs(counts[m]).astype(int)
            # Replicate each pixel |count| times -> one row per event.
            rep_y = np.repeat(yy, cc)
            rep_x = np.repeat(xx, cc)
            n = rep_x.size
            # Random sub-slice timestamps in [t0 + k*dt, t0 + (k+1)*dt).
            tt = t0_us + (k + rng.random(n)) * dt_us
            xs.append(rep_x); ys.append(rep_y); ts.append(tt)
            ps.append(np.full(n, sign, dtype=np.int8))
    if not xs:
        return np.empty((0, 4))
    ev = np.stack([np.concatenate(xs).astype(np.float64),
                   np.concatenate(ys).astype(np.float64),
                   np.concatenate(ts),
                   np.concatenate(ps).astype(np.float64)], axis=1)
    return ev[np.argsort(ev[:, 2], kind="stable")]  # sort by timestamp


# ---------------------------------------------------------------- models
def load_gimm(variant="f"):
    """Load a frozen GIMM-VFI model ('f' = flow-based, 'r' = RAFT-based).

    GIMM-VFI's config loader resolves relative paths against the CWD, so we
    temporarily chdir into its repo while constructing the model.
    """
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, "models", "GIMM-VFI"))
    from models import create_model
    from utils.config import load_config, augment_arch_defaults
    cfg = load_config(os.path.join(ROOT, "models", "GIMM-VFI", "configs",
                                   "gimmvfi", f"gimmvfi_{variant}_arb.yaml"))
    arch = augment_arch_defaults(cfg.arch)
    model, _ = create_model(arch)
    ckpt = torch.load(os.path.join(ROOT, "weights", "gimmvfi",
                                   f"gimmvfi_{variant}_arb.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.cuda().eval()
    os.chdir(cwd)
    return model


@torch.no_grad()
def gimm_infer(model, a0, a1, alpha):
    """a0,a1 uint8 HWC -> uint8 HWC at fraction alpha."""
    t0 = torch.from_numpy(a0).permute(2, 0, 1)[None].float().cuda() / 255.0
    t1 = torch.from_numpy(a1).permute(2, 0, 1)[None].float().cuda() / 255.0
    xs = torch.cat((t0.unsqueeze(2), t1.unsqueeze(2)), dim=2)
    # GIMM-VFI is arbitrary-time: query its implicit motion field at alpha.
    coord = model.sample_coord_input(1, xs.shape[-2:], [alpha], device=xs.device,
                                     upsample_ratio=1.0)
    tt = alpha * torch.ones(1, device=xs.device)
    out = model(xs, [(coord, None)], t=[tt], ds_factor=1.0)["imgt_pred"][0]
    img = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def load_timelens():
    """Load the frozen TimeLens attention-average network + its input
    transformers (5-bin voxel grids on each side of the target time)."""
    from timelens import attention_average_network
    from timelens.common import pytorch_tools, transformers
    net = attention_average_network.AttentionAverage()
    net.from_legacy_checkpoint(os.path.join(ROOT, "weights",
                                            "timelens_checkpoint.bin"))
    net.cuda().eval()
    tl = transformers.initialize_transformers(number_of_bins_in_voxel_grid=5)
    return net, tl


@torch.no_grad()
def timelens_infer(net, tl_transforms, a0, a1, events, t0, t1, t_target):
    """Run TimeLens on one crop: anchors + synthetic event list -> uint8 HWC."""
    from timelens.common import pytorch_tools, transformers
    from timelens.common.event import EventSequence
    feats = np.empty((events.shape[0], 4), dtype=np.float64)
    if events.shape[0]:
        feats[:, 0] = events[:, 0]  # x
        feats[:, 1] = events[:, 1]  # y
        feats[:, 2] = events[:, 2]  # t
        feats[:, 3] = events[:, 3]  # p already ±1
    # TimeLens wants events split at the target time: "before" events warp
    # anchor0 forward, "after" events warp anchor1 backward.
    full = EventSequence(feats, CROP, CROP, start_time=t0, end_time=t1)
    left_ev, right_ev = full.split_in_two(float(t_target))
    weight = (t_target - t0) / (t1 - t0)
    example = {
        "before": {"rgb_image": Image.fromarray(a0), "events": left_ev},
        "middle": {"weight": float(weight)},
        "after": {"rgb_image": Image.fromarray(a1), "events": right_ev},
    }
    example = transformers.apply_transforms(example, tl_transforms)
    example = transformers.collate([example])
    example = pytorch_tools.move_tensors_to_cuda(example)
    frame, _ = net.run_fast(example)
    img = torch.clamp(frame.squeeze(0), 0, 1).permute(1, 2, 0).cpu().numpy()
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "data_release"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "refiner_shards"))
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--shard-size", type=int, default=100)
    ap.add_argument("--start-shard", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Enumerate HQ-EVFI RGB sequences long enough for the largest skip.
    seqs = []
    for d in sorted(glob.glob(os.path.join(args.data_root, "**", "visual_RGB"),
                              recursive=True)):
        frames = sorted(glob.glob(os.path.join(d, "*.png")),
                        key=lambda p: int(os.path.basename(p).split("_")[0]))
        if len(frames) >= 20:
            seqs.append(frames)
    print(f"{len(seqs)} source sequences", flush=True)

    gimm = load_gimm('f')
    gimmr = load_gimm('r')
    tl_net, tl_tf = load_timelens()
    # Seeds offset by --start-shard so parallel workers produce disjoint data.
    rng_np = np.random.default_rng(12345 + args.start_shard)
    rng = random.Random(54321 + args.start_shard)

    shard, shard_idx = [], args.start_shard
    n_done = 0
    t_start = time.time()
    while n_done < args.n_samples:
        # Draw sequence, skip level, anchor pair (a, b) and target index m.
        frames = rng.choice(seqs)
        skip = rng.choice(SKIP_CHOICES)
        step = skip + 1
        if len(frames) < step + 1:
            continue
        a = rng.randrange(0, len(frames) - step)
        b = a + step
        m = rng.randrange(a + 1, b)  # any interior frame, not just the middle
        imgs = [np.asarray(Image.open(frames[i]).convert("RGB"), np.float32)
                for i in range(a, b + 1)]
        H, W = imgs[0].shape[:2]
        if H < CROP or W < CROP:
            continue
        y0 = rng.randrange(0, H - CROP + 1)
        x0 = rng.randrange(0, W - CROP + 1)
        clip = np.stack([im[y0:y0 + CROP, x0:x0 + CROP] for im in imgs])
        gray = clip.mean(axis=3)

        # Temporal upsample: SUB linear substeps per source interval give the
        # event emulator a smooth intensity trajectory to threshold.
        n_src = gray.shape[0]
        up = np.empty(((n_src - 1) * SUB + 1, CROP, CROP), dtype=np.float32)
        for i in range(n_src - 1):
            for s in range(SUB):
                w_ = s / SUB
                up[i * SUB + s] = (1 - w_) * gray[i] + w_ * gray[i + 1]
        up[-1] = gray[-1]

        # Randomized simulator parameters: contrast threshold, optional
        # pos/neg asymmetry, mild shot noise and hot pixels.
        base_thres = rng.uniform(0.2, 1.2)
        asym = rng.uniform(1.0, 1.4)
        emu = EventEmulator(
            pos_thres=base_thres * (asym if rng.random() < 0.5 else 1.0),
            neg_thres=base_thres * (1.0 if rng.random() < 0.5 else asym),
            base_noise_std=rng.uniform(0.0, 0.03),
            hot_pixel_fraction=rng.uniform(0.0, 0.001),
            hot_pixel_std=rng.uniform(0.0, 0.03),
        )
        slices = emu.video_to_voxel(up).astype(np.float32)

        # Timeline in fake microseconds: one source interval = 6667us (~150fps),
        # only relative times matter to TimeLens.
        dt_sub = 6667.0 / SUB
        t0_us = 0.0
        t1_us = (step) * 6667.0
        t_target = (m - a) * 6667.0
        alpha = (m - a) / step

        vox = slices_to_voxel(slices)  # over whole gap
        events = slices_to_events(slices, t0_us, dt_sub, rng_np)

        a0 = np.clip(clip[0], 0, 255).astype(np.uint8)
        a1 = np.clip(clip[-1], 0, 255).astype(np.uint8)
        gt = np.clip(clip[m - a], 0, 255).astype(np.uint8)
        # Trivial baseline member: alpha-weighted average of the anchors.
        lin = np.clip(np.rint((1 - alpha) * clip[0] + alpha * clip[-1]),
                      0, 255).astype(np.uint8)
        try:
            gimm_out = gimm_infer(gimm, a0, a1, alpha)
            gimmr_out = gimm_infer(gimmr, a0, a1, alpha)
            tl_out = timelens_infer(tl_net, tl_tf, a0, a1, events,
                                    t0_us, t1_us, t_target)
        except Exception as e:  # noqa: BLE001
            print("infer fail:", e, flush=True)
            continue

        shard.append(dict(a0=a0, a1=a1, gt=gt, lin=lin, gimm=gimm_out, gimmr=gimmr_out,
                          tl=tl_out, vox=vox.astype(np.float16),
                          alpha=np.float32(alpha), skip=np.int16(skip)))
        n_done += 1
        if len(shard) >= args.shard_size:
            # Flush shard: flatten dict-of-samples to "<field>_<i>" npz keys.
            path = os.path.join(args.out, f"shard_{shard_idx:05d}.npz")
            np.savez_compressed(path, **{
                f"{k}_{i}": s[k] for i, s in enumerate(shard) for k in s})
            shard, shard_idx = [], shard_idx + 1
            rate = n_done / (time.time() - t_start)
            print(f"{n_done}/{args.n_samples} ({rate:.1f} samples/s) -> {path}",
                  flush=True)
    print("generation done", flush=True)


if __name__ == "__main__":
    main()
