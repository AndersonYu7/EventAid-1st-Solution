#!/usr/bin/env python3
"""TimeLens-XL TLXNet+ (FinalBidirectionAttenfusion, Expv8_large) inference runner
for the EventAid Frame Interpolation Challenge.

Pipeline role (1st-place solution, team yunyu8, Codabench 16375):
  MEMBER-INFERENCE stage script. TimeLens-XL is an event-based member of
  the 28-member ensemble; it predicts ALL intermediate frames of an anchor
  gap in one forward pass. Pipeline: data prep -> training -> member
  inference (THIS script) -> per-frame quadratic-form fusion
  (recipe_v15/v16.json) -> submission assembly (scripts/build_final.py).

Inputs:
  --data-dir  challenge data root; --weights the TLXNet+ checkpoint (.pt),
              default weights/Expv8_large_HQEVFI.pt
Outputs:
  uint8 PNGs at {output-dir}/{skip}/{seq}/{index:06d}.png per TODO frame,
  consumed by the fusion stage.

Voxelization detail — replicates
dataset/RC_4816/mixloader.py:sample_events_to_grid exactly:
  - per base interval (between consecutive frame timestamps inside an anchor gap)
    a voxel of echannel//interp_ratio channels is built
  - hard temporal binning relative to the FIRST/LAST event of that interval:
      t_step = (t_end - t_start + 1) / voxel_channels; ind = (t - t_start) // t_step
  - bilinear spatial splatting (no-op for integer challenge coords)
  - SIGNED polarity: HQ-EVFI npz stores p in {-1,+1} (evidenced by
    loader_RC_timelens_mix.py negating the reversed left voxel), so challenge
    {0,1} polarity is mapped p -> 2p-1.
Inputs are reflect-padded (right/bottom) to multiples of 32 and outputs cropped
back to the exact shape.txt resolution.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 scripts/run_tlx.py \
      --data-dir challenge_data --output-dir results_tlx_hqevfi \
      --weights weights/Expv8_large_HQEVFI.pt \
      --splits validation --skips 1skip,3skip,7skip,15skip
"""
import argparse
import importlib
import os
import sys
import time
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, 'models', 'TimeLens-XL')
sys.path.insert(0, ROOT)

from evlib.dataset import load_sequence, save_result_png  # noqa: E402

# interp_ratio = anchor stride (skip + 1); ECHANNEL = total event channels
# fed to the network (split as ECHANNEL // interp_ratio bins per base interval).
SKIP_TO_RATIO = {'1skip': 2, '3skip': 4, '7skip': 8, '15skip': 16}
ECHANNEL = 128


def build_net():
    """Construct FinalBidirectionAttenfusion bypassing heavy package __init__.

    The TimeLens-XL repo's models/__init__.py imports every experiment
    (pulling in heavy/broken deps), so we register stub namespace packages
    that point straight at the Expv8_large arch directory and import only
    the arch module we need.
    """
    for name, path in [
        ('tlx_models', f'{REPO}/models'),
        ('tlx_models.Expv8_large', f'{REPO}/models/Expv8_large'),
        ('tlx_models.Expv8_large.archs', f'{REPO}/models/Expv8_large/archs'),
        ('models', f'{REPO}/models'),
        ('models.Expv8_large', f'{REPO}/models/Expv8_large'),
        ('models.Expv8_large.archs', f'{REPO}/models/Expv8_large/archs'),
        ('tools', f'{REPO}/tools'),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    arch_mod = importlib.import_module('models.Expv8_large.archs.XXNet_final_attenfusion_arch')
    cfg = dict(img_chn=6, ev_chn=2, num_encoders=2, base_num_channels=32, out_chn=3,
               num_block=1, num_residual_blocks=2, base_channel=32, echannel=ECHANNEL,
               interp_ratio=16, pos_e=0.2, neg_e=0.2, num_decoder=8,
               type='FinalBidirectionAttenfusion')
    return arch_mod.FinalBidirectionAttenfusion(**cfg)


def load_weights(net, path):
    # Accept both raw state dicts and training checkpoints; strip the
    # trainer's 'net.' key prefix when present.
    ck = torch.load(path, map_location='cpu', weights_only=True)
    state = ck['model_state'] if 'model_state' in ck else ck
    sd = {k[len('net.'):]: v for k, v in state.items() if k.startswith('net.')}
    if not sd:
        sd = state
    net.load_state_dict(sd, strict=True)
    return ck.get('epoch', '?') if isinstance(ck, dict) else '?'


def sample_events_to_grid(voxel_channels, h, w, ev):
    """Exact numpy replication of mixloader.sample_events_to_grid.

    ev: (N,4) [t,x,y,p] with p already SIGNED (-1/+1). Challenge coords are
    integers, so the bilinear spatial splat collapses to the base pixel
    (fractional weights are zero), matching the numba loop output exactly.
    """
    voxel = np.zeros((voxel_channels, h, w), dtype=np.float32)
    if ev.shape[0] == 0:
        return voxel
    # Hard temporal binning anchored on the FIRST/LAST event of the interval
    # (not the frame timestamps) — this matches the training loader exactly.
    t = ev[:, 0]
    t_start, t_end = t[0], t[-1]
    t_step = (t_end - t_start + 1) / voxel_channels
    ind = ((t - t_start) // t_step).astype(np.int64)
    np.clip(ind, 0, voxel_channels - 1, out=ind)
    x = ev[:, 1].astype(np.int64)
    y = ev[:, 2].astype(np.int64)
    p = ev[:, 3].astype(np.float32)
    m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    # Scatter-add signed polarities into (bin, y, x) cells via bincount on
    # flattened indices — vectorized equivalent of the repo's numba loop.
    flat = ind[m] * (h * w) + y[m] * w + x[m]
    voxel.ravel()[:] = np.bincount(flat, weights=p[m], minlength=voxel_channels * h * w).astype(np.float32)
    return voxel


def build_pair_voxel(seq, frames_in_gap, interp_ratio, h, w):
    """frames_in_gap: list of Frame from left anchor to right anchor inclusive
    (interp_ratio+1 frames). One sub-voxel per base interval, concatenated."""
    vc = ECHANNEL // interp_ratio
    voxels = []
    for a, b in zip(frames_in_gap[:-1], frames_in_gap[1:]):
        ev = seq.load_events(a.timestamp, b.timestamp)
        if ev.shape[0]:
            ev = ev.copy()
            ev[:, 3] = ev[:, 3] * 2.0 - 1.0  # {0,1} -> {-1,+1}
        voxels.append(sample_events_to_grid(vc, h, w, ev))
    return np.concatenate(voxels, 0)  # (128, h, w)


def pad_to_mult(x, mult=32):
    """Reflect-pad NCHW tensor on right/bottom to a multiple of `mult`."""
    h, w = x.shape[-2:]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='reflect')
    return x


def run_sequence(net, seq, skip, out_dir, device):
    interp_ratio = SKIP_TO_RATIO[skip]
    frames = sorted(seq.frames, key=lambda f: f.index)
    h, w = seq.height, seq.width
    n_done = 0
    t_seq = time.time()
    i = 0
    # Walk the sequence anchor-to-anchor; each gap is [anchor,
    # interp_ratio-1 TODO frames, anchor] and is solved in ONE forward pass.
    while i < len(frames):
        assert frames[i].is_input, f'expected anchor at position {i}'
        # collect gap: anchor, interp_ratio-1 todos, anchor
        if i + interp_ratio >= len(frames):
            break
        gap = frames[i:i + interp_ratio + 1]
        assert gap[-1].is_input and all(not f.is_input for f in gap[1:-1]), \
            f'unexpected frame types in gap starting at index {gap[0].index}'
        im0 = seq.load_frame(gap[0])
        im1 = seq.load_frame(gap[-1])
        voxel = build_pair_voxel(seq, gap, interp_ratio, h, w)

        # Network input: the two anchors stacked channel-wise (6ch) plus the
        # 128-channel event voxel, both padded to a multiple of 32.
        x = torch.from_numpy(np.concatenate([im0.transpose(2, 0, 1),
                                             im1.transpose(2, 0, 1)], 0))[None].to(device)
        ev = torch.from_numpy(voxel)[None].to(device)
        x = pad_to_mult(x)
        ev = pad_to_mult(ev)
        with torch.no_grad():
            # Request all interp_ratio-1 intermediate frames at once.
            out = net(x, ev, interp_ratio, range(0, interp_ratio - 1))
        res = out[0][0]  # (interp_ratio-1, 3, Hp, Wp)
        res = res[:, :, :h, :w].clamp(0, 1).cpu().numpy()  # crop padding
        for k, fr in enumerate(gap[1:-1]):
            save_result_png(out_dir, skip, seq.name, fr.index,
                            res[k].transpose(1, 2, 0))
            n_done += 1
        i += interp_ratio
    dt = time.time() - t_seq
    print(f'  {skip}/{seq.name}: wrote {n_done} frames in {dt:.1f}s '
          f'({dt / max(n_done, 1):.2f}s/frame), '
          f'peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'challenge_data'))
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--weights', default=os.path.join(ROOT, 'weights', 'Expv8_large_HQEVFI.pt'))
    ap.add_argument('--splits', default='validation')
    ap.add_argument('--skips', default='1skip,3skip,7skip,15skip')
    args = ap.parse_args()

    device = 'cuda'
    net = build_net().to(device).eval()
    ep = load_weights(net, args.weights)
    print(f'loaded {args.weights} (epoch {ep})', flush=True)

    splits = args.splits.split(',')
    skips = args.skips.split(',')
    total = 0
    t0 = time.time()
    for split in splits:
        for skip in skips:
            skip_dir = os.path.join(args.data_dir, split, skip)
            if not os.path.isdir(skip_dir):
                continue
            for name in sorted(os.listdir(skip_dir)):
                if not os.path.isdir(os.path.join(skip_dir, name)):
                    continue
                seq = load_sequence(args.data_dir, split, skip, name)
                total += run_sequence(net, seq, skip, args.output_dir, device)
    print(f'DONE: {total} frames in {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
