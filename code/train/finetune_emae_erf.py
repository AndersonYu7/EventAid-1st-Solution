#!/usr/bin/env python3
"""Fine-tune EMA-E on ERF-X170FPS REAL events (the data-side sim-real bet).

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
TRAINING-stage script (pipeline: data prep -> **training** -> member
inference -> fusion -> submission).  It produces the ERF-trained "real-event
specialist" EMA-E checkpoints that the final ensemble routes (by event
density) to fast/medium-motion scenes — one of the key +dB contributors of
the winning submission.

Identical to finetune_emae.py except the dataset is ERFDataset (real ERF
events) instead of SynEvDataset (simulated). ERF's sensor stats (|max|~5.9)
match EventAid (5.78), so this trains on the right sensor domain rather than
an approximate simulator.  Extra knobs vs. finetune_emae.py: architecture
variants (base / b4 event->flow / FiLM), dataset selection (erf / syn /
real EventAid voxels / mixed), backbone freezing, and a multi-scale
Charbonnier + census loss option.

Inputs:
  * --erf-root  prepared ERF-X170FPS training crops (see data-prep scripts)
  * --init      pretrained EMA-VFI (or EMA-E) weights to start from
Outputs:
  * <out-dir>/emae_step{N}.pkl state dicts consumed by member inference.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/finetune_emae_erf.py \
      --erf-root data/erf_train --init weights/emavfi/ours.pkl \
      --out-dir checkpoints/emae_erf --steps 8000 --batch 8
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

# Repo root; expose repo modules, helper scripts, and the EMA-VFI submodule.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "models", "EMA-VFI"))

import torch.nn.functional as F  # noqa: E402
from emae_arch import build_emae  # noqa: E402


def _charb(x, y, eps=1e-3):
    """Charbonnier (smooth-L1) penalty; eps keeps the gradient finite at 0."""
    return torch.sqrt((x - y) ** 2 + eps ** 2).mean()


def charbonnier_ms(x, y, levels=3):
    """Multi-scale Charbonnier (champion VFI-challenge recipe)."""
    loss, p, g = 0.0, x, y
    for i in range(levels):
        loss = loss + _charb(p, g)
        if i < levels - 1:
            p, g = F.avg_pool2d(p, 2), F.avg_pool2d(g, 2)
    return loss / levels


def census_loss(x, y):
    """Gradient/census term — helps SSIM without hurting PSNR."""
    def gray(t):
        # ITU-R BT.601 luma weights.
        return (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)
    gx, gy = gray(x), gray(y)
    # Charbonnier on horizontal + vertical finite-difference gradients.
    return (_charb(gx[:, :, :, 1:] - gx[:, :, :, :-1], gy[:, :, :, 1:] - gy[:, :, :, :-1])
            + _charb(gx[:, :, 1:] - gx[:, :, :-1], gy[:, :, 1:] - gy[:, :, :-1]))
# These imports sit below the loss helpers because emae_* modules extend
# sys.path side-effectfully; kept as-is to preserve import order.
from emae_b4_arch import build_emae_b4  # noqa: E402
from emae_film_arch import build_emae_film  # noqa: E402
from erf_data import ERFDataset  # noqa: E402
from syn_ev_data import lr_mult, SynEvDataset  # noqa: E402  (standalone)

# Architecture variants: base graft, b4 (events feed the flow path), FiLM
# (events modulate backbone features via scale/shift).
BUILDERS = {"base": build_emae, "b4": build_emae_b4, "film": build_emae_film}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--erf-root", default=os.path.join(ROOT, "data", "erf_train"))
    ap.add_argument("--init", default=os.path.join(ROOT, "weights", "emavfi", "ours.pkl"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints", "emae_erf"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-new", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gaps", type=int, nargs="+", default=None,
                    help="ERF frame gaps to sample (default 8 12 16)")
    ap.add_argument("--arch", choices=list(BUILDERS), default="base",
                    help="base | b4 (event->flow path) | film")
    ap.add_argument("--dataset", choices=["erf", "syn", "realevs", "mix"], default="erf",
                    help="erf | syn | realevs (real EventAid) | mix (real+syn)")
    ap.add_argument("--mix-real-prob", type=float, default=0.5)
    ap.add_argument("--realevs-root", default=os.path.join(
        ROOT, "data", "data_release"))
    ap.add_argument("--evenc-fast", action="store_true",
                    help="give event-encoder the high (new) LR even if in init")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="train only the event encoder (no content overfit)")
    ap.add_argument("--loss", choices=["lap", "char_ms"], default="lap",
                    help="lap (Laplacian) | char_ms (multi-scale Charbonnier + census)")
    ap.add_argument("--hq-root", default=os.path.join(ROOT, "data", "data_release"))
    ap.add_argument("--adobe-root",
                    default=os.path.join(ROOT, "data", "corpora", "Adobe240_frames"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    net = BUILDERS[args.arch](args.init); net.train()
    from model.loss import LapLoss
    lap = LapLoss()

    # "new" = params absent from the init checkpoint (freshly added event
    # modules) -> high LR; everything restored from init -> low LR.
    init_sd = torch.load(args.init, map_location="cpu")
    init_keys = set(k.replace("module.", "") for k in init_sd)
    # new (high LR) = params absent from init OR (when --evenc-fast) the event
    # encoder, so it re-adapts faster to a new event domain.
    def is_new(n):
        if n not in init_keys:
            return True
        return args.evenc_fast and ("ev_encoder" in n or "ev_flow" in n)
    if args.freeze_backbone:
        # adapt ONLY the event encoder to real events; keep the well-generalized
        # appearance/flow backbone fixed -> real-sensor adaptation w/o content overfit
        for n, p in net.named_parameters():
            if not ("ev_encoder" in n or "ev_flow" in n):
                p.requires_grad_(False)
    new = [p for n, p in net.named_parameters() if is_new(n) and p.requires_grad]
    old = [p for n, p in net.named_parameters() if not is_new(n) and p.requires_grad]
    print(f"arch={args.arch}: {len(new)} new param tensors, "
          f"{len(old)} restored", flush=True)
    pgs = []
    if old:
        pgs.append({"params": old, "lr": args.lr, "base_lr": args.lr})
    if new:
        pgs.append({"params": new, "lr": args.lr_new, "base_lr": args.lr_new})
    opt = torch.optim.AdamW(pgs, weight_decay=1e-4)
    allp = old + new
    print(f"params {sum(p.numel() for p in allp)/1e6:.1f}M, "
          f"new {sum(p.numel() for p in new)/1e3:.1f}K  REAL ERF events", flush=True)

    # Dataset selection: real ERF events (default), pure synthetic, real
    # EventAid voxels, or a probabilistic real/synthetic mix.
    gaps = tuple(args.gaps) if args.gaps else None
    length = args.steps * args.batch + args.batch  # enough samples for all steps
    if args.dataset == "syn":
        ds = SynEvDataset(args.hq_root, args.adobe_root, crop=args.crop,
                          length=length)
    elif args.dataset == "realevs":
        from real_evs_data import RealEVSDataset
        ds = RealEVSDataset(args.realevs_root, crop=args.crop, length=length,
                            **({"gaps": gaps} if gaps else {}))
    elif args.dataset == "mix":
        from real_evs_data import RealEVSDataset, MixedDataset
        real = RealEVSDataset(args.realevs_root, crop=args.crop, length=length,
                              **({"gaps": gaps} if gaps else {}))
        syn = SynEvDataset(args.hq_root, args.adobe_root, crop=args.crop,
                           length=length)
        ds = MixedDataset(real, syn, p_real=args.mix_real_prob, length=length)
    else:
        ds = ERFDataset(args.erf_root, crop=args.crop, length=length,
                        **({"gaps": gaps} if gaps else {}))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=True)

    step, acc, t0 = 0, 0.0, time.time()
    for imgs9, vox in loader:
        m = lr_mult(step, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = pg["base_lr"] * m
        imgs9 = imgs9.cuda(non_blocking=True) / 255.0
        vox = vox.cuda(non_blocking=True)
        imgs, gt = imgs9[:, :6], imgs9[:, 6:]  # ch 0-5: I0+I1, ch 6-8: GT middle
        # Model input = 6 frame channels + 16 event-voxel channels.
        flow, mask, merged, pred = net(torch.cat([imgs, vox], 1))
        # Either loss adds 0.5x deep supervision on the coarse merged preds.
        if args.loss == "char_ms":          # champion recipe: MS-Charbonnier + census
            loss = charbonnier_ms(pred, gt) + 0.1 * census_loss(pred, gt)
            for mg in merged:
                loss = loss + charbonnier_ms(mg, gt) * 0.5
        else:
            loss = lap(pred, gt).mean()
            for mg in merged:
                loss = loss + lap(mg, gt).mean() * 0.5
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(allp, 1.0)
        opt.step()
        acc += float(loss); step += 1
        if step % 50 == 0:
            print(f"step {step}/{args.steps} loss={acc/50:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/50:.2f}s/it "
                  f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
            acc, t0 = 0.0, time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save(net.state_dict(),
                       os.path.join(args.out_dir, f"emae_step{step}.pkl"))
            print(f"saved step {step}", flush=True)
        if step >= args.steps:
            break
    print("training done", flush=True)


if __name__ == "__main__":
    main()
