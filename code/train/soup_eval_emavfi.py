#!/usr/bin/env python3
"""Model-soup + validation eval driver for the EMA-VFI fine-tune.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375):
  TRAINING-STAGE script. Produced and selected the image-only EMA-VFI model
  soup shipped as models/emavfi_soup_ft_s3000_a80.pkl (0.2*pretrained +
  0.8*ft_step3000), which feeds the 7skip diversity member results_emaft7_tta.

For each job "step:alpha":
  1. soup = (1-alpha) * ours.pkl + alpha * checkpoints/emavfi_ft/ft_step{step}.pkl
     written as weights/emavfi/ours_ft_s{step}_a{int(100a)}.pkl ('module.' keys,
     attn_mask/HW buffers dropped -- the runner filters them anyway).
  2. run scripts/run_emavfi.py --variant <tag> on validation 7skip+15skip
     (recursive midpoint bisection, non-TTA), outputs to results_emavfi_ft/<tag>.
  3. PSNR vs challenge_data/validation/{skip}/{seq}/gt/{idx:06d}_img.jpg
     (float64, 20*log10(255/sqrt(mse)), mean over frames).
Appends rows to logs/emavfi_ft_eval.csv. alpha=0 evaluates the original model
(eval-path sanity check); --wait blocks until a checkpoint file appears.

Usage:
  python3 scripts/soup_eval_emavfi.py --gpu 3 --wait \
      --jobs 0:0.0,1000:0.2,1000:0.35,1000:0.5,...
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "weights" / "emavfi" / "ours.pkl"
CKPT_DIR = ROOT / "checkpoints" / "emavfi_ft"
RES_ROOT = ROOT / "results_emavfi_ft"
VAL = ROOT / "challenge_data" / "validation"
CSV = ROOT / "logs" / "emavfi_ft_eval.csv"
SKIPS = ("7skip", "15skip")


def build_soup(step, alpha):
    if alpha == 0.0:
        return "ours"
    tag = f"ours_ft_s{step}_a{int(round(alpha * 100)):02d}"
    out = ROOT / "weights" / "emavfi" / f"{tag}.pkl"
    if out.exists():
        return tag
    orig = torch.load(ORIG, map_location="cpu")
    ft = torch.load(CKPT_DIR / f"ft_step{step}.pkl", map_location="cpu")
    soup = {}
    for k, v in orig.items():
        if "attn_mask" in k or "HW" in k:
            continue
        kf = k.replace("module.", "")
        assert kf in ft, kf
        assert ft[kf].shape == v.shape, kf
        soup[k] = ((1.0 - alpha) * v.float() + alpha * ft[kf].float()).to(v.dtype)
    torch.save(soup, out)
    return tag


def psnr_eval(res_dir):
    out = {}
    for skip in SKIPS:
        vals = []
        for png in sorted((res_dir / skip).glob("*/*.png")):
            gt_path = VAL / skip / png.parent.name / "gt" / f"{png.stem}_img.jpg"
            if not gt_path.exists():
                continue
            pred = np.asarray(Image.open(png).convert("RGB"), dtype=np.float64)
            gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.float64)
            mse = float(np.mean((pred - gt) ** 2))
            vals.append(99.0 if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse)))
        out[skip] = (float(np.mean(vals)) if vals else float("nan"), len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--jobs", required=True,
                    help="comma list of step:alpha, e.g. 1000:0.2,2000:0.35")
    ap.add_argument("--wait", action="store_true",
                    help="wait for checkpoint files to appear (during training)")
    ap.add_argument("--wait-timeout", type=float, default=7200)
    args = ap.parse_args()

    jobs = []
    for tok in args.jobs.split(","):
        s, a = tok.split(":")
        jobs.append((int(s), float(a)))

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    if not CSV.exists():
        CSV.write_text("step,alpha,psnr_7skip,n7,psnr_15skip,n15,eval_s\n")

    for step, alpha in jobs:
        ck = CKPT_DIR / f"ft_step{step}.pkl"
        if alpha > 0:
            t_wait = time.time()
            while not ck.exists():
                if not args.wait or time.time() - t_wait > args.wait_timeout:
                    raise FileNotFoundError(ck)
                time.sleep(20)
            while time.time() - ck.stat().st_mtime < 10:  # finish writing
                time.sleep(5)
        t0 = time.time()
        tag = build_soup(step, alpha)
        res_dir = RES_ROOT / (tag if alpha > 0 else "baseline_ours")
        cmd = [sys.executable, str(ROOT / "scripts" / "run_emavfi.py"),
               "--variant", tag, "--splits", "validation",
               "--skips", ",".join(SKIPS), "--output-dir", str(res_dir)]
        r = subprocess.run(cmd, env=env, cwd=str(ROOT),
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"RUN FAILED step={step} a={alpha}\n{r.stdout[-2000:]}\n"
                  f"{r.stderr[-2000:]}", flush=True)
            continue
        res = psnr_eval(res_dir)
        dt = time.time() - t0
        (p7, n7), (p15, n15) = res["7skip"], res["15skip"]
        row = f"{step},{alpha},{p7:.4f},{n7},{p15:.4f},{n15},{dt:.0f}"
        with open(CSV, "a") as f:
            f.write(row + "\n")
        print(f"[gpu{args.gpu}] step={step} a={alpha}: "
              f"7skip {p7:.4f} ({n7}f)  15skip {p15:.4f} ({n15}f)  [{dt:.0f}s]",
              flush=True)
    print(f"[gpu{args.gpu}] all jobs done", flush=True)


if __name__ == "__main__":
    main()
