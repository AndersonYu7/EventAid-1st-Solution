#!/usr/bin/env python3
"""Analyze REAL EventAid event statistics (legal: input events only, no GT).

Pipeline role (1st-place EventAid-F solution, team yunyu8):
  DATA PREP / calibration (this file) -> training -> member inference ->
  fusion -> submission. This diagnostic produced the numbers used to
  calibrate the synthetic event simulator: the U[0.1, 0.7] contrast-threshold
  range in syn_ev_data.py and the "voxel |max| ~5.8" match that justified
  training real-event specialists on ERF-X170FPS (erf_data.py). It reads only
  the *input* events of the challenge data, never ground-truth frames, so it
  is legal under the challenge rules for both validation and test splits.

Computes, over all challenge sequences (val+test), the sensor-level event
statistics a simulator must match for synthetic-trained models to transfer:
  - events per pixel per anchor-gap (rate)
  - polarity ratio pos/(pos+neg)
  - spatial activity: fraction of pixels with >=1 event in a gap
  - per-pixel event count distribution (mean, p50, p95, max) over active px
  - 16-bin voxel value stats (the actual network input): nonzero frac, |max|,
    per-bin energy profile (temporal shape)

Inputs:  challenge_data/<split>/<skip>/<sequence>/ dirs readable by
         evlib.dataset.load_sequence.
Outputs: statistics printed to stdout (no files written).

Usage:
  python3 scripts/analyze_event_stats.py [--splits validation test] [--n 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evlib.dataset import load_sequence, events_to_voxel  # noqa: E402


def real_stats(splits, skips, max_gaps):
    """Accumulate per-anchor-gap statistics over up to `max_gaps` gaps."""
    rates, pol, active, pmean, pp95, pmax = [], [], [], [], [], []
    vox_nz, vox_absmax, bin_energy = [], [], []
    n = 0
    for split in splits:
        for skip in skips:
            sdir = ROOT / "challenge_data" / split / skip
            if not sdir.exists():
                continue
            for seqd in sorted(p for p in sdir.iterdir() if p.is_dir()):
                seq = load_sequence(str(ROOT / "challenge_data"), split, skip,
                                    seqd.name)
                ts = {f.index: f.timestamp for f in seq.frames}
                inp = seq.inputs
                H, W = seq.height, seq.width
                # One measurement per gap between consecutive INPUT frames
                # (exactly the event window a model sees at inference).
                for a, b in zip(inp[:-1], inp[1:]):
                    ev = seq.load_events(ts[a.index], ts[b.index])
                    if ev.shape[0] == 0:
                        continue
                    # Per-pixel event-count map for this gap.
                    cnt = np.zeros((H, W), np.int32)
                    xs = ev[:, 1].astype(int).clip(0, W - 1)
                    ysr = ev[:, 2].astype(int).clip(0, H - 1)
                    np.add.at(cnt, (ysr, xs), 1)
                    rates.append(ev.shape[0] / (H * W))
                    pol.append(float((ev[:, 3] > 0).mean()))
                    act = cnt > 0
                    active.append(float(act.mean()))
                    if act.sum():
                        # Count distribution over ACTIVE pixels only.
                        cv = cnt[act]
                        pmean.append(float(cv.mean()))
                        pp95.append(float(np.percentile(cv, 95)))
                        pmax.append(float(cv.max()))
                    # Stats on the 16-bin voxel -- the tensor networks
                    # actually consume, hence the calibration target.
                    vox = events_to_voxel(ev, 16, H, W, ts[a.index], ts[b.index])
                    vox_nz.append(float((np.abs(vox) > 1e-6).mean()))
                    vox_absmax.append(float(np.abs(vox).max()))
                    bin_energy.append(np.abs(vox).mean(axis=(1, 2)))
                    n += 1
                    if n >= max_gaps:
                        break
                if n >= max_gaps:
                    break
    be = np.mean(bin_energy, axis=0) if bin_energy else np.zeros(16)
    return {
        "n_gaps": n,
        "events_per_px": (np.mean(rates), np.std(rates)),
        "polarity_pos_frac": (np.mean(pol), np.std(pol)),
        "active_px_frac": (np.mean(active), np.std(active)),
        "active_cnt_mean": np.mean(pmean) if pmean else 0,
        "active_cnt_p95": np.mean(pp95) if pp95 else 0,
        "active_cnt_max": np.mean(pmax) if pmax else 0,
        "vox_nonzero_frac": np.mean(vox_nz),
        "vox_absmax": np.mean(vox_absmax),
        "bin_energy": be,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    ap.add_argument("--skips", nargs="+", default=["7skip", "15skip"])
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    print("=== REAL EventAid event statistics (input only) ===", flush=True)
    rs = real_stats(args.splits, args.skips, args.n)
    for k, v in rs.items():
        if k == "bin_energy":
            print(f"  {k}: {np.round(v, 4).tolist()}")
        elif isinstance(v, tuple):
            print(f"  {k}: {v[0]:.4f} +- {v[1]:.4f}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
