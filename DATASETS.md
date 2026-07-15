# Datasets — Download & Preparation Guide

Everything the release trains on or reads at inference, where to get it, and the exact
on-disk layout the scripts expect. **Path resolution**: scripts compute the project
root as the grandparent of their own file, so run release copies from a first-level
subdirectory of the project root (e.g. `<root>/scripts/`; see "Running the scripts" in
`README.md`). Two exceptions hard-code the development root and need a one-line edit on a
new machine: `code/submission/build_final.py` (`ROOT = Path("/home/ubuntu-5th/work/EventAid")`)
and `code/eval/val_blend_score.py` (same constant). All layouts below are relative to the
project root; commands follow the README convention (`scripts/*.py` = a release copy of
`code/<stage>/*.py`).

## Overview

| Dataset | Role in pipeline | Consumed by | Source |
|---|---|---|---|
| Challenge data (EventAid-F) | Inference input + validation GT for fusion/eval (never training) | `code/inference/run_*.py`, `code/submission/{make_submission,build_final}.py`, `code/eval/{val_blend_score,mini_psnr}.py`, `code/data/analyze_event_stats.py` | Codabench 16375 (login) |
| HQ-EVFI | Frames for synthetic-event EMA-E training, refiner shards, CBMNet fine-tune; real RGB-EVS events for CBMNet/realevs | `code/data/syn_ev_data.py`, `code/train/{finetune_emae,finetune_emae_erf,finetune_cbmnet}.py`, `code/data/gen_refiner_data.py` | TimeLens-XL repo (GDrive) |
| Adobe240fps | Extra frames for synthetic-event EMA-E training (events simulated) | `code/data/syn_ev_data.py` via `finetune_emae.py` / `finetune_emae_erf.py` `--adobe-root` | UBC project page (direct) |
| ERF-X170FPS | Real events for the two ERF specialists; held-out split for selection | `code/data/erf_data.py`, `code/train/finetune_emae_erf.py`, `code/eval/eval_holdout2.py`, `code/fusion/eval_wsweep.py` | CBMNet repo (GDrive) |
| Refiner shards (derived) | Fusion-refiner training data, generated from HQ-EVFI | `code/train/{train_refiner,train_refiner_v2arch}.py` | Generated locally |
| ESIM/V2V simulator | Synthetic training events | bundled: `code/data/v2v_core_esim.py` | No download needed |
| Pretrained member weights | Frozen ensemble members + inits | see `models/MANIFEST.md`, "External public pretrained weights (NOT included)" | per-author repos |

Expected local layout, consolidated:

```
<root>/challenge_data/{validation,test}/{1skip,3skip,7skip,15skip}/<seq>/...
<root>/data/data_release/ailab/group/pjlab-sail/mayongrui/dataset/realcaptured_beifen/hand/data/<session>/{visual_RGB,RGB-EVS}/
<root>/data/corpora/Adobe240_frames/<video_stem>/000001.jpg ...
<root>/data/erf_train64/train/<NNNN>/{processed_images,processed_events}/
<root>/data/erf_holdout/train/<NNNN>/{processed_images,processed_events}/
<root>/data/refiner_shards/shard_00000.npz ... shard_00099.npz
```

## 1. Official challenge data (EventAid-F, Codabench 16375)

- **Download**: https://www.codabench.org/competitions/16375/ — requires a free Codabench
  account. Register for the competition, then download `challenge_data.zip` from the
  Files/Data tab (login-gated; the page is a JS app, so use a browser). Validation GT came
  separately as `challenge_data_with_gt.zip` (adds `gt/` under `validation/` only).
  **~3.8 GB extracted.**
- **Prepare**: unzip as-is into `<root>/challenge_data/` — no renaming or restructuring.
  Test GT is not part of the release and is not needed to reproduce the submission.

```
challenge_data/
  README.md  terms.md  evaluate_results.py  run_baseline_method.py
  validation/{1skip,3skip,7skip,15skip}/room1/            # 1 sequence x 4 skips
  test/{1skip,3skip,7skip,15skip}/{ball,blocks,building,playball,room2,sculpture,traffic,umbrella,wall}/
    input/NNNNNN_img.jpg    # anchor frames; index = source frame no. (000000,000008,... for 7skip)
    gt/NNNNNN_img.jpg       # validation ONLY (7skip room1: 5 input + 28 gt)
    event/NNNNNN.txt        # one file per inter-frame interval; rows: "timestamp x y polarity"
    frame_info.txt          # [input] rows + [TODO] rows giving required output paths results/{skip}/{seq}/NNNNNN.png
    event_info.txt          # "event_path start_ts end_ts" per event txt
    shape.txt               # "WIDTH HEIGHT" (room1: 944 624; ball: 1168 704)
```

All inference runners default to `--data-dir <root>/challenge_data`; `val_blend_score.py`
and `mini_psnr.py` read `challenge_data/validation/.../gt/*.jpg`.

## 2. HQ-EVFI (TimeLens-XL, ECCV 2024) — `--hq-root` / `--data-root`

- **Download**: linked from the Dataset section of https://github.com/OpenImagingLab/TimeLens-XL
  as a single Google Drive file:
  https://drive.google.com/file/d/104ZMJ-M_frImOOCGfLk_HDb2FV1trveT/view?usp=drive_link
  (project page: https://openimaginglab.github.io/TimeLens-XL/). **~76 GB extracted.**
  Download while logged into a Google account to avoid the anonymous GDrive quota error.
- **Prepare**: unzip into `<root>/data/data_release/`, keeping the archive's internal path
  verbatim. Only `visual_RGB/*.png` matters to the loaders (`syn_ev_data.py` globs
  `<hq_root>/**/visual_RGB` recursively), so the deep nesting is harmless. No renaming.

```
data_release/ailab/group/pjlab-sail/mayongrui/dataset/realcaptured_beifen/hand/data/
  <session>/                              # 50 sessions; most date-named (2024-01-05-14-52),
                                          # some scene-named (Frog1, NewtonBall, hand)
    visual_RGB/<idx>_<timestamp_us>.png   # RGB frames 773x618; 36-2812 frames/session
    RGB-EVS/<idx>_<timestamp_us>.npz      # per-frame REAL events; keys: sync_rgb, x, y, p, t, start_t, end_t, ...
```

**Important**: the events used for EMA-E and refiner training are **not** these RGB-EVS
events — they are V2V/ESIM-simulated on the fly from the `visual_RGB` frames
(`code/data/v2v_core_esim.py`). Contrast thresholds differ per consumer: EMA-E training
samples U[0.1, 0.7] (`syn_ev_data.py`, recalibrated against the challenge input-event
statistics), while refiner-shard generation samples U[0.2, 1.2]
(`gen_refiner_data.py`). The RGB-EVS real events are read only by
`finetune_cbmnet.py` (`--data-root`, default `<root>/data/data_release`) and by
`finetune_emae_erf.py --dataset realevs/mix` (`--realevs-root`, dataset class
`code/data/real_evs_data.py`).

## 3. Adobe240fps (Su et al., CVPR 2017) — `--adobe-root`

- **Download**: https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/ — direct
  zip, no registration. Get `DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip`
  (2.8 GB, the original 240fps videos). The processed blur dataset
  (`DeepVideoDeblurring_Dataset.zip`, 3.7 GB) is **not** needed.
- **Prepare** (what produced our `data/corpora/Adobe240_frames`, **4.3 GB**):
  1. Unzip (133 high-fps videos).
  2. Evenly subsample 40 videos: over the **sorted** file list of length N=133, take
     indices `i*N/40` for i = 0..39 (deterministic given the sorted list).
  3. Decode each at original resolution/framerate:
     `ffmpeg -qscale:v 3 <video> <root>/data/corpora/Adobe240_frames/<stem>/%06d.jpg`
     (replace spaces in video stems with `_`).
  4. Delete the zip and raw videos.

```
Adobe240_frames/
  <video_stem>/              # 40 dirs: 720p_240fps_1, 720p_240fps_4, GOPR9633..GOPR9660, IMG_0003..IMG_0188
    000001.jpg 000002.jpg …  # 6-digit 1-indexed, 1280x720 @240fps; 33,075 jpgs total
```

`syn_ev_data.py` globs `<adobe_root>/*` then `<seq>/*.jpg` and requires ≥ 25 frames per
sequence (max gap 24 + 1). Events are always simulated. Consumed via `finetune_emae.py` /
`finetune_emae_erf.py` `--adobe-root` (default `<root>/data/corpora/Adobe240_frames`).

## 4. ERF-X170FPS (CBMNet, CVPR 2023) — `--erf-root`, holdout

- **Download**: linked from https://github.com/intelpro/CBMNet — train:
  https://drive.google.com/file/d/1Bsf9qreziPcVEuf0_v3kjdPUh27zsFXK/view (~219 GB
  compressed / 234 GB uncompressed / 104 sequences); test:
  https://drive.google.com/file/d/1Dk7jVQD29HqRVV11e8vxg5bDOh6KxrzL/view (~76 GB,
  **not needed** — never used). Anonymous/gdown downloads hit the GDrive quota error at
  these sizes; a logged-in browser session ("download anyway" on the virus-scan page)
  works. License: research and education only.
- **Prepare**: we extracted only sessions `0000`–`0069` from `train.zip` (70 dirs,
  148 GB) into `<root>/data/erf_train/train/<NNNN>/`, then made the train/holdout split
  purely with symlinks:

```bash
mkdir -p data/erf_train64/train data/erf_holdout/train
for s in 0002 0003 0023 0028 0036 0046; do ln -s ../../erf_train/train/$s data/erf_holdout/train/$s; done
for s in data/erf_train/train/*; do n=$(basename $s); \
  [ -e data/erf_holdout/train/$n ] || ln -s ../../erf_train/train/$n data/erf_train64/train/$n; done
```

  giving **erf_train64** (64 sessions, 137 GB dereferenced) and **erf_holdout** (6
  sessions: 0002, 0003, 0023, 0028, 0036, 0046; 12 GB). Two real directories with the
  same `<split>/train/<NNNN>/` shape work equally well — `erf_data.py` globs
  `<root>/**/processed_images` recursively and follows symlinks.

```
erf_train/train/<NNNN>/
  processed_images/NNNNN.png   # 990 frames/session, 1440x975 RGB @170fps (5-digit, 0-indexed)
  processed_events/NNNNN.npz   # events in [frame N, frame N+1); keys x uint32, y uint32, p int16 (0/1), t int64 (us)
```

**Sub-pixel coordinates**: `x`/`y` are stored at 128x sub-pixel resolution — divide by
128 to get pixels. `code/data/erf_data.py` does this (`SUBPIX = 128.0`); remember it if
you write your own reader.

**Gotcha**: `finetune_emae_erf.py`'s `--erf-root` argparse default is `data/erf_train`,
but both shipped specialists were trained with `--erf-root data/erf_train64` (README step
2, `models/MANIFEST.md`; the log confirms 64 sequences) — always pass it explicitly. The
holdout is read only by `eval_holdout2.py` (specialist checkpoint selection) and
`eval_wsweep.py` (chose the syn↔ERF weight w=0.85); both default to `--root
<root>/data/erf_holdout`.

## 5. Fusion-refiner training shards (derived — generate locally)

Fully synthetic, derived from HQ-EVFI: random (sequence, anchor pair, target-time) draws,
256x256 crops, V2V/ESIM-simulated events, plus the stored predictions of three frozen
base models (GIMM-VFI-F, GIMM-VFI-R, TimeLens — their checkpoints must be in place under
`weights/`, see `models/MANIFEST.md`). Generate with `code/data/gen_refiner_data.py`:

```bash
CUDA_VISIBLE_DEVICES=1 python3 scripts/gen_refiner_data.py --out data/refiner_shards --n-samples 10000
```

~0.8 samples/s (≈3.5 h on one 3090 Ti); parallelize with disjoint `--start-shard`
offsets. Output: `shard_00000.npz` … `shard_00099.npz` (100 samples each; per-sample
keys `a0_<i>`/`a1_<i>` anchors, `gt_<i>`, `lin_<i>`, `gimm_<i>`/`gimmr_<i>`, `tl_<i>`,
`vox_<i>` float16 (16,256,256), `alpha_<i>`, `skip_<i>`). Consumed by `train_refiner.py`
and `train_refiner_v2arch.py` `--shards` (default `<root>/data/refiner_shards`). This
step is only needed to retrain refiners — the release ships the trained checkpoints.

## 6. No-download items

- **Event simulator**: self-contained at `code/data/v2v_core_esim.py` (pure-numpy
  ESIM-style DVS emulator, V2V core). Upstream references only: V2V
  (https://github.com/HYLZ-2019/V2V, by the challenge organizers) and ESIM
  (https://github.com/uzh-rpg/rpg_esim, MIT).
- **Public pretrained weights**: the 13 external checkpoints and their expected
  `weights/` paths are enumerated in `models/MANIFEST.md`, section "External public
  pretrained weights (NOT included)" — e.g. EMA-VFI `ours.pkl` → `weights/emavfi/ours.pkl`
  (https://github.com/MCG-NJU/EMA-VFI), CBMNet-Large BSERGB/ERF checkpoints
  (https://github.com/intelpro/CBMNet).

## What trains on what

| Model (checkpoint) | Training data | Events | Script |
|---|---|---|---|
| EMA-E synthetic base + ft2 continuation (`emae_base_synthetic_step7000.pkl`, `emae_ft2_*.pkl`) | HQ-EVFI `visual_RGB` (50 seqs) + Adobe240 frames (40 seqs) | V2V/ESIM simulated | `code/train/finetune_emae.py` |
| ERF specialists 7skip / 15skip (`erf_specialist_*.pkl`) | `data/erf_train64` (64 seqs; init = synthetic base step-7000) | ERF real events | `code/train/finetune_emae_erf.py`; selection on `data/erf_holdout` via `code/eval/eval_holdout2.py`, blend weight via `code/fusion/eval_wsweep.py` |
| Fusion refiners V1 / b80 / V2 (`fusion_refiner_*.pth`) | `data/refiner_shards` (HQ-EVFI-derived, frozen GIMM/TimeLens preds) | simulated (baked into shards) | `code/train/train_refiner.py`, `code/train/train_refiner_v2arch.py` |
| CBMNet fine-tune → soup (`cbmnet_soup_uf35.pth`) | HQ-EVFI frames + RGB-EVS real events | HQ-EVFI real | `code/train/finetune_cbmnet.py`, `code/train/make_cbmnet_soup.py` |
| Optional realevs member (`emae_realevs_step2000.pkl`, disabled) | HQ-EVFI `visual_RGB` frames + RGB-EVS **real** events (30 sessions of `data_release`) | HQ-EVFI real (RGB-EVS) | `code/train/finetune_emae_erf.py --dataset realevs/mix` (dataset class `code/data/real_evs_data.py`) |
| Fusion recipes v10–v16 | challenge **validation** split outputs + GT | — | `code/fusion/blend_v11.py` |

## Compliance

- The **public EventAid dataset is banned** by the challenge rules for any
  training/validation/tuning — it was never downloaded and never used.
- **GoPro and REDS are permitted** by the rules but were **not used** by this solution.
- `challenge_data` is used for **inference and evaluation only**; the sole input-side
  exception is `code/data/analyze_event_stats.py`, which reads challenge **input event
  streams** (never GT frames) to calibrate the simulator. The optional — disabled —
  `realevs` member trained on HQ-EVFI's RGB-EVS real events, not on challenge data.
- All fusion weights were fit exclusively on the official validation split or independent
  held-out data (the ERF holdout for w=0.85). See "Rules compliance" in `README.md`.
