# Model Checkpoints — Team yunyu8, EventAid-F Challenge (Codabench 16375, EBMV @ ECCV 2026)

All checkpoints in this directory were trained by us. Sizes are exact byte counts, verified against the training originals.

**Note — 13 files here vs 30 ensemble members:** a "member" is a prediction run, not a
checkpoint file. One checkpoint can serve several members through different inference
settings (e.g. `emae_ft2_continuation_step2000.pkl` alone feeds four anchored members and
the per-scene TTO member), and more than half of the 30 members run on public pretrained
weights, which are listed in the external section at the bottom and are not redistributed
here. The "Feeds ensemble members" column maps every file to its members; one member's
checkpoint was lost (see "Lost checkpoints") and the TTO member has no stored checkpoint
by construction.

## Our trained checkpoints

| File | Size (bytes) | Role | Trained by | Feeds ensemble members |
|---|---|---|---|---|
| `emae_base_synthetic_step7000.pkl` | 263,501,375 | EMA-E base model (zero-init event graft on EMA-VFI) trained on synthetic V2V events; strongest event-using ensemble member and the `--init` for all ERF/realevs specialist trainings. | `scripts/finetune_emae.py` (defaults: `--init weights/emavfi/ours.pkl --out-dir checkpoints/emae_ft --steps 8000 --batch 8 --lr 2e-5 --lr-new 2e-4 --crop 256 --save-every 1000`; data = HQ-EVFI `data/data_release` + Adobe240 frames with V2V-style simulated events) | `results_emae_s7000_tta` (all 4 skips, `run_emae.py --tta`); also init checkpoint for erf_v1, erf_v2 and realevs trainings |
| `emae_ft2_continuation_step2000.pkl` | 263,501,375 | EMA-E low-LR continuation checkpoint; the workhorse for all anchored-bisection EMA-E members and the starting point of the per-scene TTO member. | `scripts/finetune_emae.py --init checkpoints/emae_ft/emae_step8000.pkl --out-dir checkpoints/emae_ft2 --steps 4000 --batch 8 --lr 1e-5 --lr-new 5e-5` | `results_emae_anch_tta` (anchor=submission_v11), `results_emae_anch2_tta` (v12), `results_emae_anch3_tta` (v13), `results_emae_anch4_tta` (v14) via `run_emae.py --tta --anchor-dir`; `results_tto7_prod` via `tto_emae.py --skips 7skip --steps 40 --lr 5e-6 --tta --anchor-dir submission_v15/results` |
| `emae_ft2_continuation_step4000.pkl` | 263,501,375 | EMA-E continuation step-4000 checkpoint; un-anchored 15skip diversity member. | Same run as the step-2000 checkpoint (`scripts/finetune_emae.py --init checkpoints/emae_ft/emae_step8000.pkl --steps 4000 --lr 1e-5 --lr-new 5e-5`), later save | `results_emaft2s4000_tta` (15skip only, `run_emae.py --tta`, no anchor) |
| `erf_specialist_7skip_gaps8-16_step12000.pkl` | 263,501,660 | ERF-X170FPS real-event EMA-E specialist for 7skip fast/medium-motion scenes (trained on gaps 8/12/16). | `scripts/finetune_emae_erf.py --init checkpoints/emae_ft/emae_step7000.pkl --erf-root data/erf_train64 --out-dir checkpoints/erf_v1 --steps 12000 --batch 8 --seed 1234 --lr 1.5e-5 --lr-new 8e-5 --gaps 8 12 16` (log confirms 64-sequence ERF train set, held-out excluded) | `results_test_v1_8way` (`EMAE_DIHEDRAL=1` 8-way self-ensemble, `--tta --anchor-dir submission_v16/results`, test 7skip) = the dir `build_final.py` reads; earlier `results_test_v1bt` / `results_test_v1_medium` superseded by it |
| `erf_specialist_15skip_gaps12-20_step12000.pkl` | 263,501,660 | ERF-X170FPS real-event EMA-E specialist for 15skip large-motion scenes (trained on gaps 12/16/20). | `scripts/finetune_emae_erf.py --init checkpoints/emae_ft/emae_step7000.pkl --erf-root data/erf_train64 --out-dir checkpoints/erf_v2 --steps 12000 --batch 8 --seed 777 --lr 1.5e-5 --lr-new 8e-5 --gaps 12 16 20` | `results_test_v2_8way` (`EMAE_DIHEDRAL=1` 8-way, `--tta --anchor-dir submission_v16/results`, test 15skip) read by `build_final.py`; earlier `results_test_v2bt` / `results_test_v2_medium` superseded |
| `fusion_refiner_v1_s5000.pth` | 4,309,033 | RefinerV1: our event-conditioned U-Net fusion refiner (per-pixel gating of GIMM/TimeLens/linear + residual), step-5000 checkpoint (transfer peaks early). | `scripts/train_refiner.py --shards data/refiner_shards --out-dir checkpoints/refiner --steps 30000 --batch 16` (step-5000 save used); shards built by `scripts/gen_refiner_data.py` from HQ-EVFI frames + calibrated V2V-sim events | `results_refined_s5k` (`scripts/run_refiner.py --ckpt checkpoints/refiner/refiner_step5000.pth` = this file's old path) |
| `fusion_refiner_b80_s5000.pth` | 11,875,625 | RefinerV1 capacity variant (base=80 channels), step-5000; second refiner member fed with full-res-flow GIMM inputs. | `scripts/train_refiner.py --shards data/refiner_shards --out-dir checkpoints/refiner_b80 --steps 12500 --batch 16 --save-every 2500 --base 80` (step-5000 save) | `results_refined_b80_g10` (`scripts/run_refiner.py --ckpt checkpoints/refiner_b80/refiner_step5000.pth --gimm-dir results_gimmf_ds10_tta`) |
| `fusion_refiner_v2arch_s2500.pth` | 19,471,418 | RefinerV2 (4.86M params at the trained feat48/base96 width: event encoder with per-bin temporal attention + zero-init deformable alignment, feat48/base96), step-2500; best refiner member (7skip 41.92 / 15skip 38.77). | `scripts/train_refiner_v2arch.py --shards data/refiner_shards --out-dir checkpoints/refiner_v2arch --steps 15000 --batch 12 --feat 48 --base 96` (step-2500 save); arch in `scripts/refiner_v2arch.py` | `results_refined_v2a` (`scripts/run_refiner_v2arch.py --ckpt checkpoints/refiner_v2arch/v2arch_step2500.pth --splits validation test --skips 7skip 15skip`) |
| `cbmnet_soup_uf35.pth` | 89,095,259 | CBMNet-Large model soup: 0.65 * BSERGB-pretrained + 0.35 * our unfrozen-flownet synthetic fine-tune (step 8000); best CBMNet ensemble member. | `scripts/finetune_cbmnet_syn.py --data-root data/data_release --init weights/ours_large_bsergb.pth --out-dir checkpoints/cbmnet_ftsyn_uf --steps 8000 --batch 2 --lr 1e-5 --unfreeze-flownet`; soup created by an inline torch snippet (soup = 0.65*orig + 0.35*ftsyn_step8000, saved as `cbmnet_ftsyn_uf/soupuf35.pth`, later consolidated) | `results_cbmnet_soupuf35_tta` (`run_cbmnet.py --ckpt ...soupuf35.pth --model-name ours_large`, plus `--flip h/v/hv` runs averaged into the `_tta` dir by an inline script) |
| `emavfi_soup_ft_s3000_a80.pkl` | 262,737,035 | EMA-VFI (image-only) model soup: 0.20 * pretrained `ours.pkl` + 0.80 * our synthetic fine-tune step-3000; 7skip diversity member. OUR file despite originally living in `weights/`. | `scripts/finetune_emavfi.py` -> `checkpoints/emavfi_ft/ft_step3000.pkl` (still on disk); soup built and selected by `scripts/soup_eval_emavfi.py --jobs 3000:0.8` | `results_emaft7_tta` (`scripts/run_emavfi.py --variant ours_ft_s3000_a80 --tta`, 7skip validation+test) |
| `emae_realevs_step2000.pkl` | 263,501,375 | OPTIONAL: EMA-E adapted on **HQ-EVFI RGB-EVS real events** (30 `data_release` sessions; dataset class `code/data/real_evs_data.py`); referenced by `build_final.py` as `ROOM1_REALEVS = results_val_realevs2k` but DISABLED (`ROOM1_W = {}`, gain below uint8 quantization floor) — not loaded to reproduce the final zip; included only for completeness. | `scripts/finetune_emae_erf.py --arch base --dataset realevs --init checkpoints/emae_ft/emae_step7000.pkl --out-dir checkpoints/realevs --steps 8000 --batch 8 --seed 1234 --lr 1.5e-5 --lr-new 6e-5 --evenc-fast --gaps 4 8 12 16` | `results_val_realevs2k` (`run_emae.py --seqs room1 --tta --anchor-dir submission_v16/results`) |
| `emae_base_synthetic_step8000_ft2init.pkl` | 263,501,375 | OPTIONAL (retraining reproducibility): the step-8000 save of the synthetic base run; not loaded at inference, but it is the `--init` of the `emae_ft2` continuation training above. | Same run as `emae_base_synthetic_step7000.pkl` (`scripts/finetune_emae.py`, later save = `checkpoints/emae_ft/emae_step8000.pkl`) | none directly; `--init` of `emae_ft2_continuation_*` |
| `emavfi_ft_raw_step3000.pkl` | 263,406,331 | OPTIONAL (retraining reproducibility): raw EMA-VFI synthetic fine-tune behind the `emavfi_soup_ft_s3000_a80.pkl` soup; lets you re-create the soup with `code/train/soup_eval_emavfi.py`. | `scripts/finetune_emavfi.py` -> `checkpoints/emavfi_ft/ft_step3000.pkl` | none directly; soup ingredient (alpha=0.8) |

## Lost checkpoints (disclosed)

Two intermediate checkpoints were deleted in disk cleanups before this release was
assembled. Their training commands are fully specified, and the artifacts they produced
(member PNG directories / consolidated soup) survive, so the final submission remains
bit-exactly reproducible without them:

| Lost file | What it was | Training command (documented) | Surviving artifact |
|---|---|---|---|
| `checkpoints/emae_v2/emae_step5000.pkl` | EMA-E seed/gap variant (diversity member) | `scripts/finetune_emae.py --out-dir checkpoints/emae_v2 --seed 777 --hq-gaps 4 6 10 14 --adobe-gaps 8 12 18 28` | member dir `results_emaev2_anch_tta` (used by the fusion recipes) |
| `checkpoints/cbmnet_ftsyn_uf/ftsyn_step8000.pth` | raw unfrozen CBMNet synthetic fine-tune (soup ingredient) | `scripts/finetune_cbmnet_syn.py --init weights/ours_large_bsergb.pth --steps 8000 --unfreeze-flownet` | consolidated soup `cbmnet_soup_uf35.pth` (shipped here) |

Two release helpers re-create steps that were historically inline one-liners:
`code/inference/average_tta.py` (averaging per-flip runs into a `_tta` member dir) and
`code/train/make_cbmnet_soup.py` (weight-space souping).

## External public pretrained weights (NOT included)

The full pipeline additionally requires the following publicly available pretrained weights. They are standard public downloads from the respective authors' repositories and are therefore not redistributed here. Paths are where the pipeline scripts expect them.

| Expected path | What it is / what it feeds |
|---|---|
| `weights/emavfi/ours.pkl` | EMA-VFI pretrained (member `results_emavfi_tta`; init for EMA-E training; soup base for `ours_ft_s3000_a80`) |
| `weights/rife_train_log/flownet.pkl` | RIFE HDv3 pretrained (`results_rife_tta`, `results_rife_anch_tta`; loaded via `train_log/RIFE_HDv3` in `run_rife.py`) |
| `weights/timelens_checkpoint.bin` | TimeLens (CVPR'21) pretrained (`results_timelens_tta`, `results_timelens_hier_tta`, `results_tlhier_anch_tta`) |
| `weights/gimmvfi/gimmvfi_f_arb.pt` | GIMM-VFI-F non-LPIPS arb-time checkpoint (`results_gimmvfi_tta`, `results_gimmf_ds10_tta`) |
| `weights/gimmvfi/raft-things.pth` | RAFT flow weights required by GIMM-VFI-F |
| `weights/gimmvfi/gimmvfi_r_arb.pt` | GIMM-VFI-R variant (`results_gimmvfi_r_tta`, `results_gimmr_ds10_tta`) |
| `weights/gimmvfi/flowformer_sintel.pth` | FlowFormer flow weights required by GIMM-VFI-R |
| `weights/vfimamba/VFIMamba.pkl` | VFIMamba pretrained (`results_vfimamba_tta`, `results_vfimamba_sc10_tta`, `results_vfimamba_anch_tta`, `results_vfimamba_anch2_tta`) |
| `weights/Expv8_large_HQEVFI.pt` | TimeLens-XL HQ-EVFI checkpoint (`results_tlx_hqevfi`; `run_tlx.py` default `--weights`) |
| `weights/refid/REFID-HighREV-7skip.pth` | REFID HighREV 7skip checkpoint (`results_refid_tta`) |
| `weights/refid/REFID-HighREV-15skip.pth` | REFID HighREV 15skip checkpoint (`results_refid_tta`) |
| `weights/ours_large_bsergb.pth` | CBMNet-Large BSERGB pretrained (`results_cbmnet_bsergb_tta`; base of `cbmnet_soup_uf35` soup; `--init` of `finetune_cbmnet_syn.py`) |
| `weights/ours_large_erf.pth` | CBMNet-Large ERF-X170FPS pretrained (`results_cbmL_erf_test`, the `CBM_DIR` diversity member in `build_final.py` 3-way fusion) |
