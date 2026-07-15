#!/usr/bin/env python3
"""EMA-E: event-aware EMA-VFI ('ours', F=32) — architecture definition.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
Pipeline stages: data prep -> training -> member inference -> fusion ->
submission. This module belongs to the TRAINING and MEMBER-INFERENCE
stages: it defines the network for the EMA-E ensemble members (the
event-grafted EMA-VFI variants of the 28-member ensemble). Training
scripts import `build_emae()` to fine-tune from the pretrained image-only
EMA-VFI weights; per-member inference scripts import it again to load
the fine-tuned checkpoints and produce the frame predictions that are
later combined by the quadratic-form fusion stage (recipe_v15/v16.json).

Architecture
------------
Extends the pretrained image-only EMA-VFI with a 16-bin event-voxel branch:
a small encoder Conv(16->32) + PReLU + Conv(32->2F) whose output is split
into two F-channel maps and ADDED to the stage-0 (full-resolution) conv
features of frame0 / frame1 inside the MotionFormer backbone. The encoder's
last conv is ZERO-INITIALIZED, so at step 0 the network is bit-identical to
the pretrained image-only model; fine-tuning then learns to exploit events.

Inputs / outputs
----------------
Input packing: x = cat([img0(3), img1(3), voxel(16)], dim=1) -> (B, 22, H, W),
where img0/img1 are the two boundary RGB frames in [0, 1] and voxel is the
16-bin event voxel grid covering the img0->img1 interval. Forward returns
(flow_list, mask_list, merged, pred) with `pred` the interpolated frame at
`timestep`, clamped to [0, 1].
Everything downstream of stage 0 (cross-scale embed, motion former, flow
heads, refine Unet) is reused unchanged, so `ours.pkl` loads with
strict=False and the only missing keys are the event encoder's.

Usage example
-------------
    # from a script living next to this file (models/EMA-VFI must exist
    # two directories up, see the sys.path setup below):
    python3 -c "
    import torch
    from emae_arch import build_emae
    net = build_emae(init='ckpt/emae_ft.pth', device='cuda').eval()
    x = torch.rand(1, 22, 256, 320, device='cuda')  # img0|img1|voxel16
    with torch.no_grad():
        _, _, _, pred = net(x, timestep=0.5)
    print(pred.shape)  # (1, 3, 256, 320)
    "
Set EMAE_EVBINS=32 in the environment before import to build the
finer-temporal-resolution B2 variant (32 voxel bins).
"""
import os
import sys

import torch
import torch.nn as nn

# Make the vendored EMA-VFI repo importable: this file is expected to live
# one directory below the project root, with the upstream code checked out
# at <root>/models/EMA-VFI (provides `model.*` and `config`).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMAVFI = os.path.join(ROOT, "models", "EMA-VFI")
if EMAVFI not in sys.path:
    sys.path.insert(0, EMAVFI)

from model.feature_extractor import MotionFormer  # noqa: E402
from model.flow_estimation import MultiScaleFlow  # noqa: E402
from model.warplayer import warp  # noqa: E402
import config as cfg  # noqa: E402

# Number of temporal bins in the event voxel grid. Default 16; setting the
# env var EMAE_EVBINS=32 builds the B2 ensemble variant with a finer
# temporal voxelization (must match how the voxels were precomputed).
EV_BINS = int(os.environ.get("EMAE_EVBINS", 16))  # 32 -> finer temporal voxel (B2)


class EventMotionFormer(MotionFormer):
    """MotionFormer + zero-init event encoder injected at stage-0 features."""

    def __init__(self, ev_bins=EV_BINS, **kargs):
        super().__init__(**kargs)
        f0 = kargs['embed_dims'][0]
        self.ev_encoder = nn.Sequential(
            nn.Conv2d(ev_bins, 32, 3, 1, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 2 * f0, 3, 1, 1),
        )
        # zero-init the LAST layer only: step-0 output == pretrained model
        nn.init.zeros_(self.ev_encoder[2].weight)
        nn.init.zeros_(self.ev_encoder[2].bias)

    def forward(self, x1, x2, vox):
        # Upstream EMA-VFI trick: stack frame0 and frame1 along the batch
        # dim (-> 2B) so both share one backbone pass; keep B to split later.
        B = x1.shape[0]
        x = torch.cat([x1, x2], 0)
        # Encode the event voxel once, then split the 2F channels into the
        # frame0-half and frame1-half and stack them batch-wise to mirror
        # the (2B, F, H, W) layout of the image features above.
        ev = self.ev_encoder(vox)              # (B, 2*F, H, W)
        f0 = ev.shape[1] // 2
        ev = torch.cat([ev[:, :f0], ev[:, f0:]], 0)  # (2B, F, H, W)
        motion_features = []
        appearence_features = []
        xs = []
        # Stage loop copied from upstream MotionFormer.forward; the ONLY
        # functional change is the `x = x + ev` injection after stage 0.
        for i in range(self.num_stages):
            motion_features.append([])
            patch_embed = getattr(self, f"patch_embed{i + 1}", None)
            block = getattr(self, f"block{i + 1}", None)
            norm = getattr(self, f"norm{i + 1}", None)
            if i < self.conv_stages:
                # Early convolutional stages (appearance only).
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
                if i == 0:
                    x = x + ev                 # event injection (zero at init)
                xs.append(x)
            else:
                # Transformer stages: tokenize, add positional correlation,
                # and collect per-block inter-frame motion features.
                if i == self.conv_stages:
                    # First transformer stage embeds the multi-scale conv
                    # feature pyramid gathered in `xs` (cross-scale embed).
                    x, H, W = patch_embed(xs)
                else:
                    x, H, W = patch_embed(x)
                cor = self.get_cor((x.shape[0], H, W), x.device)
                for blk in block:
                    x, x_motion = blk(x, cor, H, W, B)
                    # (2B*H*W, C) tokens -> (2B, C, H, W) feature maps.
                    motion_features[i].append(
                        x_motion.reshape(2 * B, H, W, -1).permute(0, 3, 1, 2).contiguous())
                x = norm(x)
                x = x.reshape(2 * B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # Concatenate motion features of all blocks in this stage.
                motion_features[i] = torch.cat(motion_features[i], 1)
            appearence_features.append(x)
        return appearence_features, motion_features


class EventMultiScaleFlow(MultiScaleFlow):
    """MultiScaleFlow whose forward takes x = cat(img0, img1, voxel)."""

    def forward(self, x, timestep=0.5):
        # Unpack the (B, 22, H, W) input: RGB frame0, RGB frame1, event voxel.
        img0, img1, vox = x[:, :3], x[:, 3:6], x[:, 6:]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        # Backbone (EventMotionFormer): appearance + motion feature pyramids.
        af, mf = self.feature_bone(img0, img1, vox)
        # Coarse-to-fine flow estimation, copied from upstream MultiScaleFlow;
        # each stage refines the bidirectional flow and the occlusion mask.
        for i in range(self.flow_num_stage):
            # Broadcast the target timestep over the motion features so the
            # flow heads know which intermediate time to predict.
            t = torch.full(mf[-1 - i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow is not None:
                # Later stages: predict residual flow/mask on top of the
                # previous estimate, conditioned on the current warps.
                flow_d, mask_d = self.block[i](
                    torch.cat([t * mf[-1 - i][:B], (1 - timestep) * mf[-1 - i][B:],
                               af[-1 - i][:B], af[-1 - i][B:]], 1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                # First stage: estimate flow/mask from scratch.
                flow, mask = self.block[i](
                    torch.cat([t * mf[-1 - i][:B], (1 - t) * mf[-1 - i][B:],
                               af[-1 - i][:B], af[-1 - i][B:]], 1),
                    torch.cat((img0, img1), 1), None)
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            # Backward-warp both inputs to time t and blend with the mask.
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))

        # Refinement U-Net predicts a residual in [-1, 1] on top of the last
        # warped blend; final prediction is clamped back to valid RGB range.
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        pred = torch.clamp(merged[-1] + res, 0, 1)
        return flow_list, mask_list, merged, pred


def build_emae(init=None, device="cuda"):
    """Build EMA-E (full 'ours' arch). `init` may be the pretrained image-only
    ours.pkl ('module.'-prefixed; ev_encoder missing -> zero-init kept) or an
    EMA-E fine-tune checkpoint (raw state_dict, strict load)."""
    # F=32 / depth=[2,2,2,4,4] reproduces the upstream 'ours' configuration.
    bbcfg, mscfg = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
    net = EventMultiScaleFlow(EventMotionFormer(**bbcfg), **mscfg)
    if init is not None:
        sd = torch.load(init, map_location="cpu")
        if any(k.startswith("module.") for k in sd):
            # Case 1: upstream DataParallel checkpoint (ours.pkl). Strip the
            # 'module.' prefix and drop cached attention buffers; only the
            # event encoder is allowed to stay missing (it keeps zero-init).
            sd = {k.replace("module.", ""): v for k, v in sd.items()
                  if "module." in k and "attn_mask" not in k and "HW" not in k}
            missing, unexpected = net.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"
            bad = [k for k in missing if "ev_encoder" not in k]
            assert not bad, f"missing non-event keys: {bad[:5]}"
        else:
            # Case 2: our own EMA-E fine-tune checkpoint (plain state_dict);
            # every parameter, including the event encoder, must be present.
            sd = {k: v for k, v in sd.items()
                  if "attn_mask" not in k and "HW" not in k}
            missing, unexpected = net.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"
            assert not missing, f"missing keys: {missing[:5]}"
    return net.to(device)
