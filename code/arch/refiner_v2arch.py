"""RefinerV2: upgraded fusion refiner — deformable base alignment + event
temporal attention.

Role in the 1st-place EventAid-F pipeline (team yunyu8, Codabench 16375)
------------------------------------------------------------------------
Pipeline stages: data prep -> training -> member inference -> fusion ->
submission. This module belongs to the FUSION stage: it defines the
~1.74M-parameter learned fusion refiners of the 28-member ensemble. After
the individual members (EMA-E, RIFE, GIMM-VFI, VFIMamba, TimeLens, REFID,
CBMNet, ...) have produced their per-frame predictions, a RefinerV2 takes
a small set of those base predictions plus the event voxel and anchor
frames and outputs a single refined frame: a per-pixel softmax gate over
the bases plus a small learned residual. The refined outputs feed the
final per-frame quadratic-form blend (recipe_v15/v16.json) that
scripts/build_final.py assembles into the uint8 PNG submission.

Paper-grade components on top of the v1 gating U-Net:
  - EventEncoder: per-bin conv + learned temporal-attention pooling over the
    16 voxel bins (instead of treating bins as flat channels)
  - DeformAlign: per-base deformable-conv alignment guided by event/motion
    features (lets the network correct small residual misalignments of each
    base prediction before fusing)
  - Charbonnier + gradient loss (helpers defined here, applied in trainer)
Output head identical to v1: per-pixel softmax gate over bases + residual.

Inputs / outputs
----------------
forward(x, bases) where
  x     = (vox, anchors): vox (B, 16, H, W) event voxel grid over the
          interpolation interval; anchors (B, 7, H, W) = anchor frame a0 (3)
          + anchor frame a1 (3) + scalar alpha map (1) giving the target
          temporal position.
  bases = tuple of n_bases tensors (B, 3, H, W): the base member
          predictions to be fused (RGB in [0, 1]).
Returns (out, gate): fused frame (B, 3, H, W) and the per-pixel softmax
gate (B, n_bases, H, W) for inspection/regularization.

Usage example
-------------
    python3 -c "
    import torch
    from refiner_v2arch import RefinerV2
    net = RefinerV2(n_bases=3, bins=16).cuda().eval()
    vox = torch.rand(1, 16, 256, 320, device='cuda')
    anchors = torch.rand(1, 7, 256, 320, device='cuda')
    bases = tuple(torch.rand(1, 3, 256, 320, device='cuda') for _ in range(3))
    with torch.no_grad():
        out, gate = net((vox, anchors), bases)
    print(out.shape, gate.shape)  # (1,3,256,320) (1,3,256,320)
    "
Requires torchvision (for `deform_conv2d`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class ConvBlock(nn.Module):
    """Two 3x3 convs with GELU — the basic building block of the U-Net."""

    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())

    def forward(self, x):
        return self.body(x)


class EventEncoder(nn.Module):
    """(B,16,H,W) voxel -> (B,ch,H,W) via per-bin conv + temporal attention."""

    def __init__(self, bins=16, ch=32):
        super().__init__()
        self.bins = bins
        self.bin_conv = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1))
        self.time_emb = nn.Parameter(torch.zeros(bins, ch))
        self.query = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(2 * ch, ch, 1)

    def forward(self, vox):
        B, T, H, W = vox.shape
        # Encode each temporal bin independently with a shared conv...
        f = self.bin_conv(vox.reshape(B * T, 1, H, W))           # (B*T,C,H,W)
        C = f.shape[1]
        # ...then add a learned per-bin (temporal position) embedding.
        f = f.reshape(B, T, C, H, W) + self.time_emb[None, :, :, None, None]
        # Per-pixel scaled dot-product attention over the T bins: the query
        # comes from the temporal mean, keys/values are the bin features.
        mean = f.mean(1)                                          # (B,C,H,W)
        q = self.query(mean)                                      # (B,C,H,W)
        att = torch.einsum("bchw,btchw->bthw", q, f) / (C ** 0.5)
        att = att.softmax(dim=1)                                  # over bins T
        pooled = torch.einsum("bthw,btchw->bchw", att, f)
        # Fuse attention-pooled and mean-pooled features into `ch` channels.
        return self.proj(torch.cat([pooled, mean], dim=1))


class DeformAlign(nn.Module):
    """Deformable 3x3 alignment of base features, offsets from guide feat."""

    def __init__(self, ch, guide_ch, groups=4):
        super().__init__()
        self.groups = groups
        # Predicts, per offset group and 3x3 kernel tap: 2 offset channels
        # (groups*2*9) plus 1 modulation-mask channel (groups*9).
        self.offset = nn.Sequential(
            nn.Conv2d(ch + guide_ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, groups * 2 * 9 + groups * 9, 3, padding=1))
        # Zero-init offsets/masks: at step 0 this behaves like a plain
        # (grouped) 3x3 conv with no spatial deformation.
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        # Explicit weight/bias parameters for the functional deform_conv2d.
        self.weight = nn.Parameter(torch.randn(ch, ch // groups, 3, 3) * 0.05)
        self.bias = nn.Parameter(torch.zeros(ch))

    def forward(self, feat, guide):
        # Offsets/masks are conditioned on both the base feature and the
        # event+anchor guide feature (which carries the motion cues).
        om = self.offset(torch.cat([feat, guide], dim=1))
        # Split into sampling offsets (groups*18 ch) and modulation mask.
        o, m = om[:, :self.groups * 18], om[:, self.groups * 18:]
        mask = torch.sigmoid(m)
        return deform_conv2d(feat, o, self.weight, self.bias, padding=1,
                             mask=mask)


class RefinerV2(nn.Module):
    """Gating U-Net that fuses `n_bases` member predictions into one frame."""

    def __init__(self, n_bases=3, bins=16, feat=32, base=56):
        super().__init__()
        self.n_bases = n_bases
        self.base_enc = ConvBlock(3, feat)          # shared across bases
        self.ev_enc = EventEncoder(bins, feat)
        self.anchor_enc = ConvBlock(7, feat)        # a0,a1,alpha-map
        # One deformable alignment module per base prediction, all guided
        # by the shared event+anchor feature.
        self.aligns = nn.ModuleList(
            DeformAlign(feat, feat * 2) for _ in range(n_bases))
        # 3-level U-Net over aligned base features + guide feature.
        cin = feat * n_bases + feat * 2
        self.enc0 = ConvBlock(cin, base)
        self.enc1 = ConvBlock(base, base * 2)
        self.enc2 = ConvBlock(base * 2, base * 4)
        self.dec1 = ConvBlock(base * 4 + base * 2, base * 2)
        self.dec0 = ConvBlock(base * 2 + base, base)
        # Output heads: per-pixel softmax gate over bases + RGB residual.
        self.head_gate = nn.Conv2d(base, n_bases, 3, padding=1)
        self.head_res = nn.Conv2d(base, 3, 3, padding=1)
        # Identity-friendly init: zero residual, and gate biases of
        # [1, ..., 1, 0] so the initial softmax slightly down-weights the
        # last base — the fusion starts close to a uniform blend of the
        # (usually stronger) leading bases instead of random gating.
        nn.init.zeros_(self.head_res.weight)
        nn.init.zeros_(self.head_res.bias)
        nn.init.zeros_(self.head_gate.weight)
        with torch.no_grad():
            b = torch.ones(n_bases)
            b[-1] = 0.0
            self.head_gate.bias.copy_(b)

    def forward(self, x, bases):
        """x: dict-free interface — x = (vox(B,16,H,W), anchors(B,7,H,W));
        bases: tuple of (B,3,H,W)."""
        vox, anchors = x
        # Shared guide feature: event temporal encoding + anchor encoding.
        ev = self.ev_enc(vox)
        an = self.anchor_enc(anchors)
        guide = torch.cat([ev, an], dim=1)
        # Encode each base prediction (shared encoder) and deformably align
        # it, correcting small per-member misalignments before fusion.
        feats = []
        for i, b in enumerate(bases):
            f = self.base_enc(b)
            feats.append(self.aligns[i](f, guide))
        # Standard U-Net with skip connections (avg-pool down, bilinear up).
        z = torch.cat(feats + [guide], dim=1)
        e0 = self.enc0(z)
        e1 = self.enc1(F.avg_pool2d(e0, 2))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        d1 = self.dec1(torch.cat([F.interpolate(e2, scale_factor=2,
                                                mode="bilinear"), e1], 1))
        d0 = self.dec0(torch.cat([F.interpolate(d1, scale_factor=2,
                                                mode="bilinear"), e0], 1))
        # Fused frame = gate-weighted sum of the bases plus a small residual
        # (tanh-bounded to +-0.1 so it can only make local corrections).
        gate = torch.softmax(self.head_gate(d0), dim=1)
        res = 0.1 * torch.tanh(self.head_res(d0))
        out = sum(gate[:, i:i + 1] * bases[i] for i in range(self.n_bases)) + res
        return out, gate


def charbonnier(pred, gt, eps=1e-3):
    """Charbonnier (smooth L1) loss — robust alternative to L2 for VFI."""
    return torch.sqrt((pred - gt) ** 2 + eps * eps).mean()


def gradient_l1(pred, gt):
    """L1 distance between the spatial gradients of pred and gt.

    Encourages sharp, edge-aligned reconstructions; used by the trainer as
    an auxiliary term next to the Charbonnier loss.
    """
    def g(x):  # kept for API compatibility; unused by the loss below
        return (x[..., :, 1:] - x[..., :, :-1]).abs().mean() + \
               (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    # Horizontal (dx) and vertical (dy) finite differences.
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_g = gt[..., :, 1:] - gt[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_g = gt[..., 1:, :] - gt[..., :-1, :]
    return (dx_p - dx_g).abs().mean() + (dy_p - dy_g).abs().mean()
