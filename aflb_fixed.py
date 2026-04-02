"""
aflb_fixed.py -- Fixed AFLB Module for HoCVid Fusion Pipeline
==============================================================

This module provides AFLBFixed, a standalone AFLB class with bug fixes
from the HoCVid chat summary applied. The original notebook aflb.ipynb
is NOT modified.

Bug 1 -- torch.abs() tinting:
    Original: abs() on complex IFFT result -- breaks conjugate symmetry. When you
    take abs() of a complex IFFT result, you discard sign information and
    fold negative lobes into positive, causing systematic positive bias
    (= color tinting, especially yellow/green shift).
    Fix: high = high.real (same as low branch already uses).
    Mathematical justification: For a real-valued input signal x, FFT(x)
    has Hermitian symmetry. Masking preserves symmetry, so IFFT returns
    real values with negligible imaginary part (~1e-7 from float rounding).
    .real discards this rounding noise correctly; abs() does not.

Bug 2 -- Empty mask for images < 128px:
    Original: h_ = (h // n * threshold[i, 0, :, :]).int() where n=128.
    When h < 128, h // 128 = 0, so h_ = 0 for ALL thresholds, producing
    an all-zero mask. This means low = 0 and high = entire spectrum,
    completely defeating frequency decomposition.
    Fix: Use h_ = max(1, int(h * threshold[i, 0, 0, 0].item() * 0.5))
    with proper centered mask construction. The 0.5 factor ensures the
    mask radius stays reasonable (at most half the spectrum).

Note on FFT norm: norm='ortho' is kept here. The chat summary confirms
norm='forward' for the pretrained model's internal FFT but the AFLB
notebook uses norm='ortho' for its own frequency decomposition. These
are two different FFT call sites -- the AFLB.fft() decomposition is
separate from the pretrained AdaIR model's internal DFFN/FSAS blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =====================================================================
# Supporting modules (copied from aflb notebook, unchanged)
# =====================================================================

class SpatialGate(nn.Module):
    """H->L Unit: spatial attention from high-frequency features."""
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, 1, keepdim=True)[0]
        mean_pool = torch.mean(x, 1, keepdim=True)
        scale = torch.cat([max_pool, mean_pool], dim=1)
        scale = self.spatial(scale)
        return torch.sigmoid(scale)


class ChannelGate(nn.Module):
    """L->H Unit: channel attention from low-frequency features."""
    def __init__(self, dim):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // 16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max_pool = self.mlp(self.max(x))
        return torch.sigmoid(avg + max_pool)


class FreRefine(nn.Module):
    """FMoM: Bidirectional cross-frequency modulation."""
    def __init__(self, dim):
        super().__init__()
        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight
        return self.proj(low + high)


class ChannelCrossAttention(nn.Module):
    """Channel-wise cross attention."""
    def __init__(self, dim, num_head=1, bias=False):
        super().__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1))
        self.q = nn.Conv2d(dim, dim, 1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)
        return self.project_out(out)


# =====================================================================
# Fixed AFLB
# =====================================================================

class AFLBFixed(nn.Module):
    """
    Adaptive Frequency Learning Block -- FIXED version.

    Produces (B, dim, H, W) frequency features from raw input images.
    Default dim=48 matches the pretrained AdaIR/EvoIR architecture.

    Pipeline:
        1. conv1: project input (3ch) -> dim channels
        2. fft(): adaptive frequency decomposition (FIXED)
            - FFT -> learned mask -> separate high/low -> IFFT (.real on BOTH)
        3. Cross-attention: freq components attend to encoder features
        4. FreRefine (FMoM): bidirectional cross-frequency modulation
        5. Aggregation cross-attention + learnable residual

    For the fusion pipeline, we call forward_features(x) which runs
    just the FFT decomposition and FreRefine, returning 48-channel
    frequency features without needing encoder features from AdaIR.

    Args:
        dim:    Feature channels (default 48)
        num_heads: Attention heads (default 1)
        bias:   Use bias in projections (default False)
        in_dim: Input image channels (default 3)
    """

    def __init__(self, dim=48, num_heads=1, bias=False, in_dim=3):
        super().__init__()

        # Input projection
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        # Adaptive threshold for frequency mask
        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        # Learnable residual scaling
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        # Cross-attention modules
        self.channel_cross_l = ChannelCrossAttention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = ChannelCrossAttention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = ChannelCrossAttention(dim, num_head=num_heads, bias=bias)

        # Frequency refinement (FMoM)
        self.frequency_refine = FreRefine(dim)

        # Learnable rate for adaptive mask threshold
        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, bias=False),
        )

    def shift(self, x):
        """Shift FFT to center (fftshift)."""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        """Inverse fftshift."""
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x, n=128):
        """
        FFT-based frequency decomposition with learnable adaptive mask.

        FIXES APPLIED:
            Bug 1: Both branches use .real instead of torch.abs().
            Bug 2: Mask construction uses max(1, ...) and proper scaling
                   for images of any size, not just >= 128px.

        Args:
            x: Input tensor (B, in_dim, H, W)
            n: Base divisor for threshold scaling (default 128)

        Returns:
            high: High-frequency features (B, dim, H, W)
            low:  Low-frequency features (B, dim, H, W)
        """
        x = self.conv1(x)  # Project to dim channels
        mask = torch.zeros(x.shape, device=x.device)
        h, w = x.shape[-2:]

        # Adaptive threshold from global context
        threshold = F.adaptive_avg_pool2d(x, 1)  # (B, dim, 1, 1)
        threshold = self.rate_conv(threshold).sigmoid()  # (B, 2, 1, 1)

        # Build frequency mask per sample
        for i in range(mask.shape[0]):
            # == Bug 2 Fix: Safe mask construction ==
            # Original: h_ = (h // n * threshold[...]).int()
            #   When h < n (128), h // n = 0, so h_ = 0 -> empty mask!
            # Fixed: Scale threshold by half the spectrum dimension directly.
            #   max(1, ...) ensures at least a 1-pixel mask radius.
            h_ = max(1, int(h * threshold[i, 0, 0, 0].item() * 0.5))
            w_ = max(1, int(w * threshold[i, 1, 0, 0].item() * 0.5))

            # Clamp to valid range (can't exceed half the spectrum)
            h_ = min(h_, h // 2)
            w_ = min(w_, w // 2)

            # Centered mask in frequency domain
            h_center = h // 2
            w_center = w // 2
            mask[i, :, h_center - h_:h_center + h_, w_center - w_:w_center + w_] = 1

        # Forward FFT (norm='ortho' for AFLB's own decomposition)
        fft = torch.fft.fft2(x, norm='ortho', dim=(-2, -1))
        fft = self.shift(fft)  # Center low frequencies

        # High frequency: everything OUTSIDE the mask
        fft_high = fft * (1 - mask)
        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='ortho', dim=(-2, -1))
        # == Bug 1 Fix: .real NOT torch.abs() ==
        # torch.abs() breaks conjugate symmetry -> positive bias -> color tint.
        # .real correctly discards negligible imaginary rounding noise.
        high = high.real

        # Low frequency: INSIDE the mask
        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='ortho', dim=(-2, -1))
        low = low.real  # (already correct in original, kept for consistency)

        return high, low

    def forward(self, x, y):
        """
        Full AFLB forward with encoder features (for use inside AdaIR).

        Args:
            x: Input image (B, in_dim, H_in, W_in)
            y: Encoder features (B, dim, H, W)
        Returns:
            Modulated features (B, dim, H, W)
        """
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)

        high_feature, low_feature = self.fft(x)

        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)

        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def forward_features(self, x, target_h=None, target_w=None):
        """
        Extract frequency features WITHOUT encoder features.

        For the fusion pipeline, we need AFLB's 48-channel frequency
        decomposition independently (there are no AdaIR encoder features
        to attend to). This method runs:
            1. FFT decomposition -> high, low
            2. FreRefine (FMoM) -> bidirectional gating
            3. Returns fused 48-channel features

        Args:
            x:        Input image (B, 3, H, W)
            target_h: Optional target height for output
            target_w: Optional target width for output

        Returns:
            freq_features: (B, 48, H, W) or (B, 48, target_h, target_w)
        """
        high, low = self.fft(x)

        # Bidirectional cross-frequency modulation
        fused = self.frequency_refine(low, high)

        if target_h is not None and target_w is not None:
            fused = F.interpolate(
                fused, size=(target_h, target_w),
                mode='bilinear', align_corners=False
            )

        return fused
