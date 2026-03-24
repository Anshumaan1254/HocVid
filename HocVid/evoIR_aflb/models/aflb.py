"""
Adaptive Frequency Learning Block (AFLB) — EvoIR Stage 1

Converted from aflb.ipynb with the following bug fixes applied:
- Bug 1:  torch.abs() → .real on IFFT outputs (fixes color tint)
- Bug 1b: norm='forward'+'forward' → norm='ortho'+'ortho'
- Bug 3:  Integer division h//n=0 for small feature maps
- Bug 5:  num_heads=48 → num_heads=4 in FrequencyGuidedAttentionModule
- Dead code: Removed unused self.conv from AFLB.__init__
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


# ===================== UTILITY MODULES =====================

def to_3d(x):
    """Reshape (B, C, H, W) → (B, H*W, C) for layer norm."""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """Reshape (B, H*W, C) → (B, C, H, W)."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Spatial-aware layer norm: flatten → normalize → reshape."""
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ===================== FD — Frequency Decomposition =====================

class FD(nn.Module):
    """
    Frequency Decomposition via learnable low-pass filter.
    Uses adaptive average pooling + learned convolution to generate
    a dynamic low-pass filter kernel.
    """
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2,
                              kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity_input = x
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(
            self.pad(x), kernel_size=self.kernel_size
        ).reshape(n, self.group, c // self.group, self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(
            n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q
        ).unsqueeze(2)
        low_filter = self.act(low_filter)

        low = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        high = identity_input - low
        return low, high


# ===================== FMgM — Frequency Modulation generating Module =====================

class FMgM(nn.Module):
    """
    Two modes:
    - High-freq (flag_highF=True): depthwise separable spatial conv
    - Low-freq (flag_highF=False): FFT → spectral gating → IFFT
    """
    def __init__(self, in_channels, flag_highF):
        super().__init__()
        self.flag_highF = flag_highF
        k_size = 3
        dim = in_channels

        if flag_highF:
            self.body = nn.Sequential(
                nn.Conv2d(dim, dim, (1, k_size), padding=(0, k_size // 2), groups=dim),
                nn.Conv2d(dim, dim, (k_size, 1), padding=(k_size // 2, 0), groups=dim),
                nn.GELU()
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=1),
                nn.GELU(),
            )

    def forward(self, ffm):
        if self.flag_highF:
            y_att = self.body(ffm) * ffm
        else:
            bs, c, H, W = ffm.shape
            y = torch.fft.rfft2(ffm.to(torch.float32))
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = torch.chunk(y_f, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y_att = torch.fft.irfft2(y, s=(H, W))
        return y_att


# ===================== Spatial & Channel Gates (for FMoM) =====================

class SpatialGate(nn.Module):
    """H→L Unit: max+mean pooling → 7×7 conv → sigmoid."""
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, 1, keepdim=True)[0]
        mean_pool = torch.mean(x, 1, keepdim=True)
        scale = torch.cat([max_pool, mean_pool], dim=1)
        scale = self.spatial(scale)
        scale = torch.sigmoid(scale)
        return scale


class ChannelGate(nn.Module):
    """L→H Unit: GAP + GMP → MLP(C→C/16→C) → sigmoid."""
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
        scale = avg + max_pool
        scale = torch.sigmoid(scale)
        return scale


# ===================== FMoM — FreRefine =====================

class FreRefine(nn.Module):
    """Bidirectional cross-frequency modulation."""
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
        out = low + high
        out = self.proj(out)
        return out


# ===================== Channel Cross Attention =====================

class ChannelCrossAttention(nn.Module):
    """Channel-wise cross attention (transposed)."""
    def __init__(self, dim, num_head=1, bias=False):
        super().__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

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
        out = self.project_out(out)
        return out


# ===================== Frequency-Guided Attention =====================

class FrequencyGuidedAttention(nn.Module):
    """Cross-attention: Q from frequency features, K/V from backbone."""
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, freq_feat):
        B, C, H, W = x.shape
        q = self.q_dwconv(self.q_proj(freq_feat))
        kv = self.kv_dwconv(self.kv_proj(x))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.project_out(out)
        return out


class FrequencyGuidedAttentionModule(nn.Module):
    """
    Combines high-freq and low-freq guided attention with fusion.

    BUG 5 FIX: Previously passed out_channels as num_heads (e.g. 48).
    Now explicitly passes num_heads=4.
    """
    def __init__(self, in_channels, out_channels, decoder_flag=False):
        super().__init__()
        self.decoder_flag = decoder_flag
        # FIX Bug 5: use num_heads=4, NOT out_channels as num_heads
        self.high_freq_attention = FrequencyGuidedAttention(in_channels, num_heads=4)
        self.low_freq_attention = FrequencyGuidedAttention(in_channels, num_heads=4)

        if self.decoder_flag:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, high_freq, low_freq):
        high_freq_output = self.high_freq_attention(x, high_freq)
        low_freq_output = self.low_freq_attention(x, low_freq)

        if self.decoder_flag:
            alpha = torch.sigmoid(self.alpha)
            output = self.proj(alpha * high_freq_output + (1 - alpha) * low_freq_output)
        else:
            output = torch.cat([high_freq_output, low_freq_output], dim=1)
            output = self.final_proj(output)
        return output


# ===================== Full AFLB — FreModule =====================

class AFLB(nn.Module):
    """
    Adaptive Frequency Learning Block.

    BUG FIXES APPLIED:
    - Bug 1:  .real instead of torch.abs() on IFFT outputs
    - Bug 1b: norm='ortho' instead of norm='forward'
    - Bug 3:  Safe mask computation for small feature maps
    - Dead code: Removed unused self.conv (only self.conv1 is used)
    """
    def __init__(self, dim, num_heads=1, bias=False, in_dim=3):
        super().__init__()
        # FIX Dead code: removed self.conv (was never used, only self.conv1 is)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)
        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = ChannelCrossAttention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = ChannelCrossAttention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = ChannelCrossAttention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        high_feature, low_feature = self.fft(x)
        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)
        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)
        return out * self.para1 + y * self.para2

    def shift(self, x):
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h / 2), int(w / 2)), dims=(2, 3))

    def unshift(self, x):
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(-int(h / 2), -int(w / 2)), dims=(2, 3))

    def fft(self, x):
        """
        FFT-based frequency decomposition with learnable adaptive mask.

        BUG FIXES:
        - Bug 1:  Uses .real instead of torch.abs() (fixes tint)
        - Bug 1b: Uses norm='ortho' for both FFT and IFFT
        - Bug 3:  Uses max(1, ...) to prevent zero-size masks
        """
        x = self.conv1(x)
        mask = torch.zeros(x.shape, device=x.device)
        h, w = x.shape[-2:]

        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            # FIX Bug 3: Use float division and max(1, ...) to prevent zero-size masks
            h_ = max(1, int(h * threshold[i, 0, :, :].item() / 2))
            w_ = max(1, int(w * threshold[i, 1, :, :].item() / 2))
            mask[i, :, h // 2 - h_:h // 2 + h_, w // 2 - w_:w // 2 + w_] = 1

        # FIX Bug 1b: norm='ortho' instead of norm='forward'
        fft = torch.fft.fft2(x, norm='ortho', dim=(-2, -1))
        fft = self.shift(fft)

        # High frequency
        fft_high = fft * (1 - mask)
        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='ortho', dim=(-2, -1))
        # FIX Bug 1: .real instead of torch.abs()
        high = high.real

        # Low frequency
        fft_low = fft * mask
        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='ortho', dim=(-2, -1))
        # FIX Bug 1: .real instead of torch.abs()
        low = low.real

        return high, low


# ===================== FMMPreWork =====================

class FMMPreWork(nn.Module):
    """Complete FD → FMgM → FrequencyGuidedAttention + residual."""
    def __init__(self, decoder_flag=False, inchannels=48):
        super().__init__()
        self.fd = FD(inchannels)
        self.decoder_flag = decoder_flag
        self.freguide = FrequencyGuidedAttentionModule(
            in_channels=inchannels,
            out_channels=inchannels,
            decoder_flag=self.decoder_flag
        )
        self.FSPG_high = FMgM(in_channels=inchannels, flag_highF=True)
        self.FSPG_low = FMgM(in_channels=inchannels, flag_highF=False)

    def forward(self, x):
        low_fre, high_fre = self.fd(x)
        high_fre_aft = self.FSPG_high(high_fre)
        low_fre_aft = self.FSPG_low(low_fre)
        prompt = self.freguide(x, high_fre_aft, low_fre_aft)
        return prompt + x
