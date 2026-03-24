"""
RES-FFTB Blocks & AdaIR Encoder-Decoder — EvoIR Stage 1

Converted from res_fftb.ipynb with the following bug fixes applied:
- Bug 2: Bottleneck uses num_fft_blocks[3] (not [1])
- Bug 5: FrequencyGuidedAttentionModule uses num_heads=4 (not out_channels)
- Bug 6: Sigmoid gate multiplies (not adds) in FFTransformerBlock
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


# ===================== UTILITY MODULES =====================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
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
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ===================== MDTA — Multi-DConv Head Transposed Self-Attention =====================

class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA) from Restormer."""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


# ===================== GDFN =====================

class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)."""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ===================== TransformerBlock =====================

class TransformerBlock(nn.Module):
    """Restormer Transformer block: MDTA + GDFN with pre-norm."""
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ===================== FSAS =====================

class FSAS(nn.Module):
    """Frequency-domain Self-Attention via patch-level rfft2."""
    def __init__(self, dim, bias=False):
        super().__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 2

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_patch = rearrange(q, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        k_patch = rearrange(k, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        out = self.norm(out)
        output = v * out
        return self.project_out(output)


# ===================== DFFN =====================

class DFFN(nn.Module):
    """Dynamic FFN with learnable FFT filter weights."""
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.patch_size = 2
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden, bias=bias)
        self.fft = nn.Parameter(torch.ones((hidden, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=self.patch_size, p2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


# ===================== Frequency Modules (inline from AFLB concepts) =====================

class FD(nn.Module):
    """Frequency Decomposition (same as in aflb.py)."""
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
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
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            n, self.group, c // self.group, self.kernel_size ** 2, h * w)
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(
            n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)
        low_filter = self.act(low_filter)
        low = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        high = identity_input - low
        return low, high


class FMgM(nn.Module):
    """Frequency Modulation generating Module (pre_att)."""
    def __init__(self, in_channels, flag_highF):
        super().__init__()
        self.flag_highF = flag_highF
        k_size = 3
        dim = in_channels
        if flag_highF:
            self.body = nn.Sequential(
                nn.Conv2d(dim, dim, (1, k_size), padding=(0, k_size // 2), groups=dim),
                nn.Conv2d(dim, dim, (k_size, 1), padding=(k_size // 2, 0), groups=dim),
                nn.GELU())
        else:
            self.body = nn.Sequential(nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, stride=1), nn.GELU())

    def forward(self, ffm):
        if self.flag_highF:
            return self.body(ffm) * ffm
        else:
            bs, c, H, W = ffm.shape
            y = torch.fft.rfft2(ffm.to(torch.float32))
            y_f = torch.cat([y.real, y.imag], dim=1)
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = torch.chunk(y_f, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            return torch.fft.irfft2(y, s=(H, W))


class FrequencyGuidedAttention(nn.Module):
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
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return self.project_out(out)


class FrequencyGuidedAttentionModule(nn.Module):
    """BUG 5 FIX: Uses num_heads=4, not out_channels."""
    def __init__(self, in_channels, out_channels, decoder_flag=False):
        super().__init__()
        self.decoder_flag = decoder_flag
        # FIX Bug 5: num_heads=4 explicitly
        self.high_freq_attention = FrequencyGuidedAttention(in_channels, num_heads=4)
        self.low_freq_attention = FrequencyGuidedAttention(in_channels, num_heads=4)
        if decoder_flag:
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.final_proj = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, hf, lf):
        ho = self.high_freq_attention(x, hf)
        lo = self.low_freq_attention(x, lf)
        if self.decoder_flag:
            a = torch.sigmoid(self.alpha)
            return self.proj(a * ho + (1 - a) * lo)
        else:
            return self.final_proj(torch.cat([ho, lo], dim=1))


class FMMPreWork(nn.Module):
    """FD -> FMgM -> FrequencyGuidedAttention + residual."""
    def __init__(self, decoder_flag=False, inchannels=48):
        super().__init__()
        self.fd = FD(inchannels)
        self.decoder_flag = decoder_flag
        self.freguide = FrequencyGuidedAttentionModule(inchannels, inchannels, decoder_flag)
        self.FSPG_high = FMgM(inchannels, flag_highF=True)
        self.FSPG_low = FMgM(inchannels, flag_highF=False)

    def forward(self, x):
        low_fre, high_fre = self.fd(x)
        high_fre_aft = self.FSPG_high(high_fre)
        low_fre_aft = self.FSPG_low(low_fre)
        prompt = self.freguide(x, high_fre_aft, low_fre_aft)
        return prompt + x


# ===================== FFTransformerBlock (RES-FFTB) =====================

class FFTransformerBlock(nn.Module):
    """
    RES-FFTB: FMMPreWork + optional FSAS + DFFN.

    BUG 6 FIX: Sigmoid gate now multiplies (was adding).
    """
    def __init__(self, dim, decoder_flag=True, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', att=False):
        super().__init__()
        self.att = att
        if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FSAS(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)
        self.prompt_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.Wp = nn.Parameter(torch.randn(dim, dim))
        self.prompt_block = FMMPreWork(decoder_flag=decoder_flag, inchannels=dim)

    def forward(self, x):
        prompt1 = self.prompt_block(x)
        prompt2 = self.prompt_conv(prompt1)
        prompt = torch.sigmoid(prompt2)
        # FIX Bug 6: multiply instead of add (sigmoid gate should modulate)
        x = prompt * prompt1
        if self.att:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ===================== Resizing Modules =====================

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ===================== AdaIR — Full Encoder-Decoder =====================

class AdaIR(nn.Module):
    """
    Full Restormer-style Encoder-Decoder with RES-FFTB blocks.

    BUG 2 FIX: Bottleneck uses num_fft_blocks[3] (was [1]).
    """
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4, 6, 6, 8], num_fft_blocks=[4, 2, 2, 1],
                 num_refinement_blocks=4, heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # --- Encoder ---
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])])
        self.encoder_level1_fft = nn.Sequential(*[
            FFTransformerBlock(dim) for _ in range(num_fft_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])])
        self.encoder_level2_fft = nn.Sequential(*[
            FFTransformerBlock(dim * 2) for _ in range(num_fft_blocks[1])])
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])])
        self.encoder_level3_fft = nn.Sequential(*[
            FFTransformerBlock(dim * 4) for _ in range(num_fft_blocks[2])])
        self.down3_4 = Downsample(dim * 4)

        # --- Bottleneck ---
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[3])])
        # FIX Bug 2: use num_fft_blocks[3] for bottleneck (was num_fft_blocks[1])
        self.latent_fft = nn.Sequential(*[
            FFTransformerBlock(dim * 8) for _ in range(num_fft_blocks[3])])

        # --- Decoder ---
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[2])])
        self.decoder_level3_fft = nn.Sequential(*[
            FFTransformerBlock(dim * 4, decoder_flag=True) for _ in range(num_fft_blocks[2])])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[1])])
        self.decoder_level2_fft = nn.Sequential(*[
            FFTransformerBlock(dim * 2, decoder_flag=True) for _ in range(num_fft_blocks[1])])

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_blocks[0])])
        self.decoder_level1_fft = nn.Sequential(*[
            FFTransformerBlock(dim, decoder_flag=True) for _ in range(num_fft_blocks[0])])

        # --- Refinement ---
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
            for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        # Encoder
        out_enc_level1_fft = self.encoder_level1_fft(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(out_enc_level1_fft) + inp_enc_level1

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2_fft = self.encoder_level2_fft(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(out_enc_level2_fft) + inp_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3_fft = self.encoder_level3_fft(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(out_enc_level3_fft) + inp_enc_level3

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4_fft = self.latent_fft(inp_enc_level4)
        latent = self.latent(out_enc_level4_fft) + inp_enc_level4

        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(self.decoder_level3_fft(inp_dec_level3)) + inp_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(self.decoder_level2_fft(inp_dec_level2)) + inp_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(self.decoder_level1_fft(inp_dec_level1)) + inp_dec_level1

        out_dec_level1 = self.refinement(out_dec_level1)
        return self.output(out_dec_level1) + inp_img  # Global residual
