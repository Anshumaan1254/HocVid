import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# ══════════════════════════════════════════════════════════════
# M2Restore Decoder Components
# Extracted from M2Restore-main/net/M2Restore.py to avoid 
# mamba_ssm compilation errors on Windows environments.
# These are identical to the official M2Restore implementation.
# ══════════════════════════════════════════════════════════════

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# ══════════════════════════════════════════════════════════════
# HoCVid M2Restore Decoder Bridge Wrap
# ══════════════════════════════════════════════════════════════

class M2RestoreDecoder(nn.Module):
    """
    M2Restore decoder head adapted for the HoCVid pipeline.
    
    Architecture sourced from: https://github.com/yz-wang/M2Restore
    File: `M2restore-main/net/M2Restore.py`

    **Bridge Strategy:**
    HoCVid's outputs are deeply aligned full-resolution structural priors `(B, 512, H, W)`.
    M2Restore's latent bottleneck expects a massive downsample `(B, 384, H/8, W/8)`. 
    To avoid destroying our SAM boundaries, we mathematically bridge HoCVid's output directly 
    into M2Restore's final, full-resolution "Level 1" decoder head via a `1x1` zero-initialized 
    ControlNet-style projection.

    Args:
        in_dim: Input channels from MCDB (default: 512)
        out_dim: Output RGB channels (default: 3)
        dec_dim: M2Restore Level 1 Decoder inner dimension (default: 96)
        num_blocks: Number of TransformerBlocks per sequence (default: 4)
        num_refinement_blocks: Number of final TransformerBlocks (default: 4)
        heads: Attention heads (default: 1 for Level 1)
        ffn_expansion_factor: Expansion factor for GDFN (default: 2.66)
        bias: Bias setting for convolutional layers (default: False)
        LayerNorm_type: 'WithBias' or 'BiasFree' (default: 'WithBias')
    """
    def __init__(
        self,
        in_dim=512,
        out_dim=3,
        dec_dim=96,
        num_blocks=4,
        num_refinement_blocks=4,
        heads=1,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ):
        super().__init__()
        
        # 1. Zero-Initialized ControlNet Bridge Projection 
        # Resolves the (512 -> 96) channel mismatch
        self.bridge_proj = nn.Conv2d(in_dim, dec_dim, kernel_size=1)
        nn.init.constant_(self.bridge_proj.weight, 0.0)
        nn.init.constant_(self.bridge_proj.bias, 0.0)

        # 2. M2Restore `decoder_level1`
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dec_dim, 
                num_heads=heads, 
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, 
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks)
        ])

        # 3. M2Restore `refinement`
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=dec_dim, 
                num_heads=heads, 
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias, 
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])

        # 4. M2Restore `output`
        self.output = nn.Conv2d(dec_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        """
        Forward pass restoring the residual image.
        
        Args:
            x: (B, 512, H, W) Output from HoCVid MCDB block.
            
        Returns:
            residual: (B, 3, H, W) The learned degradation offset.
        """
        # (B, 512, H, W) -> (B, 96, H, W)
        x = self.bridge_proj(x)
        
        # Safe propagation through M2Restore blocks at H,W resolution
        x = self.decoder_level1(x)
        x = self.refinement(x)
        
        # (B, 96, H, W) -> (B, 3, H, W)
        residual = self.output(x)
        
        return residual
