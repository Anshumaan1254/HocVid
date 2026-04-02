"""
m2restore_decoder.py -- Full M2Restore VSS Decoder for HoCVid
=============================================================
Converts (B, 512, H, W) fusion features -> (B, 3, H, W) RGB residual.

Architecture:
  bridge_proj  -> Zero-init Conv2d(512->96)     [ControlNet trick]
  encoder_l1   -> 2x VSSBlock @ 96ch           [Multi-scale encode]
  downsample   -> Conv2d(96->192, stride=2)
  encoder_l2   -> 2x VSSBlock @ 192ch          [Deeper features]
  bottleneck   -> 4x VSSBlock @ 192ch           [Core SSM processing]
  upsample     -> ConvTranspose2d(192->96)
  decoder_l1   -> 2x VSSBlock @ 96ch + skip    [Decode + skip conn]
  refinement   -> 4x VSSBlock @ 96ch            [Final refinement]
  output       -> Conv2d(96->3)

SSM backend: mamba_ssm CUDA if available, pure PyTorch fallback otherwise.

CRITICAL DESIGN CONTRACT:
  - Input: (B, 512, H, W) from HoCVid fusion pipeline
  - Output: (B, 3, H, W) RGB residual offset
  - At init: output is EXACTLY 0 (bridge_proj zero-initialized)
  - Class name MUST stay M2RestoreDecoder (hocvid_complete.py imports it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- SSM Backend Detection ------------------------------------------------

def check_mamba_available():
    """Returns (bool, str) -- availability and backend name."""
    try:
        from mamba_ssm import Mamba
        return True, "mamba_ssm (CUDA accelerated)"
    except (ImportError, Exception):
        return False, "pure PyTorch SSM fallback"

_mamba_ok, _mamba_mode = check_mamba_available()
print(f"[M2RestoreDecoder] SSM backend: {_mamba_mode}")


# --- Component 1: Windows-Safe Selective Scan 2D -------------------------

class SelectiveScan2D(nn.Module):
    """
    1D selective scan applied to 2D feature sequences.
    Processes (B, L, d_model) -- sequence format.

    Uses mamba_ssm CUDA kernel if available.
    Falls back to pure PyTorch implementation automatically.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.use_mamba = False

        if _mamba_ok:
            try:
                from mamba_ssm import Mamba
                self.ssm = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.use_mamba = True
            except Exception:
                pass  # Fall through to PyTorch fallback

        if not self.use_mamba:
            # Pure PyTorch SSM -- full selective state space formulation
            inner = d_model * expand
            self.in_proj  = nn.Linear(d_model, inner * 2)
            self.conv1d   = nn.Conv1d(
                inner, inner,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=inner
            )
            self.act      = nn.SiLU()
            self.A_log    = nn.Parameter(
                torch.log(
                    torch.arange(1, d_state + 1, dtype=torch.float32)
                    .repeat(inner, 1)
                )
            )
            self.D        = nn.Parameter(torch.ones(inner))
            self.dt_proj  = nn.Linear(inner, inner)
            self.out_proj = nn.Linear(inner, d_model)

    def _pytorch_ssm(self, x):
        """
        x: (B, L, d_model) -> (B, L, d_model)
        Implements discretized selective SSM scan in pure PyTorch.
        """
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        # Causal depthwise conv along sequence dimension
        x_in = self.conv1d(
            x_in.transpose(1, 2)
        )[..., :L].transpose(1, 2)
        x_in = self.act(x_in)
        # Discretized state space: A, B, C selection
        A  = -torch.exp(self.A_log.float())          # (inner, d_state)
        dt = F.softplus(self.dt_proj(x_in))          # (B, L, inner)
        dA = torch.exp(dt.unsqueeze(-1) * A)         # (B, L, inner, d_state)
        y  = (x_in.unsqueeze(-1) * dA).sum(-1)       # (B, L, inner)
        y  = y + x_in * self.D                       # skip connection
        y  = y * self.act(z)                         # gating
        return self.out_proj(y)                      # (B, L, d_model)

    def forward(self, x):
        if self.use_mamba:
            return self.ssm(x)
        return self._pytorch_ssm(x)


# --- Component 2: GDFN ---------------------------------------------------

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network from Restormer.
    Input/output: (B, C, H, W)
    """
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.project_in  = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.dconv       = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2,
            kernel_size=3, padding=1, groups=hidden_dim * 2
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop        = nn.Dropout(drop)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dconv(x).chunk(2, dim=1)
        return self.project_out(self.drop(F.gelu(x1) * x2))


# --- Component 3: Channel Attention --------------------------------------

class ChannelAttention(nn.Module):
    """
    Squeeze-Excitation style channel attention.
    Input/output: (B, C, H, W)
    """
    def __init__(self, dim, reduction=16):
        super().__init__()
        mid = max(dim // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.fc((self.avg(x) + self.mx(x)).view(B, C))
        return x * a.view(B, C, 1, 1)


# --- Component 4: VSSBlock -----------------------------------------------

class VSSBlock(nn.Module):
    """
    Visual State Space Block -- core of M2Restore.

    Scans the 2D feature map in 4 independent directions:
      -> Horizontal forward   (left-to-right context)
      <- Horizontal backward  (right-to-left context)
      v  Vertical forward     (top-to-bottom context)
      ^  Vertical backward    (bottom-to-top context)

    Merges all 4 scans with element-wise interaction term.
    Then applies GDFN for channel mixing and ChannelAttention.

    Input/output: (B, C, H, W)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 4 independent directional SSMs
        self.ssm_hf = SelectiveScan2D(dim, d_state, d_conv, expand)
        self.ssm_hb = SelectiveScan2D(dim, d_state, d_conv, expand)
        self.ssm_vf = SelectiveScan2D(dim, d_state, d_conv, expand)
        self.ssm_vb = SelectiveScan2D(dim, d_state, d_conv, expand)

        # Merge: [h_scan, v_scan, h*v interaction, skip] -> dim
        self.merge = nn.Linear(dim * 4, dim)

        # Channel mixing + attention
        self.gdfn = GDFN(dim, int(dim * mlp_ratio))
        self.ca   = ChannelAttention(dim)

    def _scan(self, x2d, ssm_f, ssm_b, horizontal=True):
        """
        Scan x2d (B, H, W, C) along one axis in both directions.
        Returns (B, H, W, C) merged result.
        """
        B, H, W, C = x2d.shape
        if horizontal:
            seq = x2d.reshape(B * H, W, C)
            fwd = ssm_f(seq)
            bwd = ssm_b(seq.flip(1)).flip(1)
            return (fwd + bwd).reshape(B, H, W, C)
        else:
            seq = x2d.permute(0, 2, 1, 3).reshape(B * W, H, C)
            fwd = ssm_f(seq)
            bwd = ssm_b(seq.flip(1)).flip(1)
            return (fwd + bwd).reshape(B, W, H, C).permute(0, 2, 1, 3)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x2d = x.permute(0, 2, 3, 1)                # (B, H, W, C)

        # LayerNorm before scanning
        normed = self.norm1(x2d)

        # 4-directional scans
        h_scan = self._scan(normed, self.ssm_hf,
                            self.ssm_hb, horizontal=True)
        v_scan = self._scan(normed, self.ssm_vf,
                            self.ssm_vb, horizontal=False)

        # Merge: horizontal + vertical + cross-interaction + skip
        merged = torch.cat([h_scan,
                            v_scan,
                            h_scan * v_scan,
                            x2d],
                           dim=-1)                  # (B, H, W, C*4)
        out_2d = self.merge(merged) + x2d           # residual

        # Back to (B, C, H, W) for channel attention
        out = out_2d.permute(0, 3, 1, 2)
        out = self.ca(out)

        # GDFN with pre-norm
        out_2d = self.norm2(out.permute(0, 2, 3, 1))
        out = self.gdfn(out_2d.permute(0, 3, 1, 2)) + out

        return out


# --- Component 5: Full M2RestoreDecoder -----------------------------------

class M2RestoreDecoder(nn.Module):
    """
    Full M2Restore Decoder with VSS (Visual State Space) blocks.

    Input:  (B, 512, H, W) -- mcdb_input from HoCVid fusion pipeline
    Output: (B, 3, H, W)   -- RGB residual offset

    bridge_proj is ZERO-INITIALIZED (ControlNet trick):
      At init, output is EXACTLY 0 for any input.
    """

    def __init__(
        self,
        in_channels=512,
        base_dim=96,
        num_vss_enc=2,
        num_vss_bot=4,
        num_vss_dec=2,
        num_vss_ref=4,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        # Accept but ignore legacy kwargs for backward compatibility
        in_dim=None,
        out_dim=None,
        dec_dim=None,
        num_blocks=None,
        num_refinement_blocks=None,
    ):
        super().__init__()

        # Handle legacy kwarg from old hocvid_complete.py calls
        if in_dim is not None:
            in_channels = in_dim
        if dec_dim is not None:
            base_dim = dec_dim

        vss_kwargs = dict(
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
        )
        D, D2 = base_dim, base_dim * 2

        # -- Bridge: 512 -> 96 -- ZERO-INITIALIZED --
        self.bridge_proj = nn.Conv2d(in_channels, D, kernel_size=1)
        nn.init.zeros_(self.bridge_proj.weight)
        nn.init.zeros_(self.bridge_proj.bias)

        # -- Encoder Level 1: 96ch --
        self.enc1 = nn.Sequential(*[
            VSSBlock(D, **vss_kwargs) for _ in range(num_vss_enc)
        ])

        # -- Downsample: 96 -> 192, H/2 x W/2 --
        self.down = nn.Conv2d(D, D2, kernel_size=2, stride=2)

        # -- Encoder Level 2: 192ch --
        self.enc2 = nn.Sequential(*[
            VSSBlock(D2, **vss_kwargs) for _ in range(num_vss_enc)
        ])

        # -- Bottleneck: 192ch -- core SSM processing --
        self.bottleneck = nn.Sequential(*[
            VSSBlock(D2, **vss_kwargs) for _ in range(num_vss_bot)
        ])

        # -- Upsample: 192 -> 96 --
        self.up = nn.ConvTranspose2d(D2, D, kernel_size=2, stride=2)

        # -- Skip connection: cat([up, enc1]) = 192ch -> 96ch --
        self.skip_proj = nn.Conv2d(D2, D, kernel_size=1)

        # -- Decoder Level 1: 96ch --
        self.dec1 = nn.Sequential(*[
            VSSBlock(D, **vss_kwargs) for _ in range(num_vss_dec)
        ])

        # -- Refinement: 96ch --
        self.refine = nn.Sequential(*[
            VSSBlock(D, **vss_kwargs) for _ in range(num_vss_ref)
        ])

        # -- Output: 96 -> 3 --
        self.out_proj = nn.Conv2d(D, 3, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            x: (B, 512, H, W) -- fused features from HoCVid pipeline
        Returns:
            (B, 3, H, W) -- RGB residual offset
        """
        # Bridge: 512->96. Zero-init -> outputs 0 at step 0.
        x = self.bridge_proj(x)                    # (B, 96, H, W)

        # Encode
        e1  = self.enc1(x)                         # (B, 96, H, W)
        e2  = self.enc2(self.down(e1))             # (B, 192, H/2, W/2)

        # Bottleneck (4 VSSBlocks -- full 4-directional SSM)
        bot = self.bottleneck(e2)                  # (B, 192, H/2, W/2)

        # Decode with skip connection from enc1
        up  = self.up(bot)                         # (B, 96, H, W)
        # Handle odd spatial dimensions from downsampling
        if up.shape[-2:] != e1.shape[-2:]:
            up = F.interpolate(up, size=e1.shape[-2:],
                               mode='bilinear', align_corners=False)
        dec = self.skip_proj(
            torch.cat([up, e1], dim=1)
        )                                          # (B, 96, H, W) via skip
        dec = self.dec1(dec)                       # (B, 96, H, W)

        # Refinement
        ref = self.refine(dec)                     # (B, 96, H, W)

        # Output projection
        return self.out_proj(ref)                  # (B, 3, H, W)
