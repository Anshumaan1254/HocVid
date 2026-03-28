"""
fusion.py — HoCVid Fusion Pipeline Modules
============================================

This module implements the complete fusion pathway from AFLB + DDER features
through to the (B, 512, H, W) tensor that MCDB expects.

Architecture overview:
    ┌────────────┐    ┌────────────┐
    │ AFLB (48ch)│    │DDER (512ch)│
    └─────┬──────┘    └─────┬──────┘
          │                 │
          └──────┬──────────┘
                 ▼
        CrossFrequencyGate   ─────→  (B, 512, H, W)
                 │
                 +
                 │
        SPMAdapter(SAM+MiDaS) ───→ (B, 512, H, W)
                 │
                 ▼
           mcdb_input (B, 512, H, W)

Design rationale (from chat summary):

CrossFrequencyGate (~0.32M params):
    Naive concat treats frequency and degradation channels as same-kind info.
    Element-wise add after projection drowns AFLB because freshly initialized
    Conv2d(48→512) starts near zero (Xavier: σ ≈ 0.025).
    
    CrossFrequencyGate uses the SAME bidirectional gating logic as FreRefine
    (FMoM) in the AFLB notebook, but applied between modalities:
    
    - DDER's spatial statistics (max+mean pool → 7×7 conv → sigmoid) gate
      WHERE AFLB's frequency info enters. This is the SpatialGate pattern.
    
    - AFLB's channel statistics (GAP+GMP → MLP → sigmoid) gate WHICH DDER
      channels get boosted by frequency info. This is the ChannelGate pattern.
    
    Why this works: DDER knows about degradation-affected spatial regions;
    AFLB knows about frequency-domain artifact signatures. The gate lets
    each branch control the other's contributions based on its expertise.

SPMAdapter (~0.28M params):
    Converts SAM boundary maps (1ch) + MiDaS depth maps (1ch) → (B, 512, H, W).
    Uses a 3-stage ConvBN-LeakyReLU pyramid: 2→32→128→512.
    
    The last conv is ZERO-INITIALIZED (ControlNet trick): at training step 0,
    SPMAdapter outputs exactly zero, so the model starts from a stable
    AFLB+DDER-only baseline. Structural priors fade in as training progresses.
    
    Why this matters: Raw depth values are in [0,1] while learned features
    are ~N(0,σ). Zero-init avoids the scale mismatch problem entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFrequencyGate(nn.Module):
    """
    Fuses AFLB (48ch) + DDER (512ch) using bidirectional mutual gating.

    This is the FreRefine (FMoM) pattern from AFLB, applied cross-modality:
    
    1. Project AFLB 48→512 channels via 1×1 conv
    2. DDER's spatial statistics gate WHERE AFLB info enters (SpatialGate)
    3. AFLB's channel statistics gate WHICH DDER channels get boosted (ChannelGate)
    4. Fuse gated branches via learned projection

    Parameter count:
        aflb_proj:    48 * 512 = 24,576
        spatial_conv: 2 * 1 * 7 * 7 = 98
        channel_mlp:  512 * 32 + 32 * 512 = 32,768
        fuse_proj:    512 * 512 * 1 * 1 = 262,144
        Total: ~319,586 (~0.32M)

    Args:
        aflb_dim:  AFLB output channels (default 48)
        dder_dim:  DDER output channels (default 512)
    """

    def __init__(self, aflb_dim=48, dder_dim=512):
        super().__init__()

        # ── Step 1: Project AFLB channels to match DDER ──
        self.aflb_proj = nn.Conv2d(aflb_dim, dder_dim, kernel_size=1, bias=False)

        # ── Step 2: SpatialGate — DDER tells WHERE AFLB should contribute ──
        # Same pattern as AFLB's SpatialGate: max+mean pool → conv → sigmoid
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        # ── Step 3: ChannelGate — AFLB tells WHICH DDER channels to boost ──
        # Same pattern as AFLB's ChannelGate: GAP+GMP → shared MLP → sigmoid
        reduction = max(1, dder_dim // 16)  # 512 // 16 = 32
        self.channel_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_max = nn.AdaptiveMaxPool2d((1, 1))
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dder_dim, reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, dder_dim, 1, bias=False)
        )

        # ── Step 4: Fusion projection ──
        self.fuse_proj = nn.Conv2d(dder_dim, dder_dim, kernel_size=1, bias=False)

    def forward(self, aflb_feat, dder_feat):
        """
        Args:
            aflb_feat: (B, 48, H, W)  — AFLB frequency features
            dder_feat: (B, 512, H, W) — DDER degradation features

        Returns:
            fused: (B, 512, H, W) — gated cross-modal fusion
        """
        # Project AFLB to 512 channels
        aflb_512 = self.aflb_proj(aflb_feat)  # (B, 512, H, W)

        # ── SpatialGate: DDER controls WHERE aflb contributes ──
        # Pool DDER across channels → (B, 2, H, W) → conv → sigmoid
        dder_max = torch.max(dder_feat, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        dder_mean = torch.mean(dder_feat, dim=1, keepdim=True)    # (B, 1, H, W)
        spatial_desc = torch.cat([dder_max, dder_mean], dim=1)     # (B, 2, H, W)
        spatial_gate = torch.sigmoid(self.spatial_conv(spatial_desc))  # (B, 1, H, W)

        # Apply spatial gate to AFLB → WHERE to inject frequency info
        aflb_gated = aflb_512 * spatial_gate  # (B, 512, H, W)

        # ── ChannelGate: AFLB controls WHICH DDER channels get boosted ──
        # GAP + GMP of projected AFLB → shared MLP → sigmoid
        aflb_avg = self.channel_mlp(self.channel_avg(aflb_512))  # (B, 512, 1, 1)
        aflb_max = self.channel_mlp(self.channel_max(aflb_512))  # (B, 512, 1, 1)
        channel_gate = torch.sigmoid(aflb_avg + aflb_max)          # (B, 512, 1, 1)

        # Apply channel gate to DDER → WHICH channels to boost
        dder_gated = dder_feat * channel_gate  # (B, 512, H, W)

        # ── Fuse gated branches ──
        fused = self.fuse_proj(aflb_gated + dder_gated)  # (B, 512, H, W)

        return fused


class SPMAdapter(nn.Module):
    """
    Structural Prior Map Adapter.

    Encodes SAM boundary (1ch) + MiDaS depth (1ch) into (B, 512, H, W).
    Uses a 3-stage Conv-BN-LeakyReLU pyramid with zero-init on the last conv.

    Zero-init (ControlNet trick):
        At training step 0, the last conv outputs exactly zero regardless of
        input. This means the model starts from a stable AFLB+DDER-only
        baseline. Structural priors gradually fade in as the zero-initialized
        weights learn nonzero values.

    Parameter count:
        conv1: 2 * 32 * 3 * 3 = 576
        bn1:   32 * 2 = 64
        conv2: 32 * 128 * 3 * 3 = 36,864
        bn2:   128 * 2 = 256
        conv3: 128 * 512 * 3 * 3 = 589,824  (zero-init)
        Total: ~627,584 (~0.63M)
        
    Note: The parameter count is higher than the original ~0.28M estimate
    because 3×3 convs are used for spatial awareness. Can be reduced to
    1×1 convs if needed, but 3×3 gives the adapter local spatial context
    which is important for boundary/depth priors.

    Args:
        in_channels: Input channels (default 2: SAM + MiDaS concatenated)
        out_channels: Output channels (default 512)
    """

    def __init__(self, in_channels=2, out_channels=512):
        super().__init__()

        self.encoder = nn.Sequential(
            # Stage 1: 2 → 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: 32 → 128
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 3: 128 → 512 (ZERO-INITIALIZED)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=3, padding=1, bias=False)

        # ── ControlNet zero-init trick ──
        # At step 0: output = conv(features) = 0 for any input
        # Training gradually learns nonzero weights
        nn.init.zeros_(self.final_conv.weight)

    def forward(self, sam_boundary, midas_depth):
        """
        Args:
            sam_boundary: (B, 1, H, W) — SAM-S boundary probability map
            midas_depth:  (B, 1, H, W) — MiDaS-Small inverse depth map

        Returns:
            encoded: (B, 512, H, W) — structural prior features (zero at init)
        """
        # Concatenate structural priors
        x = torch.cat([sam_boundary, midas_depth], dim=1)  # (B, 2, H, W)

        # Encode through pyramid
        x = self.encoder(x)       # (B, 128, H, W)
        x = self.final_conv(x)    # (B, 512, H, W) — zero at init

        return x


class FinalFusionAdapter(nn.Module):
    """
    Small adapter to intelligently combine the gated frequency/degradation 
    features with the structural prior features.
    """
    def __init__(self, dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        )
        
        # Zero-init the last conv to maintain baseline stability initially
        nn.init.zeros_(self.conv[2].weight)

    def forward(self, gated_fused, spm_encoded):
        # Concatenate along channel dim (512 + 512 = 1024)
        x = torch.cat([gated_fused, spm_encoded], dim=1)
        # Adapt and add residual connection to baseline
        return self.conv(x) + gated_fused


class HocVidFusionPipeline(nn.Module):
    """
    Complete HoCVid Fusion Pipeline.

    Orchestrates the full forward pass from raw inputs to MCDB-ready
    (B, 512, H, W) features. This module owns the CrossFrequencyGate
    and SPMAdapter, and calls AFLB/DDER as external modules.

    Pipeline:
        1. AFLB: raw_input → (B, 48, H, W) frequency features
        2. DDER: raw_input → (B, 512, h_p, w_p) degradation features
        3. Spatial alignment via F.interpolate
        4. CrossFrequencyGate: AFLB + DDER → (B, 512, H, W) gated fusion
        5. SPMAdapter: SAM + MiDaS → (B, 512, H, W) structural priors
        6. Addition: gated_fused + spm_encoded → mcdb_input (B, 512, H, W)

    MCDB receives mcdb_input with ZERO code changes — it loads pretrained
    M2Restore weights with strict=True.

    Trainable components (everything else is frozen):
        - CrossFrequencyGate: ~0.32M params
        - SPMAdapter: ~0.63M params
        - DDER vit_proj: ~0.39M params
        - DDER MoE router: existing params from DDER training
        - AFLB FreModule: existing params from AFLB training

    Args:
        aflb_module:  AFLBFixed instance (external, caller manages freeze)
        dder_module:  DDERFixedModule instance (external, caller manages freeze)
        aflb_dim:     AFLB channel count (default 48)
        dder_dim:     DDER/MCDB channel count (default 512)
    """

    def __init__(self, aflb_dim=48, dder_dim=512):
        super().__init__()

        # Fusion modules (TRAINABLE)
        self.cross_freq_gate = CrossFrequencyGate(aflb_dim=aflb_dim, dder_dim=dder_dim)
        self.spm_adapter = SPMAdapter(in_channels=2, out_channels=dder_dim)
        self.final_adapter = FinalFusionAdapter(dim=dder_dim)

    def forward(self, aflb_features, dder_features, sam_boundary, midas_depth, target_h, target_w):
        """
        Fuse all feature streams into (B, 512, H, W) for MCDB.

        Args:
            aflb_features:  (B, 48, H_a, W_a)  — from AFLBFixed.forward_features()
            dder_features:  (B, 512, h_p, w_p) — from DDERFixedModule.forward_features()
            sam_boundary:   (B, 1, H_s, W_s)   — from frozen SAM-S
            midas_depth:    (B, 1, H_m, W_m)   — from frozen MiDaS-Small
            target_h:       Target spatial height (original input H)
            target_w:       Target spatial width (original input W)

        Returns:
            mcdb_input: (B, 512, target_h, target_w) — ready for MCDB
        """
        # ── Spatial alignment ──
        # All feature maps must match (target_h, target_w) before fusion
        aflb_aligned = F.interpolate(
            aflb_features, size=(target_h, target_w),
            mode='bilinear', align_corners=False
        )
        dder_aligned = F.interpolate(
            dder_features, size=(target_h, target_w),
            mode='bilinear', align_corners=False
        )
        sam_aligned = F.interpolate(
            sam_boundary, size=(target_h, target_w),
            mode='bilinear', align_corners=False
        )
        midas_aligned = F.interpolate(
            midas_depth, size=(target_h, target_w),
            mode='bilinear', align_corners=False
        )

        # ── Gated fusion of AFLB + DDER ──
        gated_fused = self.cross_freq_gate(aflb_aligned, dder_aligned)  # (B, 512, H, W)

        # ── Encode structural priors ──
        spm_encoded = self.spm_adapter(sam_aligned, midas_aligned)  # (B, 512, H, W)

        # ── Combine via Final Adapter ──
        # Uses concatenation and convolutions instead of direct element-wise addition
        mcdb_input = self.final_adapter(gated_fused, spm_encoded)  # (B, 512, H, W)

        return mcdb_input
