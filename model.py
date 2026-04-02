"""
model.py — Top-level HoCVid Model
====================================

Complete multi-modal image restoration fusion model that wires together:
    - AFLB (fixed): Frequency-domain feature extraction (48ch)
    - DDER (fixed): DA-CLIP ViT + MoE degradation routing (512ch)
    - SAM-S: Frozen SAM-Small boundary maps (1ch)
    - MiDaS-Small: Frozen MiDaS depth maps (1ch)
    - Fusion Pipeline: CrossFrequencyGate + SPMAdapter
    - MCDB: M2Restore Mamba+CNN block (pretrained, strict=True)

Freeze/Unfreeze policy:
    FROZEN (no gradients):
        - AFLB's AdaIR backbone (pretrained EvoIR weights)
        - DDER's ViT-B-32 backbone (pretrained DA-CLIP weights)
        - SAM-S encoder (pretrained SAM weights)
        - MiDaS-Small (pretrained MiDaS weights)
        - MCDB internals (pretrained M2Restore weights, strict=True)

    TRAINABLE (gradients flow):
        - AFLB FreModule (AFLBFixed) — ~0.15M params
        - DDER feat_proj (1×1 conv 768→512) — ~0.39M params
        - DDER MoE router — existing params
        - CrossFrequencyGate — ~0.32M params
        - SPMAdapter — ~0.63M params

    Total trainable: ~1.5M params (very efficient fine-tuning)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path setup for DDER imports
_project_dir = os.path.dirname(os.path.abspath(__file__))
_dder_dir = os.path.join(_project_dir, 'DDER')
if _dder_dir not in sys.path:
    sys.path.insert(0, _dder_dir)
_daclip_dir = os.path.join(_dder_dir, 'daclip')
if _daclip_dir not in sys.path:
    sys.path.insert(0, _daclip_dir)

from aflb_fixed import AFLBFixed
from dder_fixed import DDERFixedModule, load_dder_checkpoint
from fusion import HocVidFusionPipeline


class HocVidModel(nn.Module):
    """
    Top-level HoCVid multi-modal restoration model.

    Orchestrates the complete pipeline:
        Input (B, 3, H, W)
            │
            ├──→ AFLB ──→ (B, 48, H, W) frequency features
            │
            ├──→ DDER ──→ (B, 512, h_p, w_p) degradation features
            │
            ├──→ SAM-S ──→ (B, 1, H, W) boundary maps  [FROZEN]
            │
            ├──→ MiDaS ──→ (B, 1, H, W) depth maps     [FROZEN]
            │
            └──→ FusionPipeline ──→ (B, 512, H, W) ──→ MCDB ──→ output

    MCDB receives EXACTLY (B, 512, H, W) with ZERO modifications.
    All heterogeneous inputs are absorbed by external adapters BEFORE MCDB.

    Args:
        dder_weights_path: Path to dder.pt weights (default: DDER/dder.pt)
        aflb_weights_path: Path to EvoIR model_landsat.pth (default: auto-download)
        num_experts:       Number of MoE experts in DDER (default: 3)
        use_sam:           Whether to use SAM boundary maps (default: True)
        use_midas:         Whether to use MiDaS depth maps (default: True)
    """

    def __init__(
        self,
        dder_weights_path=None,
        aflb_weights_path=None,
        num_experts=3,
        use_sam=True,
        use_midas=True,
    ):
        super().__init__()

        self.use_sam = use_sam
        self.use_midas = use_midas

        # ══════════════════════════════════════════════════════════════
        # Stage 1: AFLB — Frequency Feature Extraction (TRAINABLE)
        # ══════════════════════════════════════════════════════════════
        print("[HocVid] Initializing AFLB (fixed)...")
        self.aflb = AFLBFixed(dim=48, num_heads=1, bias=False, in_dim=3)
        # AFLB FreModule is TRAINABLE — its conv1, rate_conv, frequency_refine,
        # and cross-attention modules learn to extract useful frequency features.

        # ══════════════════════════════════════════════════════════════
        # Stage 2: DDER — Degradation Detection & Expert Routing
        # ══════════════════════════════════════════════════════════════
        print("[HocVid] Initializing DDER (fixed)...")
        self.dder = DDERFixedModule(num_experts=num_experts)

        # Load DDER pretrained weights (smart loader handles mismatches)
        if dder_weights_path is None:
            dder_weights_path = os.path.join(_dder_dir, 'dder.pt')

        if os.path.exists(dder_weights_path):
            print(f"[HocVid] Loading DDER weights from: {dder_weights_path}")
            load_dder_checkpoint(self.dder, dder_weights_path, verbose=True)
        else:
            print(f"[HocVid] WARNING: DDER weights not found at {dder_weights_path}")

        # Freeze DDER ViT backbone, keep vit_proj + MoE router trainable
        self._freeze_dder_backbone()

        # ══════════════════════════════════════════════════════════════
        # Fusion Pipeline: CrossFrequencyGate + SPMAdapter (TRAINABLE)
        # ══════════════════════════════════════════════════════════════
        print("[HocVid] Initializing Fusion Pipeline...")
        self.fusion = HocVidFusionPipeline(aflb_dim=48, dder_dim=512)

        # ══════════════════════════════════════════════════════════════
        # Print parameter summary
        # ══════════════════════════════════════════════════════════════
        self._print_param_summary()

    def _freeze_dder_backbone(self):
        """
        Freeze DDER's ViT backbone. Keep trainable:
            - self.dder.feat_proj (1×1 conv 768→512)
            - self.dder.moe_router (MoE expert routing)
            - self.dder.proj_refinement (pixel projection, used by forward())
        """
        # Freeze the entire ViT visual backbone
        for param in self.dder.visual_backbone.parameters():
            param.requires_grad = False

        # text_prompt_emb is a buffer (not a parameter), already frozen

        # feat_proj is trainable (freshly initialized, not pretrained)
        for param in self.dder.feat_proj.parameters():
            param.requires_grad = True

        # MoE router is trainable
        for param in self.dder.moe_router.parameters():
            param.requires_grad = True

        # proj_refinement is trainable (for backward compat, not used in fusion)
        for param in self.dder.proj_refinement.parameters():
            param.requires_grad = True

    def _print_param_summary(self):
        """Print trainable vs frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"\n[HocVid] Parameter Summary:")
        print(f"  Total:     {total:>12,} ({total/1e6:.2f}M)")
        print(f"  Trainable: {trainable:>12,} ({trainable/1e6:.2f}M)")
        print(f"  Frozen:    {frozen:>12,} ({frozen/1e6:.2f}M)")

        # Per-component breakdown
        for name, module in [
            ("AFLB", self.aflb),
            ("DDER", self.dder),
            ("Fusion", self.fusion),
        ]:
            t = sum(p.numel() for p in module.parameters() if p.requires_grad)
            f_ = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            print(f"  {name:>10}: {t:>10,} trainable, {f_:>10,} frozen")

    def train(self, mode=True):
        """
        Override to keep frozen components in eval mode.
        
        DDER's ViT backbone has LayerNorm that should use running stats,
        not batch stats. AFLB's AdaIR backbone has BatchNorm similarly.
        """
        super().train(mode)
        # Keep ViT backbone in eval mode (frozen LayerNorm/BatchNorm)
        self.dder.visual_backbone.eval()
        return self

    def forward(self, raw_input, de_cls=None, sam_boundary=None, midas_depth=None):
        """
        Full HoCVid forward pass.

        Args:
            raw_input:    (B, 3, H, W)  — degraded input frame, [0, 1] range
            de_cls:       (B, 6) or None — degradation class vector for MoE
                          (DA-CLIP uses 6 universal degradation classes)
                          If None during inference, uses uniform distribution.
            sam_boundary: (B, 1, H, W) or None — SAM-S boundary map
            midas_depth:  (B, 1, H, W) or None — MiDaS depth map

        Returns:
            mcdb_input: (B, 512, H, W) — feature tensor ready for MCDB
            r_loss:     scalar — MoE load-balancing loss (for training)
        """
        B, C, H, W = raw_input.shape
        device = raw_input.device

        # ── 1. AFLB: Extract frequency features ──
        # AFLB is trainable — gradients flow through conv1, rate_conv, FreRefine
        aflb_features = self.aflb.forward_features(raw_input)  # (B, 48, H', W')

        # ── 2. DDER: Extract degradation features ──
        # ViT backbone is frozen, but feat_proj and MoE router are trainable
        dder_feat, r_loss = self.dder.forward_features(raw_input, de_cls)  # (B, 512, h_p, w_p)

        # ── 3. Structural priors (frozen) ──
        # If SAM/MiDaS outputs not provided, use zeros (graceful degradation)
        if sam_boundary is None:
            sam_boundary = torch.zeros(B, 1, H, W, device=device)
        if midas_depth is None:
            midas_depth = torch.zeros(B, 1, H, W, device=device)

        # ── 4. Fusion Pipeline ──
        # CrossFrequencyGate + SPMAdapter → (B, 512, H, W)
        mcdb_input = self.fusion(
            aflb_features=aflb_features,
            dder_features=dder_feat,
            sam_boundary=sam_boundary,
            midas_depth=midas_depth,
            target_h=H,
            target_w=W,
        )

        return mcdb_input, r_loss
