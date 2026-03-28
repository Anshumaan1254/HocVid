"""
dder_fixed.py — Fixed DDER Module for HoCVid Fusion Pipeline
=============================================================

This module wraps the original DDERModule and fixes the following bugs
documented in the HoCVid chat summary:

Bug 3 — Arbitrary channel truncation:
    Original: feats_map = feats_map[:, :512, :, :] — discards 256 of 768
    ViT channels with zero learning.
    Fix: Adds self.vit_proj = nn.Conv2d(768, 512, 1, bias=False), a 1×1
    convolution that LEARNS to project all 768 channels down to 512.

Bug 4 — forward() returns pixels instead of features:
    Original: returns torch.clamp(x + refinement, 0, 1) — 3-channel pixel
    output. MCDB needs (B,512,H,W) feature maps.
    Fix: Adds forward_features() that returns the MoE router output directly
    as (res_feat_map, r_loss) where res_feat_map is (B,512,H,W).

Design decision: We do NOT modify dder.py. Instead, DDERFixedModule
subclasses DDERModule, overrides __init__ to add vit_proj, and adds
forward_features() as a separate method. The original forward() is left
intact for backward compatibility / weight-loading sanity checks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Ensure DDER's directory is on path
_dder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DDER')
if _dder_dir not in sys.path:
    sys.path.insert(0, _dder_dir)

_daclip_dir = os.path.join(_dder_dir, 'daclip')
if _daclip_dir not in sys.path:
    sys.path.insert(0, _daclip_dir)

from DDER.dder import DDERModule


class DDERFixedModule(DDERModule):
    """
    Fixed DDER Module with learned ViT projection and feature extraction.

    Inherits DDERModule and adds:
        - vit_proj: nn.Conv2d(768, 512, 1, bias=False)
            Replaces the naive truncation feats_map[:, :512, :, :]
            with a learned 1×1 projection. This is trainable even though
            the ViT backbone itself is frozen.

        - forward_features(x, de_cls) → (feat_map, r_loss)
            Returns the (B, 512, H, W) feature map from the MoE router
            BEFORE spatial upsampling and pixel-space conversion.
            This is what the CrossFrequencyGate / MCDB expect.

    Parameter budget: vit_proj adds 768*512 = 393,216 parameters (~0.39M).
    """

    def __init__(self, num_experts=3):
        super().__init__(num_experts=num_experts)

        # ── Bug 3 Fix: Learned 1×1 projection instead of truncation ──
        # The ViT-B-32 backbone outputs 768-channel spatial feature maps.
        # The original code just sliced feats_map[:, :512, :, :], throwing
        # away the top 256 channels with no learning. This conv learns the
        # optimal linear combination of all 768 channels → 512.
        self.vit_proj = nn.Conv2d(768, 512, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.vit_proj.weight, mode='fan_out', nonlinearity='relu')

    def _extract_vit_features(self, x):
        """
        Run ViT backbone and return (B, 512, h_p, w_p) feature map.

        This replicates the frozen forward pass from DDERModule.forward()
        but uses self.vit_proj instead of naive channel truncation.

        Args:
            x: Input image tensor (B, C, H, W), expected to be [0,1] range.

        Returns:
            feats_map: (B, 512, h_p, w_p) projected ViT spatial features.
        """
        B, C, H, W = x.shape
        device = x.device

        with torch.no_grad():
            # Patch embedding via conv1
            x_tokens = self.visual_backbone.conv1(x)
            h_p, w_p = x_tokens.shape[-2:]
            x_tokens = x_tokens.reshape(B, 768, -1).permute(0, 2, 1)

            # Interpolate positional embeddings to current spatial size
            pos_embed = self.visual_backbone.positional_embedding
            cls_pos = pos_embed[:1, :]
            spatial_pos = pos_embed[1:, :]
            spatial_pos = spatial_pos.reshape(1, 7, 7, 768).permute(0, 3, 1, 2)
            spatial_pos = F.interpolate(
                spatial_pos, size=(h_p, w_p),
                mode='bicubic', align_corners=False
            )
            spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(-1, 768)
            current_pos = torch.cat([cls_pos, spatial_pos], dim=0).to(device)

            # Add CLS token and positional embeddings
            cls_token = self.visual_backbone.class_embedding.to(device, dtype=x.dtype)
            x_tokens = torch.cat([cls_token.expand(B, 1, -1), x_tokens], dim=1)
            x_tokens = x_tokens + current_pos.to(x.dtype)

            # Transformer forward pass
            x_tokens = self.visual_backbone.ln_pre(x_tokens)
            x_tokens = self.visual_backbone.transformer(x_tokens)

            # Extract spatial features (drop CLS token)
            spatial_feats = x_tokens[:, 1:, :].permute(0, 2, 1)
            feats_map_768 = spatial_feats.reshape(B, 768, h_p, w_p)

        # ── Bug 3 Fix: Learned projection replaces truncation ──
        # This 1×1 conv IS trainable (not inside torch.no_grad)
        feats_map = self.vit_proj(feats_map_768)  # (B, 512, h_p, w_p)

        return feats_map, h_p, w_p

    def forward_features(self, x, de_cls=None):
        """
        Bug 4 Fix: Return (B, 512, H_feat, W_feat) features for MCDB.

        Unlike forward() which returns pixel-space output [0,1],
        this method returns the raw MoE router output at the ViT
        spatial resolution (h_p × w_p). The caller is responsible
        for F.interpolate to the desired (H, W).

        Args:
            x:      Input image (B, 3, H, W).
            de_cls: Degradation classification vector (B, 6).
                    If None, defaults to zeros.

        Returns:
            res_feat_map: (B, 512, h_p, w_p) — MoE-routed features.
            r_loss:       Scalar — MoE load-balancing loss.
        """
        B = x.shape[0]
        device = x.device

        if de_cls is None:
            de_cls = torch.zeros(B, 6, device=device)

        # Extract ViT features with learned projection
        feats_map, h_p, w_p = self._extract_vit_features(x)

        # Text prompt embedding
        prompt = self.text_prompt_emb.expand(B, -1)

        # MoE router: routes features through top-k experts
        res_feat_map, r_loss = self.moe_router(feats_map, prompt, de_cls)

        return res_feat_map, r_loss

    def forward(self, x, de_cls=None):
        """
        Pixel-space forward pass (kept for backward compatibility).

        Now uses vit_proj instead of truncation but still outputs (B, 3, H, W).
        """
        B, C, H, W = x.shape
        device = x.device

        if de_cls is None:
            de_cls = torch.zeros(B, 6, device=device)

        # Use the corrected feature extraction
        feats_map, h_p, w_p = self._extract_vit_features(x)

        prompt = self.text_prompt_emb.expand(B, -1)
        res_feat_map, r_loss = self.moe_router(feats_map, prompt, de_cls)

        # Upsample and project to pixel space
        res_feat_upsampled = F.interpolate(
            res_feat_map, size=(H, W), mode='bilinear', align_corners=False
        )
        refinement = self.proj_refinement(res_feat_upsampled)
        output = torch.clamp(x + refinement, 0, 1)

        return (output, r_loss) if self.training else output
