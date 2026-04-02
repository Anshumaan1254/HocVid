"""
dder_fixed.py — Fixed DDER Module for HoCVid Fusion Pipeline
=============================================================

This module wraps the original DDERModule and provides:

  - forward_features(x, de_cls) → (feat_map, r_loss)
      Returns the (B, 512, H, W) feature map from the MoE router
      BEFORE spatial upsampling and pixel-space conversion.
      This is what CrossFrequencyGate / MCDB expect.

All core bug fixes are now in the base DDERModule (dder.py):
  - feat_proj: Learnable 1×1 Conv2d(768, 512) replaces channel truncation
  - num_degradations=6 for DA-CLIP universal IR (6 degradation classes)
  - de_cls is required (no more silent zeros)
  - ES_EE.ee() no longer applies .exp()/.log() (fixed in MOE_CLIPbias.py)

Design decision: DDERFixedModule subclasses DDERModule and adds
forward_features() as a separate method. The original forward()
is left intact for backward compatibility / weight-loading.
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


def load_dder_checkpoint(model, ckpt_path, verbose=True):
    """
    Smart weight loader for DDER checkpoints.

    Handles mismatches gracefully:
      - feat_proj: NEW layer (not in old checkpoints) → kept at random init
      - degra_proj shape mismatch: if checkpoint has different num_degradations
        than model, the layer is skipped and kept at random init
      - All ViT backbone + MoE expert weights load normally

    Args:
        model:     DDERFixedModule (or DDERModule) instance
        ckpt_path: Path to dder.pt checkpoint
        verbose:   Print loading report

    Returns:
        dict with keys: matched, missing, unexpected, shape_mismatch
    """
    import torch

    if not os.path.exists(ckpt_path):
        if verbose:
            print(f"[load_dder_checkpoint] WARNING: {ckpt_path} not found")
        return {'matched': 0, 'missing': [], 'unexpected': [], 'shape_mismatch': []}

    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Handle nested checkpoint formats
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    model_dict = model.state_dict()
    shape_mismatch = []
    loadable = {}

    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                loadable[k] = v
            else:
                shape_mismatch.append(
                    f"  {k}: ckpt={list(v.shape)} vs model={list(model_dict[k].shape)}"
                )
        # else: unexpected key (not in model)

    # Load what we can
    model_dict.update(loadable)
    model.load_state_dict(model_dict, strict=False)

    missing = [k for k in model_dict if k not in state_dict]
    unexpected = [k for k in state_dict if k not in model_dict]

    if verbose:
        print(f"[load_dder_checkpoint] Loaded from: {ckpt_path}")
        print(f"  Matched:        {len(loadable)} / {len(model_dict)} parameters")
        print(f"  Missing (new):  {len(missing)} — kept at random init")
        for m in missing:
            print(f"    {m}: {list(model_dict[m].shape)}")
        if shape_mismatch:
            print(f"  Shape mismatch: {len(shape_mismatch)} — skipped (kept at random init)")
            for s in shape_mismatch:
                print(s)
        if unexpected:
            print(f"  Unexpected:     {len(unexpected)} — ignored")
            for u in unexpected[:5]:
                print(f"    {u}")

    return {
        'matched': len(loadable),
        'missing': missing,
        'unexpected': unexpected,
        'shape_mismatch': shape_mismatch,
    }


class DDERFixedModule(DDERModule):
    """
    Fixed DDER Module with feature extraction for the fusion pipeline.

    Inherits DDERModule (which now has all core fixes) and adds:

        - forward_features(x, de_cls) → (feat_map, r_loss)
            Returns the (B, 512, h_p, w_p) feature map from the MoE router
            BEFORE spatial upsampling and pixel-space conversion.
            This is what the CrossFrequencyGate / MCDB expect.

    The base class now includes:
        - self.feat_proj: nn.Conv2d(768, 512, 1) — learned ViT projection
        - num_degradations=6 — matches DA-CLIP universal IR checkpoint
        - de_cls is required (raises ValueError if None)
    """

    def __init__(self, num_experts=3):
        super().__init__(num_experts=num_experts)
        # feat_proj is already created in base DDERModule.__init__()
        # with truncation-equivalent identity init — do NOT re-initialize here

    def _extract_vit_features(self, x):
        """
        Run ViT backbone and return (B, 512, h_p, w_p) feature map.

        This replicates the frozen forward pass from DDERModule.forward()
        but returns features at the ViT spatial resolution.

        Args:
            x: Input image tensor (B, C, H, W), expected to be [0,1] range.

        Returns:
            feats_map: (B, 512, h_p, w_p) projected ViT spatial features.
            h_p: int — height of the patch grid.
            w_p: int — width of the patch grid.
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

        # Learnable projection (OUTSIDE no_grad — gradients flow)
        feats_map = self.feat_proj(feats_map_768)  # (B, 512, h_p, w_p)

        return feats_map, h_p, w_p

    def forward_features(self, x, de_cls=None):
        """
        Return (B, 512, H_feat, W_feat) features for the fusion pipeline.

        Unlike forward() which returns pixel-space output [0,1],
        this method returns the raw MoE router output at the ViT
        spatial resolution (h_p × w_p). The caller is responsible
        for F.interpolate to the desired (H, W).

        Args:
            x:      Input image (B, 3, H, W).
            de_cls: Degradation classification vector (B, 6).
                    If None, defaults to uniform distribution for inference.

        Returns:
            res_feat_map: (B, 512, h_p, w_p) — MoE-routed features.
            r_loss:       Scalar — MoE load-balancing loss.
        """
        B = x.shape[0]
        device = x.device

        # For inference, allow de_cls=None with a uniform default
        if de_cls is None:
            de_cls = torch.ones(B, 6, device=device) / 6.0

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

        Now uses feat_proj instead of truncation but still outputs (B, 3, H, W).
        """
        B, C, H, W = x.shape
        device = x.device

        # For inference, allow de_cls=None with a uniform default
        if de_cls is None:
            de_cls = torch.ones(B, 6, device=device) / 6.0

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
