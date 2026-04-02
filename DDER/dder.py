import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Ensure DDER's own directory is on path so MOE_CLIPbias and daclip resolve correctly
_dder_dir = os.path.dirname(os.path.abspath(__file__))
if _dder_dir not in sys.path:
    sys.path.insert(0, _dder_dir)

# Also add the daclip subfolder so open_clip imports work
_daclip_dir = os.path.join(_dder_dir, 'daclip')
if _daclip_dir not in sys.path:
    sys.path.insert(0, _daclip_dir)

from open_clip.model import CLIP, CLIPVisionCfg, CLIPTextCfg
from open_clip.tokenizer import tokenize
from MOE_CLIPbias import model as MOE_Router

class DDERModule(nn.Module):
    def __init__(self, num_experts=3):
        super(DDERModule, self).__init__()
        
        # 1. Initialize MoE router
        self.moe_router = MOE_Router(
            input_size=512,
            output_size=512,
            mlp_ratio=4,
            num_experts=num_experts,
            num_degradations=6
        )
        
        # Learnable 1x1 projection: ViT-B/32 768-dim → 512-dim
        # MUST be outside torch.no_grad() so gradients flow through it
        self.feat_proj = nn.Conv2d(768, 512, kernel_size=1, bias=False)
        # Initialize as truncation-equivalent: weight[i,i,0,0] = 1 for i=0..511
        # This makes feat_proj behave like the old feats_map[:,:512] at init,
        # so pretrained MoE weights work correctly without retraining.
        # During fine-tuning, it gradually learns to use all 768 channels.
        nn.init.zeros_(self.feat_proj.weight)
        with torch.no_grad():
            for i in range(512):
                self.feat_proj.weight[i, i, 0, 0] = 1.0
        
        # 2. Instantiate ViT-B-32 Architecture (Skeleton only, weights loaded later)
        vision_cfg = CLIPVisionCfg(layers=12, width=768, head_width=64, patch_size=32, image_size=224)
        text_cfg = CLIPTextCfg(context_length=77, vocab_size=49408, width=512, heads=8, layers=12)
        model = CLIP(embed_dim=512, vision_cfg=vision_cfg, text_cfg=text_cfg, quick_gelu=True)
        
        self.visual_backbone = model.visual
        for param in self.visual_backbone.parameters():
            param.requires_grad = False

        # 3. Create the text prompt embedding for "a degraded image"
        with torch.no_grad():
            tokens = tokenize(["a degraded image"])
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self.register_buffer("text_prompt_emb", text_features)
        
        # 4. Spatial Projection for MoE Output
        self.proj_refinement = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        
    def forward(self, x, de_cls):
        B, C, H, W = x.shape
        device = x.device
        
        if de_cls is None:
            raise ValueError(
                "de_cls is required — pass a (B, 6) degradation label vector. "
                "DA-CLIP uses 6 universal degradation classes. "
                "Map your dataset classes accordingly."
            )
        
        with torch.no_grad():
            x_tokens = self.visual_backbone.conv1(x) 
            h_p, w_p = x_tokens.shape[-2:]
            x_tokens = x_tokens.reshape(B, 768, -1).permute(0, 2, 1) 
            
            pos_embed = self.visual_backbone.positional_embedding 
            cls_pos = pos_embed[:1, :]
            spatial_pos = pos_embed[1:, :]
            spatial_pos = spatial_pos.reshape(1, 7, 7, 768).permute(0, 3, 1, 2)
            spatial_pos = F.interpolate(spatial_pos, size=(h_p, w_p), mode='bicubic', align_corners=False)
            spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(-1, 768)
            current_pos = torch.cat([cls_pos, spatial_pos], dim=0).to(device)

            cls_token = self.visual_backbone.class_embedding.to(device, dtype=x.dtype)
            x_tokens = torch.cat([cls_token.expand(B, 1, -1), x_tokens], dim=1)
            x_tokens = x_tokens + current_pos.to(x.dtype)
            
            x_tokens = self.visual_backbone.ln_pre(x_tokens)
            x_tokens = self.visual_backbone.transformer(x_tokens)
            
            spatial_feats = x_tokens[:, 1:, :].permute(0, 2, 1) 
            feats_map = spatial_feats.reshape(B, 768, h_p, w_p)

        # Learnable 1x1 projection (OUTSIDE no_grad — gradients flow)
        feats_map = self.feat_proj(feats_map)  # (B, 768, h, w) → (B, 512, h, w)

        prompt = self.text_prompt_emb.expand(B, -1)
        res_feat_map, r_loss = self.moe_router(feats_map, prompt, de_cls)
        
        res_feat_upsampled = F.interpolate(res_feat_map, size=(H, W), mode='bilinear', align_corners=False)
        refinement = self.proj_refinement(res_feat_upsampled)
        
        output = torch.clamp(x + refinement, 0, 1)
        
        return (output, r_loss) if self.training else output