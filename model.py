import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add DDER directory to path for its internal imports
_dder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DDER')
if _dder_dir not in sys.path:
    sys.path.insert(0, _dder_dir)

from dder import DDERModule
from evoIR_aflb.pretrained_model import AdaIR


class HocVidModel(nn.Module):
    """
    Two-stage pipeline:
      Stage 1: AFLB (AdaIR / EvoIR) - Frequency domain preprocessing (FROZEN)
      Stage 2: DDER (DDERModule)     - Degradation-aware expert routing (FROZEN)
    
    DDER receives the raw degraded frame directly (as it was trained on Colab).
    AFLB frequency features are fused with DDER output via a learnable adapter.
    """
    def __init__(self, dder_weights_path=None):
        super().__init__()
        
        # ── Stage 1: AFLB Block (frozen, loaded by pretrained_model.py itself) ──
        print("Loading AFLB Block (AdaIR)...")
        self.aflb = AdaIR()
        self.aflb.eval()  # Always eval mode — has BatchNorm that fails with small batches
        for param in self.aflb.parameters():
            param.requires_grad = False
        
        # ── Stage 2: DDER Block (frozen) ──
        print("Loading DDER Block (DDERModule)...")
        self.dder = DDERModule(num_experts=3)
        
        # Load dder.pt pretrained weights
        if dder_weights_path is None:
            dder_weights_path = os.path.join(_dder_dir, 'dder.pt')
        
        if os.path.exists(dder_weights_path):
            print(f"Loading DDER weights from: {dder_weights_path}")
            state_dict = torch.load(dder_weights_path, map_location='cpu', weights_only=False)
            # Handle wrapped state dicts
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            result = self.dder.load_state_dict(state_dict, strict=False)
            print(f"  DDER weights loaded: {len(state_dict) - len(result.unexpected_keys)} matched, "
                  f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
        else:
            print(f"WARNING: DDER weights not found at {dder_weights_path}, using random init")
        
        # Freeze all DDER parameters
        for param in self.dder.parameters():
            param.requires_grad = False
        
        # ── Fusion: Combines AFLB frequency features with DDER output ──
        # AFLB refinement produces 48-ch features, DDER produces 3-ch image
        # Fusion adapter merges them into the final output
        self.fusion = nn.Sequential(
            nn.Conv2d(48 + 3, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
        )
        
        # ── Hook to capture AFLB intermediate features ──
        self.aflb_features = None
        def hook_fn(module, inp, out):
            self.aflb_features = out
        self.aflb.refinement.register_forward_hook(hook_fn)

    def train(self, mode=True):
        """Override to keep AFLB and DDER always in eval mode (frozen with BatchNorm)."""
        super().train(mode)
        self.aflb.eval()
        self.dder.eval()
        return self

    def forward(self, input_frame):
        B, C, H, W = input_frame.shape
        
        # ── Stage 1: Run AFLB (frozen) ──
        with torch.no_grad():
            aflb_restored = self.aflb(input_frame)  # [B, 3, H, W]
        
        freq_features = self.aflb_features  # [B, 48, H, W] from hook
        if freq_features is None:
            raise RuntimeError("AFLB features not captured by the forward hook.")
        
        # ── Stage 2: Run DDER on raw frame (as trained on Colab) ──
        # DDER expects [B, 3, 224, 224] — resize, process, upscale back
        input_224 = F.interpolate(input_frame, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            dder_out = self.dder(input_224)  # [B, 3, 224, 224] or tuple
        
        if isinstance(dder_out, tuple):
            dder_img, r_loss = dder_out
        else:
            dder_img = dder_out
            r_loss = torch.tensor(0.0, device=input_frame.device)
        
        # Upscale DDER output back to original resolution
        dder_upscaled = F.interpolate(dder_img, size=(H, W), mode='bilinear', align_corners=False)
        
        # ── Fusion: Combine AFLB freq features + DDER output ──
        fused_input = torch.cat([freq_features.detach(), dder_upscaled], dim=1)  # [B, 51, H, W]
        refinement = self.fusion(fused_input)  # [B, 3, H, W]
        
        output = torch.clamp(input_frame + refinement, 0, 1)
        
        if self.training:
            return output, r_loss
        return output
