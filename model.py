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
        
        # No fusion adapter layer needed since DDER is sequential
        # No hook needed since we extract frequency residual mathematically

    def train(self, mode=True):
        """Override to keep AFLB and DDER always in eval mode (frozen with BatchNorm)."""
        super().train(mode)
        self.aflb.eval()
        self.dder.eval()
        return self

    def forward(self, input_frame):
        B, C, H, W = input_frame.shape
        
        # ── Stage 1: AFLB (AdaIR) Frequency Extraction ──
        with torch.no_grad():
            full_restored = self.aflb(input_frame)  # [B, 3, H, W]
        
        # Extract purely the frequency distribution map (AdaIR: restored = freq + input)
        freq_map = full_restored - input_frame
        
        # ── Stage 2: DDER Degradation Error Segregation ──
        # DDER expects 224x224 input
        freq_map_224 = F.interpolate(freq_map, size=(224, 224), mode='bilinear', align_corners=False)
        
        # DDER was originally trained to output: output = input + degradation_map
        # We will feed it the frequency map directly. Since DDER expects normalized values,
        # its internal batch/layer normalizations will center the frequency shifts automatically.
        with torch.no_grad():
            dder_out = self.dder(freq_map_224)
        
        if isinstance(dder_out, tuple):
            dder_full, r_loss = dder_out
        else:
            dder_full = dder_out
            r_loss = torch.tensor(0.0, device=input_frame.device)
            
        # Extract purely the degradation / error mapping 
        degradation_map_224 = dder_full - freq_map_224
        
        # Upscale back to initial resolution
        final_degradation_map = F.interpolate(degradation_map_224, size=(H, W), mode='bilinear', align_corners=False)
        
        if self.training:
            return final_degradation_map, r_loss
        return final_degradation_map
