import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the frozen upstream model exactly as requested
from model import HocVidModel
# Import our new bridged M2Restore decoder
from m2restore_decoder import M2RestoreDecoder

# --- SSIM Loss Integration ---
try:
    from pytorch_msssim import ssim as ssim_fn, SSIM
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False
    print("[HocVidComplete] WARNING: pytorch_msssim not installed.")
    print("  Install with: python -m pip install pytorch-msssim")
    print("  Falling back to L1-only loss.")


class CombinedRestorationLoss(nn.Module):
    """
    Combined L1 + SSIM loss for image restoration.

    L1 loss: pixel-level accuracy (reduces PSNR metric)
    SSIM loss: structural similarity (reduces perceptual blur)
    MoE balance loss: prevents expert collapse in router

    Formula: alpha*L1 + (1-alpha)*(1-SSIM) + beta*router_loss
    alpha=0.84 from Restormer/M2Restore paper.
    """
    def __init__(self, alpha=0.84, beta=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        if _SSIM_AVAILABLE:
            self.ssim = SSIM(
                data_range=1.0,
                size_average=True,
                channel=3,
                nonnegative_ssim=True
            )
        self.use_ssim = _SSIM_AVAILABLE

    def forward(self, pred, target, router_loss=None):
        """
        Args:
            pred:        (B, 3, H, W) -- restored image, [0,1]
            target:      (B, 3, H, W) -- clean ground truth, [0,1]
            router_loss: scalar or None -- MoE balance loss
        Returns:
            total_loss, loss_dict (for logging)
        """
        l1 = F.l1_loss(pred, target)

        if self.use_ssim:
            ssim_val  = self.ssim(pred, target)
            ssim_loss = 1.0 - ssim_val
            pixel_loss = self.alpha * l1 + (1.0 - self.alpha) * ssim_loss
        else:
            ssim_val  = torch.tensor(0.0)
            ssim_loss = torch.tensor(0.0)
            pixel_loss = l1

        if router_loss is not None:
            total = pixel_loss + self.beta * router_loss
        else:
            total = pixel_loss

        loss_dict = {
            'total':   total.item(),
            'l1':      l1.item(),
            'ssim':    ssim_val.item() if self.use_ssim else 0.0,
            'pixel':   pixel_loss.item(),
            'router':  router_loss.item() if router_loss is not None else 0.0,
        }
        return total, loss_dict

class HocVidComplete(nn.Module):
    """
    Complete HoCVid: Degraded RGB (B,3,H,W) -> Restored RGB (B,3,H,W)
    
    Architecture:
        1. HocVidModel (Upstream) -> (B, 512, H, W) structural prior map + aux MoE loss
        2. M2RestoreDecoder (Trainable head) -> (B, 3, H, W) learned degradation offset
        3. Global residual: restored_output = decoder_output + degraded_input
    
    Training stages managed internally:
        Stage 1: Train fusion adapters + decoder head (Backbones strictly frozen)
        Stage 2: Unfreeze MoE router for fine-tuning degradation selection 
        Stage 3: Unfreeze AFLB FreModule for end-to-end exact tuning
    
    Args:
        dder_weights_path: Path to DDER backbone weights
        aflb_weights_path: Path to AFLB backbone weights
        num_experts: MoE DDER experts
        use_sam: Flag for SAM boundary activation
        use_midas: Flag for MiDaS depth activation
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
        
        # == 1. HoCVid Core Pipeline (Input -> 512 Channels) ==
        print("[HocVidComplete] Loading Core Pipeline...")
        self.upstream_model = HocVidModel(
            dder_weights_path=dder_weights_path,
            aflb_weights_path=aflb_weights_path,
            num_experts=num_experts,
            use_sam=use_sam,
            use_midas=use_midas
        )

        # == 2. M2Restore Decoder (512 Channels -> RGB) ==
        print("[HocVidComplete] Loading M2Restore Decoder Head...")
        self.decoder_head = M2RestoreDecoder(
            in_dim=512, 
            out_dim=3,
            dec_dim=96,           # Hardcoded M2Restore Level 1 dim
            num_blocks=4,         # Standard M2Restore level block size
            num_refinement_blocks=4
        )

        # == 3. Combined Restoration Loss ==
        self.loss_fn = CombinedRestorationLoss(alpha=0.84, beta=0.01)
        
        # Start strictly frozen 
        self.freeze_upstream()

    def forward(self, degraded_input, de_cls=None, sam_boundary=None, midas_depth=None, return_losses=False):
        """
        Args:
            degraded_input: (B, 3, H, W) degraded frame, [0, 1] range.
            de_cls: Optional manually provided degradation class vector.
            sam_boundary: Optional pre-computed SAM boundary map.
            midas_depth: Optional pre-computed MiDaS depth map.
            return_losses: Returns (restored_tensor, losses_dict) if True.
            
        Returns:
            restored: (B, 3, H, W) clamped to [0, 1]
            losses_dict: auxiliary components if return_losses=True
        """
        # 1. Forward through fusion backbones
        # Produces MCDB aligned features and MoE load-balancing loss
        mcdb_output, r_loss = self.upstream_model(
            degraded_input, 
            de_cls=de_cls, 
            sam_boundary=sam_boundary, 
            midas_depth=midas_depth
        )
        
        # 2. Forward through M2Restore reconstruction head
        # Because bridge_proj is Zero-Init, decoder_offset is guaranteed 0 at step 0
        decoder_offset = self.decoder_head(mcdb_output)

        # 3. Global residual exactly like M2Restore: (Output + Input)
        restored_image = decoder_offset + degraded_input
        
        # Clamp to expected [0, 1] RGB range
        restored_clamped = torch.clamp(restored_image, 0.0, 1.0)
        
        if return_losses:
            losses_dict = {
                'moe_routing_loss': r_loss
            }
            return restored_clamped, losses_dict
            
        return restored_clamped

    # ==============================================================
    # Freeze States and Parameter Logic 
    # ==============================================================

    def freeze_upstream(self):
        """Stage 1: Frozen backbone mode (train only head and adapters)"""
        # Freeze ALL upstream components first
        for param in self.upstream_model.parameters():
            param.requires_grad = False
            
        # Re-enable the dynamically injected fusion adapters
        for param in self.upstream_model.fusion.parameters():
            param.requires_grad = True
            
        # Decoder is brand new, must be fully trainable
        for param in self.decoder_head.parameters():
            param.requires_grad = True

    def unfreeze_stage2(self):
        """Stage 2: Unfreeze DDER MoE router for task fine-tuning"""
        # Unfreeze the MoE router in DDER to learn degradation combinations
        for param in self.upstream_model.dder.moe_router.parameters():
            param.requires_grad = True
        
        # Also ensure feat_proj learns appropriate embeddings
        for param in self.upstream_model.dder.feat_proj.parameters():
            param.requires_grad = True

    def unfreeze_stage3(self):
        """Stage 3: Unfreeze AFLB frequency modules for full end-to-end"""
        # Unfreeze the internal FreModule layers of the AdaIR setup
        for param in self.upstream_model.aflb.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Query generator to cleanly pass directly to Optimizer."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def get_param_groups(self, base_lr=2e-4):
        """
        Orchestrate multi-LR groups based on current freeze states.
        Keeps established backbones constrained while newly initialized heads learn fast.
        """
        groups = []
        
        # 1. Base group (brand new zero-init components)
        decoder_params = list(self.decoder_head.parameters())
        fusion_params = [p for p in self.upstream_model.fusion.parameters() if p.requires_grad]
        groups.append({'params': decoder_params + fusion_params, 'lr': base_lr})
        
        # 2. MoE Router Fine-Tuning group
        dder_params = [p for p in self.upstream_model.dder.parameters() if p.requires_grad]
        if len(dder_params) > 0:
            groups.append({'params': dder_params, 'lr': base_lr * 0.1})
            
        # 3. Frequency Fine-Tuning group
        aflb_params = [p for p in self.upstream_model.aflb.parameters() if p.requires_grad]
        if len(aflb_params) > 0:
            groups.append({'params': aflb_params, 'lr': base_lr * 0.01})
            
        return groups

    def get_parameter_summary(self):
        """Exhaustive printout ensuring perfectly efficient tuning boundaries."""
        total = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - train_params
        
        print("\n=== HoCVid Complete E2E Parameter Summary ===")
        print(f"Total Params:     {total:>12,} ({total/1e6:.2f}M)")
        print(f"Frozen:           {frozen:>12,} ({frozen/1e6:.2f}M)")
        print(f"Trainable Active: {train_params:>12,} ({train_params/1e6:.2f}M)")
        print("--- Trainable Breakdown ---")
        
        cnt_dec = sum(p.numel() for p in self.decoder_head.parameters() if p.requires_grad)
        cnt_fus = sum(p.numel() for p in self.upstream_model.fusion.parameters() if p.requires_grad)
        cnt_dder= sum(p.numel() for p in self.upstream_model.dder.parameters() if p.requires_grad)
        cnt_aflb= sum(p.numel() for p in self.upstream_model.aflb.parameters() if p.requires_grad)

        print(f"  M2Restore Decoder Head : {cnt_dec:>10,} params")
        print(f"  Fusion Adapters (CFG)  : {cnt_fus:>10,} params")
        if cnt_dder > 0:
            print(f"  DDER MoE/Proj Active   : {cnt_dder:>10,} params")
        if cnt_aflb > 0:
            print(f"  AFLB FreModule Active  : {cnt_aflb:>10,} params")
        print("=============================================\n")
