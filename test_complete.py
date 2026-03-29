import os
import sys
import torch
import unittest
from torch.cuda.amp import autocast

# To correctly find all imports explicitly
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from hocvid_complete import HocVidComplete

class TestHocVidComplete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Setting up TestHocVidComplete ===")
        # Fast configuration to avoid crushing CPU memory locally
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Instantiate end-to-end model
        # Disable SAM/MiDaS for tests to avoid loading massive foundation models just for shape tests
        cls.model = HocVidComplete(use_sam=False, use_midas=False).to(cls.device)
        print("Model instantiated securely.")

    def test_01_forward_shape_128(self):
        """Test 1: Forward pass shape — (1, 3, 128, 128) → (1, 3, 128, 128)"""
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
        
        self.assertEqual(output.shape, (B, C, H, W), "Shape mismatch on 128x128!")

    def test_02_forward_shape_256(self):
        """Test 2: Forward pass shape — (2, 3, 256, 256) → (2, 3, 256, 256)"""
        B, C, H, W = 2, 3, 256, 256
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            
        self.assertEqual(output.shape, (B, C, H, W), "Shape mismatch on batch size 2, 256x256!")

    def test_03_no_nan_or_inf(self):
        """Test 3: No NaN/Inf in output"""
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            
        self.assertFalse(torch.isnan(output).any(), "NaN detected in final output!")
        self.assertFalse(torch.isinf(output).any(), "Inf detected in final output!")

    def test_04_output_clamp_range(self):
        """Test 4: Output range — values in [0, 1] after clamp"""
        B, C, H, W = 1, 3, 128, 128
        # Provide adversarial input to force output out of bounds
        dummy_input = torch.rand(B, C, H, W, device=self.device) * 10.0 - 5.0
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            
        self.assertTrue((output >= 0.0).all(), "Values found below 0.0!")
        self.assertTrue((output <= 1.0).all(), "Values found above 1.0!")

    def test_05_gradient_flow_to_decoder(self):
        """Test 5: Gradient flow — gradients reach decoder params"""
        self.model.train()
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        dummy_input.requires_grad = True
        
        output = self.model(dummy_input)
        loss = output.sum()
        self.model.zero_grad()
        loss.backward()
        
        # Check if decoder parameters received gradients
        has_grad = False
        for name, param in self.model.decoder_head.named_parameters():
            if param.grad is not None and param.requires_grad:
                has_grad = True
                break
        self.assertTrue(has_grad, "No gradients reached the M2RestoreDecoder head!")

    def test_06_gradient_isolation(self):
        """Test 6: Gradient isolation — no gradients to frozen backbone params"""
        # Ensure model is strictly enforcing freeze rules
        self.model.freeze_upstream()
        
        # ViT backbone must be frozen
        vit_frozen = all(not p.requires_grad for p in self.model.upstream_model.dder.visual_backbone.parameters())
        self.assertTrue(vit_frozen, "DDER ViT visual backbone is leaking gradients (unfrozen)!")

        # M2Restore MCDB must be absent or frozen natively 
        # (It's not explicitly in model.py unless upstream loads it)
        # We ensure no gradients are leaking to ANY frozen parameter
        frozen_with_grad = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad and param.grad is not None:
                frozen_with_grad += 1
        self.assertEqual(frozen_with_grad, 0, "Frozen parameters are receiving gradients!")

    def test_07_global_residual_zero_init(self):
        """Test 7: Global residual — when decoder weights are zero-init, output ≈ input"""
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_input)
            
        # Since decoder bridge_proj is zero-initialized and output is clamped, 
        # and dummy_input is in [0, 1], output must mathematically equal dummy_input.
        diff = torch.abs(output - dummy_input).max().item()
        self.assertLess(diff, 1e-4, f"Zero-init ControlNet broken! Max diff between input and output: {diff}")

    def test_08_parameter_count(self):
        """Test 8: Parameter count — decoder head params match expected M2Restore subset"""
        dec_params = sum(p.numel() for p in self.model.decoder_head.parameters())
        # M2Restore's Level 1 operates heavily on 96 channels
        # Our bridge is approx 512*96 ~ 49K. 8 Transformer blocks on 96 channels + Mlp = ~950K
        # We expect it to be approx ~995K parameters.
        self.assertGreater(dec_params, 500_000, "Decoder head has too few parameters! (< 500K)")
        self.assertLess(dec_params, 2_000_000, "Decoder head has too many parameters! (> 2M)")

    def test_09_return_losses_mode(self):
        """Test 9: return_losses mode — losses_dict contains expected keys"""
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        self.model.eval()
        
        with torch.no_grad():
            output, losses_dict = self.model(dummy_input, return_losses=True)
            
        self.assertIsInstance(losses_dict, dict, "return_losses=True did not return a dictionary!")
        self.assertIn('moe_routing_loss', losses_dict, "Auxiliary MoE loss is missing from the dictionary!")
        self.assertEqual(output.shape, (B, C, H, W), "Output shape altered by return_losses flag!")

    def test_10_get_trainable_params(self):
        """Test 10: get_trainable_params() — returns correct subset"""
        trainable = list(self.model.get_trainable_params())
        all_params = list(self.model.parameters())
        
        self.assertGreater(len(trainable), 0, "No trainable parameters found!")
        self.assertLess(len(trainable), len(all_params), "All parameters are trainable (Backbones un-frozen)!")

    def test_11_batch_size_1_inference(self):
        """Test 11: Inference mode — works with batch_size=1 without BatchNorm crashes"""
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        self.model.eval()
        
        try:
            with torch.no_grad():
                _ = self.model(dummy_input)
            success = True
        except Exception as e:
            success = False
            print(f"Batch size 1 inference failed: {e}")
            
        self.assertTrue(success, "Inference crash on Batch Size = 1. Likely a BatchNorm tracking bug.")

    def test_12_amp_compatibility(self):
        """Test 12: AMP compatibility — forward pass works inside autocast"""
        B, C, H, W = 1, 3, 128, 128
        dummy_input = torch.rand(B, C, H, W, device=self.device)
        self.model.train()
        
        # Test AMP Mixed Precision
        try:
            with autocast():
                output = self.model(dummy_input)
                loss = output.sum()
            loss.backward()
            success = True
        except Exception as e:
            success = False
            print(f"AMP Execution Failed: {e}")
            
        self.assertTrue(success, "Mixed-precision AMP forward/backward pass failed!")

if __name__ == '__main__':
    unittest.main()
