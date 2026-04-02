"""
test_pipeline.py -- Shape Verification Test for HoCVid Fusion Pipeline
=======================================================================

Runs a complete dummy forward pass through the entire pipeline and verifies:
    1. AFLB outputs (B, 48, H, W) -- with Bug 1 & 2 fixes
    2. DDER forward_features outputs (B, 512, h_p, w_p) -- with Bug 3 & 4 fixes
    3. CrossFrequencyGate outputs (B, 512, H, W)
    4. SPMAdapter outputs (B, 512, H, W) -- zero at init
    5. Full fusion pipeline outputs (B, 512, H, W) -- MCDB-ready
    6. No NaNs anywhere in the pipeline
    7. SPMAdapter zero-init verification (output = 0 at step 0)
    8. Gradient flow through trainable components
    9. Frozen backbone has no gradients

Usage:
    python test_pipeline.py
"""

import os
import sys
import torch
import torch.nn.functional as F

# Path setup
_project_dir = os.path.dirname(os.path.abspath(__file__))
_dder_dir = os.path.join(_project_dir, 'DDER')
if _dder_dir not in sys.path:
    sys.path.insert(0, _dder_dir)
_daclip_dir = os.path.join(_dder_dir, 'daclip')
if _daclip_dir not in sys.path:
    sys.path.insert(0, _daclip_dir)


def test_aflb_fixed():
    """Test AFLBFixed module with both bug fixes."""
    print("\n" + "=" * 60)
    print("TEST 1: AFLBFixed -- Bug 1 (.real) + Bug 2 (mask)")
    print("=" * 60)

    from aflb_fixed import AFLBFixed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    aflb = AFLBFixed(dim=48, num_heads=1, bias=False, in_dim=3).to(device)

    # Test with normal-sized image
    B, C, H, W = 2, 3, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    with torch.no_grad():
        freq_feat = aflb.forward_features(x)

    print(f"  Input:          {x.shape}")
    print(f"  Output:         {freq_feat.shape}")
    assert freq_feat.shape == (B, 48, H, W), f"Shape mismatch: {freq_feat.shape}"
    assert not torch.isnan(freq_feat).any(), "NaN detected in AFLB output!"
    print("  [OK] Normal image shape correct")

    # Test Bug 2 fix: small image (< 128px) should NOT produce empty mask
    H_small, W_small = 64, 64
    x_small = torch.randn(B, C, H_small, W_small, device=device)
    with torch.no_grad():
        freq_small = aflb.forward_features(x_small)

    print(f"  Small input:    ({B}, {C}, {H_small}, {W_small})")
    print(f"  Small output:   {freq_small.shape}")
    assert freq_small.shape == (B, 48, H_small, W_small)
    assert not torch.isnan(freq_small).any(), "NaN in small image output!"
    # Verify the output is not trivially zero (which would indicate empty mask bug)
    assert freq_small.abs().mean() > 1e-6, "AFLB output is near-zero -- mask bug may persist!"
    print("  [OK] Small image (Bug 2 fix) verified")

    # Test Bug 1 fix: output should not have systematic positive bias
    # With .real, output should have roughly zero mean. With abs(), mean > 0.
    high, low = aflb.fft(x)  # fft() internally calls conv1
    h_mean = high.mean().item()
    print(f"  High-freq mean: {h_mean:.6f} (should be ~0 with .real fix)")
    # abs() would give mean >> 0; .real should be near zero
    assert abs(h_mean) < 1.0, f"High-freq mean too large ({h_mean}), abs() bug may persist"
    print("  [OK] No positive bias (Bug 1 fix) verified")

    params = sum(p.numel() for p in aflb.parameters())
    print(f"  Parameters: {params:,} ({params / 1e6:.2f}M)")
    print("  [OK] TEST 1 PASSED")
    return True


def test_dder_fixed():
    """Test DDERFixedModule with Bug 3 (feat_proj) and Bug 4 (forward_features)."""
    print("\n" + "=" * 60)
    print("TEST 2: DDERFixedModule -- Bug 3 (feat_proj) + Bug 4 (forward_features)")
    print("=" * 60)

    from dder_fixed import DDERFixedModule

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use a smaller test to save memory
    dder = DDERFixedModule(num_experts=3).to(device)

    B = 1
    H, W = 128, 128
    x = torch.randn(B, 3, H, W, device=device)
    de_cls = torch.zeros(B, 6, device=device)

    # Test forward_features (Bug 4 fix)
    with torch.no_grad():
        feat, r_loss = dder.forward_features(x, de_cls)

    print(f"  Input:                ({B}, 3, {H}, {W})")
    print(f"  forward_features():   {feat.shape}")
    print(f"  r_loss:               {r_loss.item():.6f}")

    assert feat.shape[0] == B, f"Batch mismatch: {feat.shape[0]} vs {B}"
    assert feat.shape[1] == 512, f"Channel mismatch: {feat.shape[1]} vs 512 (Bug 3 fix failed)"
    assert not torch.isnan(feat).any(), "NaN in DDER features!"
    print("  [OK] forward_features returns (B, 512, h_p, w_p)")

    # Verify Bug 3: feat_proj exists and is a Conv2d(768, 512, 1)
    assert hasattr(dder, 'feat_proj'), "feat_proj not found (Bug 3 fix missing)"
    assert dder.feat_proj.in_channels == 768, f"feat_proj input: {dder.feat_proj.in_channels}"
    assert dder.feat_proj.out_channels == 512, f"feat_proj output: {dder.feat_proj.out_channels}"
    print("  [OK] feat_proj = Conv2d(768->512) exists")

    # Verify backward compat: forward() still returns pixel-space
    dder.eval()
    with torch.no_grad():
        pixel_out = dder(x, de_cls)
    if isinstance(pixel_out, tuple):
        pixel_out = pixel_out[0]
    assert pixel_out.shape == (B, 3, H, W), f"forward() shape: {pixel_out.shape}"
    print("  [OK] forward() still returns (B, 3, H, W) pixels")

    params = sum(p.numel() for p in dder.parameters())
    feat_proj_params = sum(p.numel() for p in dder.feat_proj.parameters())
    print(f"  Total params:   {params:,} ({params / 1e6:.2f}M)")
    print(f"  feat_proj params: {feat_proj_params:,} ({feat_proj_params / 1e6:.2f}M)")
    print("  [OK] TEST 2 PASSED")
    return True


def test_cross_frequency_gate():
    """Test CrossFrequencyGate module."""
    print("\n" + "=" * 60)
    print("TEST 3: CrossFrequencyGate")
    print("=" * 60)

    from fusion import CrossFrequencyGate

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gate = CrossFrequencyGate(aflb_dim=48, dder_dim=512).to(device)

    B, H, W = 2, 32, 32
    aflb_feat = torch.randn(B, 48, H, W, device=device)
    dder_feat = torch.randn(B, 512, H, W, device=device)

    with torch.no_grad():
        fused = gate(aflb_feat, dder_feat)

    print(f"  AFLB input:  {aflb_feat.shape}")
    print(f"  DDER input:  {dder_feat.shape}")
    print(f"  Output:      {fused.shape}")
    assert fused.shape == (B, 512, H, W), f"Shape: {fused.shape}"
    assert not torch.isnan(fused).any(), "NaN in CrossFrequencyGate output!"

    params = sum(p.numel() for p in gate.parameters())
    print(f"  Parameters:  {params:,} ({params / 1e6:.2f}M)")
    print("  [OK] TEST 3 PASSED")
    return True


def test_spm_adapter():
    """Test SPMAdapter module with zero-init verification."""
    print("\n" + "=" * 60)
    print("TEST 4: SPMAdapter -- Zero-init verification")
    print("=" * 60)

    from fusion import SPMAdapter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter = SPMAdapter(in_channels=2, out_channels=512).to(device)

    B, H, W = 2, 32, 32
    sam = torch.randn(B, 1, H, W, device=device)
    midas = torch.randn(B, 1, H, W, device=device)

    with torch.no_grad():
        out = adapter(sam, midas)

    print(f"  SAM input:   {sam.shape}")
    print(f"  MiDaS input: {midas.shape}")
    print(f"  Output:      {out.shape}")
    assert out.shape == (B, 512, H, W), f"Shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in SPMAdapter output!"

    # Zero-init check: output should be exactly zero at initialization
    max_val = out.abs().max().item()
    print(f"  Max |output|: {max_val:.10f} (should be 0.0)")
    assert max_val == 0.0, f"SPMAdapter output not zero at init: max={max_val}"
    print("  [OK] Zero-init verified (ControlNet trick working)")

    params = sum(p.numel() for p in adapter.parameters())
    print(f"  Parameters:  {params:,} ({params / 1e6:.2f}M)")
    print("  [OK] TEST 4 PASSED")
    return True


def test_fusion_pipeline():
    """Test complete HocVidFusionPipeline."""
    print("\n" + "=" * 60)
    print("TEST 5: HocVidFusionPipeline -- Full Integration")
    print("=" * 60)

    from fusion import HocVidFusionPipeline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = HocVidFusionPipeline(aflb_dim=48, dder_dim=512).to(device)

    B, H, W = 2, 64, 64
    aflb_feat = torch.randn(B, 48, 64, 64, device=device)
    dder_feat = torch.randn(B, 512, 4, 4, device=device)   # ViT spatial resolution
    sam_bnd = torch.rand(B, 1, 128, 128, device=device)      # Different resolution
    midas_dep = torch.rand(B, 1, 96, 96, device=device)      # Different resolution

    with torch.no_grad():
        mcdb_input = pipeline(
            aflb_features=aflb_feat,
            dder_features=dder_feat,
            sam_boundary=sam_bnd,
            midas_depth=midas_dep,
            target_h=H,
            target_w=W,
        )

    print(f"  AFLB features:    {aflb_feat.shape}")
    print(f"  DDER features:    {dder_feat.shape}")
    print(f"  SAM boundary:     {sam_bnd.shape}")
    print(f"  MiDaS depth:      {midas_dep.shape}")
    print(f"  Target:           ({B}, 512, {H}, {W})")
    print(f"  Output:           {mcdb_input.shape}")

    assert mcdb_input.shape == (B, 512, H, W), f"Shape: {mcdb_input.shape}"
    assert not torch.isnan(mcdb_input).any(), "NaN in fusion pipeline output!"
    print("  [OK] Output shape matches MCDB input requirement")
    print("  [OK] TEST 5 PASSED")
    return True


def test_full_model():
    """Test complete HocVidModel end-to-end."""
    print("\n" + "=" * 60)
    print("TEST 6: HocVidModel -- Complete End-to-End")
    print("=" * 60)

    from model import HocVidModel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = HocVidModel(
        dder_weights_path=os.path.join(_dder_dir, 'dder.pt'),
    ).to(device)

    B = 1
    H, W = 128, 128
    raw_input = torch.randn(B, 3, H, W, device=device)
    de_cls = torch.zeros(B, 6, device=device)
    sam_bnd = torch.rand(B, 1, H, W, device=device)
    midas_dep = torch.rand(B, 1, H, W, device=device)

    # -- Inference test --
    model.eval()
    with torch.no_grad():
        mcdb_input, r_loss = model(raw_input, de_cls, sam_bnd, midas_dep)

    print(f"  Raw input:        ({B}, 3, {H}, {W})")
    print(f"  MCDB input:       {mcdb_input.shape}")
    print(f"  r_loss:           {r_loss.item():.6f}")

    assert mcdb_input.shape == (B, 512, H, W), f"Shape: {mcdb_input.shape}"
    assert not torch.isnan(mcdb_input).any(), "NaN in model output!"
    print("  [OK] End-to-end shape correct: (B, 512, H, W)")

    # -- Gradient flow test --
    print("\n  Gradient flow test...")
    model.train()
    raw_input_grad = torch.randn(B, 3, H, W, device=device, requires_grad=False)
    mcdb_out, r_loss = model(raw_input_grad, de_cls, sam_bnd, midas_dep)
    loss = mcdb_out.mean() + r_loss
    loss.backward()

    # Check trainable components have gradients
    trainable_with_grad = 0
    trainable_without_grad = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                trainable_with_grad += 1
            else:
                trainable_without_grad += 1

    print(f"  Trainable params with gradient:    {trainable_with_grad}")
    print(f"  Trainable params without gradient:  {trainable_without_grad}")
    assert trainable_with_grad > 0, "No gradients flowing to trainable params!"

    # Check frozen components have NO gradients
    frozen_with_grad = 0
    for name, param in model.named_parameters():
        if not param.requires_grad and param.grad is not None:
            frozen_with_grad += 1
    print(f"  Frozen params with gradient:       {frozen_with_grad} (should be 0)")
    assert frozen_with_grad == 0, "Gradient leaked to frozen params!"
    print("  [OK] Gradient flow verified")

    # -- Test without SAM/MiDaS (graceful degradation) --
    model.eval()
    with torch.no_grad():
        mcdb_no_priors, _ = model(raw_input, de_cls)  # No SAM/MiDaS

    assert mcdb_no_priors.shape == (B, 512, H, W)
    assert not torch.isnan(mcdb_no_priors).any()
    print("  [OK] Graceful degradation without SAM/MiDaS")

    print("  [OK] TEST 6 PASSED")
    return True


def main():
    print("=" * 60)
    print("HoCVid Pipeline Shape Verification Tests")
    print("=" * 60)

    results = {}

    # Test individual components first
    results['AFLB Fixed'] = test_aflb_fixed()
    results['DDER Fixed'] = test_dder_fixed()
    results['CrossFrequencyGate'] = test_cross_frequency_gate()
    results['SPMAdapter'] = test_spm_adapter()
    results['FusionPipeline'] = test_fusion_pipeline()

    # Full end-to-end test
    results['Full Model'] = test_full_model()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {name:.<40} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] ALL TESTS PASSED -- Pipeline produces (B, 512, H, W) with no NaNs!")
    else:
        print("\n[FAIL] SOME TESTS FAILED")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
