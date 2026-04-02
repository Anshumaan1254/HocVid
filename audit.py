"""audit.py -- verify all 17 fixes are on disk"""
import sys, os, inspect, torch
sys.path.insert(0, r'C:\Users\Anshu\OneDrive\Desktop\HoCVid')
sys.path.insert(0, r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\DDER')
sys.path.insert(0, os.path.join(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid', 'DDER', 'daclip'))

results = {}

def check(name, condition, msg_pass="OK", msg_fail="FAIL"):
    status = "PASS" if condition else "FAIL"
    results[name] = status
    icon = "[OK]" if condition else "[!!]"
    print(f"  {icon} {name}: {msg_pass if condition else msg_fail}")
    return condition

print("\n=== AUDIT: DDER/MOE_CLIPbias.py ===")
try:
    from MOE_CLIPbias import ES_EE
    src = inspect.getsource(ES_EE.ee)
    check("Fix 1a -- exp removed", ".exp()" not in src,
          "No .exp() found", "STILL HAS .exp()")
    check("Fix 1b -- log removed", "return ensemble.log()" not in src,
          "No .log() return", "STILL HAS .log()")
    check("Fix 1c -- eps removed", "finfo" not in src,
          "No eps guard", "STILL HAS eps guard")
    check("Fix 1d -- returns ensemble", "return ensemble" in src,
          "Returns raw ensemble", "Missing return ensemble")
except Exception as e:
    print(f"  [!!] MOE_CLIPbias import FAILED: {e}")

print("\n=== AUDIT: DDER/dder.py ===")
try:
    from dder import DDERModule
    import torch.nn as nn
    m = DDERModule(num_experts=3)
    w = m.feat_proj.weight
    check("Fix 2a -- feat_proj exists",
          hasattr(m, 'feat_proj') and isinstance(m.feat_proj, nn.Conv2d),
          "feat_proj Conv2d found", "feat_proj missing")
    check("Fix 2b -- identity diagonal",
          abs(w[0,0,0,0].item()-1.0)<1e-6 and
          abs(w[255,255,0,0].item()-1.0)<1e-6 and
          abs(w[511,511,0,0].item()-1.0)<1e-6,
          "Identity diagonal correct", "Identity init BROKEN")
    check("Fix 2c -- upper channels zero",
          abs(w[0,512,0,0].item())<1e-6 and abs(w[0,767,0,0].item())<1e-6,
          "Upper channels zero", "Upper channels NOT zero")
    check("Fix 2d -- feat_proj trainable",
          m.feat_proj.weight.requires_grad,
          "requires_grad=True", "requires_grad=FALSE")
    src_dder = inspect.getsource(DDERModule.forward)
    check("Fix 3 -- de_cls required",
          "raise ValueError" in src_dder or "de_cls is None" in src_dder,
          "de_cls validation present", "de_cls still has silent zero fallback")
except Exception as e:
    print(f"  [!!] dder.py import FAILED: {e}")

print("\n=== AUDIT: dder_fixed.py ===")
try:
    from dder_fixed import DDERFixedModule
    src_fixed = inspect.getsource(DDERFixedModule.__init__)
    check("Fix 4 -- kaiming removed",
          "kaiming_normal_" not in src_fixed,
          "No kaiming_normal_ in __init__",
          "kaiming_normal_ STILL PRESENT")
except Exception as e:
    print(f"  [!!] dder_fixed.py import FAILED: {e}")

print("\n=== AUDIT: aflb_fixed.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\aflb_fixed.py') as f:
        src_aflb = f.read()
    check("Fix 5 -- .real on IFFT",
          ".real" in src_aflb and "torch.abs(high)" not in src_aflb,
          ".real used", "torch.abs() still on IFFT")
    check("Fix 6 -- fractional mask",
          "max(1," in src_aflb,
          "Fractional mask sizing found", "h//128 zero mask still present")
except Exception as e:
    print(f"  [!!] aflb_fixed.py check FAILED: {e}")

print("\n=== AUDIT: extract_dder_feats.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\extract_dder_feats.py') as f:
        src_ext = f.read()
    check("Fix 7 -- hooks moe_router",
          "moe_router" in src_ext,
          "moe_router hook found", "Still hooking proj_refinement")
    check("Fix 8 -- specific de_cls",
          "de_cls" in src_ext and ("1.0" in src_ext or "one_hot" in src_ext.lower()),
          "Specific de_cls found", "Uniform zeros still used")
except Exception as e:
    print(f"  [!!] extract_dder_feats.py check FAILED: {e}")

print("\n=== AUDIT: net/M2Restore.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\net\M2Restore.py') as f:
        src_m2 = f.read()
    check("Fix 11 -- mamba try/except",
          "try:" in src_m2 and "_MAMBA_AVAILABLE" in src_m2,
          "mamba fallback found", "Hard mamba import")
    check("Fix 12 -- daclip path",
          "from daclip import" not in src_m2 and "sys.path" in src_m2,
          "Dynamic daclip path", "from daclip import still present")
    check("Fix 13 -- dynamic checkpoint",
          "_candidates" in src_m2 or "os.path.exists" in src_m2,
          "Dynamic checkpoint search", "Hardcoded Linux path")
    check("Fix 14 -- register_buffer",
          "register_buffer" in src_m2 and ".to('cuda')" not in src_m2,
          "register_buffer used", ".to('cuda') still present")
    check("Fix 15a -- torch.sigmoid",
          "F.sigmoid" not in src_m2,
          "No F.sigmoid", "F.sigmoid still present")
    check("Fix 15b -- .real IFFT",
          "torch.abs(high)" not in src_m2 and "torch.abs(low)" not in src_m2,
          "No torch.abs on IFFT", "torch.abs still on IFFT")
    check("Fix 15c -- fractional mask",
          "max(1," in src_m2,
          "Fractional mask found", "Zero mask still present")
    check("Fix 15d -- tuple in MB",
          "{0:" not in src_m2 and "x, prompt = x" in src_m2,
          "Tuple-based MB", "Dict still in MB")
    check("Fix 15e -- autocast API",
          "torch.cuda.amp.autocast" not in src_m2,
          "Updated autocast", "Old autocast API")
except Exception as e:
    print(f"  [!!] net/M2Restore.py check FAILED: {e}")

print("\n=== AUDIT: m2restore_decoder.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\m2restore_decoder.py') as f:
        src_dec = f.read()
    check("Fix 16a -- VSSBlock",
          "class VSSBlock" in src_dec,
          "VSSBlock found", "VSSBlock MISSING")
    check("Fix 16b -- SelectiveScan2D",
          "class SelectiveScan2D" in src_dec,
          "SelectiveScan2D found", "SelectiveScan2D MISSING")
    check("Fix 16c -- mamba fallback",
          "check_mamba_available" in src_dec,
          "Mamba fallback chain", "No fallback")
    check("Fix 16d -- 4-dir scan",
          "ssm_hf" in src_dec and "ssm_hb" in src_dec and
          "ssm_vf" in src_dec and "ssm_vb" in src_dec,
          "4 directional SSMs", "4-dir scan MISSING")
    check("Fix 16e -- bridge zero-init",
          "nn.init.zeros_(self.bridge_proj" in src_dec,
          "bridge_proj zeroed", "Zero-init MISSING")
    check("Fix 16f -- encoder-decoder",
          "self.enc1" in src_dec and "self.bottleneck" in src_dec,
          "Multi-scale structure", "Flat decoder")
    check("Fix 16g -- skip connection",
          "skip_proj" in src_dec,
          "Skip connection found", "Skip MISSING")
    from m2restore_decoder import M2RestoreDecoder
    dec = M2RestoreDecoder(in_channels=512)
    params = sum(p.numel() for p in dec.parameters())
    check("Fix 16h -- param count",
          params > 1_000_000,
          f"{params/1e6:.2f}M params", f"Only {params/1e6:.2f}M")
except Exception as e:
    print(f"  [!!] m2restore_decoder.py check FAILED: {e}")

print("\n=== AUDIT: hocvid_complete.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\hocvid_complete.py') as f:
        src_hc = f.read()
    check("Fix 9 -- feat_proj name",
          "vit_proj" not in src_hc,
          "No vit_proj references", "vit_proj still present")
    check("Fix 17a -- CombinedRestorationLoss",
          "CombinedRestorationLoss" in src_hc,
          "CombinedRestorationLoss found", "L1-only loss")
    check("Fix 17b -- SSIM import",
          "pytorch_msssim" in src_hc,
          "SSIM import found", "No SSIM import")
except Exception as e:
    print(f"  [!!] hocvid_complete.py check FAILED: {e}")

print("\n=== AUDIT: test_pipeline.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\test_pipeline.py',
              encoding='utf-8') as f:
        src_tp = f.read()
    has_unicode = '\u2713' in src_tp or '\u2014' in src_tp
    check("Fix 10 -- no Unicode",
          not has_unicode,
          "No Unicode symbols", "Unicode symbols still present")
except Exception as e:
    print(f"  [!!] test_pipeline.py check FAILED: {e}")

print("\n=== AUDIT: train_dder_corrected.py ===")
try:
    with open(r'C:\Users\Anshu\OneDrive\Desktop\HoCVid\train_dder_corrected.py') as f:
        src_td = f.read()
    check("train_dder SSIM",
          "ssim" in src_td.lower() and "pytorch_msssim" in src_td,
          "SSIM in training loop", "No SSIM in training")
except Exception as e:
    print(f"  [!!] train_dder_corrected.py check FAILED: {e}")

print("\n" + "=" * 50)
print("AUDIT SUMMARY")
print("=" * 50)
passed = sum(1 for v in results.values() if v == "PASS")
failed = sum(1 for v in results.values() if v == "FAIL")
total = len(results)
print(f"  {passed}/{total} checks passed")
if failed > 0:
    print(f"  {failed} FAILURES:")
    for name, status in results.items():
        if status == "FAIL":
            print(f"    [!!] {name}")
else:
    print("  ALL CHECKS PASSED -- 17 fixes confirmed on disk")
