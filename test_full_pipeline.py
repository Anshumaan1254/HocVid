"""
test_full_pipeline.py
=====================
Full HoCVid pipeline visualization for bacha.png.
Saves 9 component outputs to test_output_full/ folder.

Run: python test_full_pipeline.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -- Path setup ----------------------------------------------------------------
ROOT       = os.path.dirname(os.path.abspath(__file__))
DDER_DIR   = os.path.join(ROOT, 'DDER')
DACLIP_DIR = os.path.join(DDER_DIR, 'daclip')
AFLB_DIR   = os.path.join(ROOT, 'evoIR_aflb')

for p in [ROOT, DDER_DIR, DACLIP_DIR, AFLB_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Output folder -------------------------------------------------------------
OUT_DIR = os.path.join(ROOT, 'test_output_full')
os.makedirs(OUT_DIR, exist_ok=True)

# -- Device --------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Setup] Device: {device}")
print(f"[Setup] Output folder: {OUT_DIR}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_image(arr, filename, title=None):
    """Save a numpy array as an image with optional title bar."""
    path = os.path.join(OUT_DIR, filename)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)

    if title:
        h, w = arr.shape[:2]
        bar_h = 36
        bar = np.ones((bar_h, w, 3), dtype=np.uint8) * 30
        cv2.putText(bar, title, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if arr.ndim == 2:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) \
                      if arr.shape[2] == 3 else arr
        combined = np.vstack([bar, arr_rgb])
        cv2.imwrite(path, combined)
    else:
        if arr.ndim == 2:
            cv2.imwrite(path, arr)
        else:
            cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    print(f"  [Saved] {filename}")
    return path


def tensor_to_np(t):
    """(1,C,H,W) or (C,H,W) tensor [0,1] -> HxWxC numpy uint8"""
    if t.dim() == 4:
        t = t.squeeze(0)
    t = t.detach().cpu().float()
    t = torch.clamp(t, 0, 1)
    arr = t.permute(1, 2, 0).numpy() if t.shape[0] == 3 \
          else t.squeeze(0).numpy()
    return (arr * 255).astype(np.uint8)


def feat_to_heatmap(feat, orig_h, orig_w):
    """Convert a (1, C, H, W) feature tensor to JET heatmap (HxWx3 RGB)."""
    if feat.dim() == 4:
        heat = feat.abs().mean(dim=1).squeeze()
    else:
        heat = feat.squeeze()
    heat = heat.detach().cpu().float().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_u8 = (heat * 255).astype(np.uint8)
    heat_resized = cv2.resize(heat_u8, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)
    colormap = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
    return cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)


def overlay(orig_rgb, heatmap_rgb, alpha=0.5):
    """Blend original image with heatmap. Both HxWx3 uint8."""
    orig_f  = orig_rgb.astype(np.float32)
    heat_f  = heatmap_rgb.astype(np.float32)
    blended = (1 - alpha) * orig_f + alpha * heat_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def power_spectrum(img_tensor):
    """Compute log power spectrum. Returns HxW numpy float [0,1]."""
    gray = img_tensor.mean(dim=1, keepdim=True)
    fft  = torch.fft.fft2(gray.squeeze(), norm='forward')
    fft  = torch.fft.fftshift(fft)
    mag  = torch.log1p(torch.abs(fft)).float()
    mag_np = mag.detach().cpu().numpy()
    mag_np = (mag_np - mag_np.min()) / (mag_np.max() - mag_np.min() + 1e-8)
    return mag_np


# =============================================================================
# LOAD IMAGE
# =============================================================================

print("\n[Load] Loading bacha.png...")

img_candidates = [
    os.path.join(AFLB_DIR, 'bacha.png'),
    os.path.join(ROOT, 'bacha.png'),
]
img_path = next((c for c in img_candidates if os.path.exists(c)), None)
if img_path is None:
    raise FileNotFoundError("bacha.png not found in evoIR_aflb/ or project root")

print(f"  Found at: {img_path}")
bgr = cv2.imread(img_path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = rgb.shape[:2]
print(f"  Image size: {orig_w}x{orig_h}")

save_image(rgb, '00_original_bacha.png', 'Original bacha.png')

img_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0) \
               .permute(2, 0, 1).unsqueeze(0).to(device)

# =============================================================================
# OUTPUT 1 -- POWER SPECTRUM
# =============================================================================

print("\n[1/9] Power Spectrum (FFT of input)...")
pspec = power_spectrum(img_tensor)
pspec_u8 = (pspec * 255).astype(np.uint8)
pspec_color = cv2.applyColorMap(pspec_u8, cv2.COLORMAP_INFERNO)
pspec_rgb   = cv2.cvtColor(pspec_color, cv2.COLOR_BGR2RGB)
save_image(pspec_rgb, '01_power_spectrum.png',
           'Power Spectrum (log magnitude FFT)')

# =============================================================================
# OUTPUTS 2 & 3 -- AFLB LOW + HIGH FREQUENCY MAPS
# =============================================================================

print("\n[2-3/9] AFLB Frequency Maps (Low + High)...")
try:
    from aflb_fixed import AFLBFixed

    aflb = AFLBFixed().to(device).eval()
    print("  AFLBFixed loaded (random init -- no pretrained weights needed)")

    with torch.no_grad():
        # fft() takes (B, 3, H, W) input, internally does conv1 -> (B, 48, H, W)
        high_feat, low_feat = aflb.fft(img_tensor)
        # forward_features gives the fused output
        freq_feat = aflb.forward_features(img_tensor)

    # Low frequency map
    low_heatmap = feat_to_heatmap(low_feat, orig_h, orig_w)
    low_overlay = overlay(rgb, low_heatmap, alpha=0.45)
    save_image(low_overlay, '02_aflb_low_frequency.png',
               'AFLB Low Frequency (structure/smooth)')

    # High frequency map
    high_heatmap = feat_to_heatmap(high_feat, orig_h, orig_w)
    high_overlay = overlay(rgb, high_heatmap, alpha=0.45)
    save_image(high_overlay, '03_aflb_high_frequency.png',
               'AFLB High Frequency (edges/detail)')

    # Combined frequency feature
    freq_heatmap = feat_to_heatmap(freq_feat, orig_h, orig_w)
    save_image(freq_heatmap, '03b_aflb_combined_freq.png',
               'AFLB Combined Frequency Features')

except Exception as e:
    print(f"  AFLB failed: {e}")
    import traceback; traceback.print_exc()
    # Fallback: compute directly without AFLB module
    print("  Using direct FFT fallback...")
    gray = img_tensor.mean(dim=1, keepdim=True)
    fft  = torch.fft.fft2(gray.squeeze(), norm='forward')
    fft  = torch.fft.fftshift(fft)
    H, W = fft.shape
    mask_low  = torch.zeros(H, W, device=device)
    cy, cx    = H // 2, W // 2
    r = max(1, min(H, W) // 8)
    mask_low[cy-r:cy+r, cx-r:cx+r] = 1

    def _fft_vis(fft_masked):
        sig = torch.fft.ifft2(torch.fft.ifftshift(fft_masked)).real
        a = sig.detach().cpu().float().numpy()
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        u = (a * 255).astype(np.uint8)
        c = cv2.applyColorMap(u, cv2.COLORMAP_JET)
        r2 = cv2.resize(c, (orig_w, orig_h))
        return cv2.cvtColor(r2, cv2.COLOR_BGR2RGB)

    save_image(overlay(rgb, _fft_vis(fft * mask_low)),
               '02_aflb_low_frequency.png', 'Low Freq (FFT fallback)')
    save_image(overlay(rgb, _fft_vis(fft * (1 - mask_low))),
               '03_aflb_high_frequency.png', 'High Freq (FFT fallback)')

# =============================================================================
# OUTPUT 4 -- DDER MoE ROUTER HEATMAP
# =============================================================================

print("\n[4/9] DDER MoE Router Feature Map...")
dder_model = None
try:
    from dder_fixed import DDERFixedModule, load_dder_checkpoint

    dder_model = DDERFixedModule(num_experts=3).to(device).eval()

    dder_weights = os.path.join(DDER_DIR, 'dder.pt')
    if os.path.exists(dder_weights):
        load_dder_checkpoint(dder_model, dder_weights, verbose=True)
    else:
        print(f"  WARNING: dder.pt not found at {dder_weights}")

    de_cls = torch.zeros(1, 6, device=device)
    de_cls[0, 1] = 1.0  # Haze

    # Hook the MoE router output
    captured = {}
    def hook_fn(module, inp, out):
        feat = out[0] if isinstance(out, tuple) else out
        captured['feat'] = feat.detach().cpu()
    hook = dder_model.moe_router.register_forward_hook(hook_fn)

    with torch.no_grad():
        dder_feat, r_loss = dder_model.forward_features(img_tensor, de_cls)
    hook.remove()

    feat_for_vis = captured.get('feat', dder_feat.cpu())
    dder_heatmap = feat_to_heatmap(feat_for_vis, orig_h, orig_w)
    dder_overlay = overlay(rgb, dder_heatmap, alpha=0.5)

    panel = np.hstack([rgb, dder_heatmap, dder_overlay])
    save_image(panel, '04_dder_moe_output.png',
               f'DDER MoE Router | r_loss={r_loss.item():.4f}')
    save_image(dder_heatmap, '04b_dder_heatmap_only.png', 'DDER MoE Heatmap')

except Exception as e:
    print(f"  DDER failed: {e}")
    import traceback; traceback.print_exc()
    placeholder = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'DDER failed', (10, orig_h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
    save_image(placeholder, '04_dder_moe_output.png', 'DDER (FAILED)')

# =============================================================================
# OUTPUT 5 -- net/M2Restore.py STANDALONE (skip if daclip heavy)
# =============================================================================

print("\n[5/9] net/M2Restore.py standalone output...")
print("  SKIPPED -- standalone M2Restore requires full daclip model load")
print("  (uses ~2GB VRAM for ViT + tokenizer). Use net/M2Restore.py directly.")
placeholder = rgb.copy()
cv2.putText(placeholder, 'M2Restore standalone: skipped (heavy model)',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
save_image(placeholder, '05_m2restore_standalone.png',
           'M2Restore Standalone (skipped -- requires daclip weights)')

# =============================================================================
# OUTPUTS 6 & 7 -- SAM BOUNDARY + MiDaS DEPTH
# =============================================================================

print("\n[6/9] SAM-S Boundary Map...")
# SAM weights likely not available -- use Canny edge proxy
gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
edges   = cv2.Canny(gray_u8, 50, 150)
kernel  = np.ones((3, 3), np.uint8)
edges_d = cv2.dilate(edges, kernel, iterations=1)
edge_color = cv2.applyColorMap(edges_d, cv2.COLORMAP_BONE)
edge_rgb   = cv2.cvtColor(edge_color, cv2.COLOR_BGR2RGB)
edge_over  = overlay(rgb, edge_rgb, 0.4)
save_image(edge_over, '06_sam_boundary.png',
           'SAM Boundary Proxy (Canny edges)')
save_image(edges_d, '06b_edges_clean.png', 'Edge Map (Canny)')

print("\n[7/9] MiDaS Depth Map...")
try:
    # Use source='local' since already cached, avoids trust_repo prompt
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small',
                            pretrained=True, source='local')
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms',
                                       source='local')
    transform = midas_transforms.small_transform
    midas = midas.to(device).eval()
    midas_input = transform(rgb).to(device)

    with torch.no_grad():
        depth = midas(midas_input)
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode='bicubic', align_corners=False
        ).squeeze()

    depth_np = depth.detach().cpu().float().numpy()
    depth_np = (depth_np - depth_np.min()) / \
               (depth_np.max() - depth_np.min() + 1e-8)
    depth_u8    = (depth_np * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    depth_rgb   = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    depth_over  = overlay(rgb, depth_rgb, 0.5)

    save_image(depth_over, '07_midas_depth.png',
               'MiDaS Depth Map (dark=near, bright=far)')
    save_image(depth_rgb, '07b_midas_depth_clean.png',
               'MiDaS Depth (PLASMA)')
    print("  MiDaS: SUCCESS")

except Exception as e:
    print(f"  MiDaS failed ({type(e).__name__}: {e})")
    print("  Using Gaussian blur as depth proxy...")
    gray_f  = gray_u8.astype(np.float32)
    blur    = cv2.GaussianBlur(gray_f, (21, 21), 0)
    d_proxy = (blur - blur.min()) / (blur.max() - blur.min() + 1e-8)
    d_u8    = (d_proxy * 255).astype(np.uint8)
    d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_PLASMA)
    d_rgb   = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)
    save_image(overlay(rgb, d_rgb, 0.5),
               '07_midas_depth.png',
               'Depth Proxy (Gaussian -- MiDaS unavailable)')

# =============================================================================
# OUTPUT 8 -- FUSION PIPELINE (mcdb_input)
# =============================================================================

print("\n[8/9] Fusion Pipeline Output (mcdb_input)...")
mcdb_input = None
try:
    from model import HocVidModel

    hocvid = HocVidModel().to(device).eval()

    de_cls = torch.zeros(1, 6, device=device)
    de_cls[0, 1] = 1.0  # Haze

    sam_dummy   = torch.zeros(1, 1, orig_h, orig_w, device=device)
    midas_dummy = torch.zeros(1, 1, orig_h, orig_w, device=device)

    with torch.no_grad():
        mcdb_input, router_loss = hocvid(
            img_tensor,
            de_cls=de_cls,
            sam_boundary=sam_dummy,
            midas_depth=midas_dummy
        )

    fusion_heatmap = feat_to_heatmap(mcdb_input, orig_h, orig_w)
    fusion_overlay = overlay(rgb, fusion_heatmap, 0.5)

    panel = np.hstack([rgb, fusion_heatmap, fusion_overlay])
    save_image(panel, '08_fusion_combined_output.png',
               f'Fusion (mcdb_input) | router_loss={router_loss.item():.4f}')
    save_image(fusion_heatmap, '08b_fusion_heatmap.png',
               'Fusion Heatmap (512ch)')

    print(f"  Fusion output: {mcdb_input.shape}, "
          f"router_loss={router_loss.item():.6f}")

except Exception as e:
    print(f"  Fusion failed: {e}")
    import traceback; traceback.print_exc()
    placeholder = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Fusion failed',
                (10, orig_h//2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 100, 255), 1)
    save_image(placeholder, '08_fusion_combined_output.png', 'Fusion (FAILED)')

# =============================================================================
# OUTPUT 9 -- M2RestoreDecoder OUTPUT (untrained)
# =============================================================================

print("\n[9/9] M2RestoreDecoder output (untrained)...")
try:
    from m2restore_decoder import M2RestoreDecoder

    decoder = M2RestoreDecoder(in_channels=512).to(device).eval()

    if mcdb_input is not None:
        fusion_feat = mcdb_input
    else:
        print("  Using random mcdb_input (fusion failed)")
        fusion_feat = torch.randn(1, 512, orig_h // 4,
                                   orig_w // 4, device=device) * 0.01

    with torch.no_grad():
        residual = decoder(fusion_feat)

        if residual.shape[-2:] != img_tensor.shape[-2:]:
            residual = F.interpolate(
                residual,
                size=(orig_h, orig_w),
                mode='bilinear', align_corners=False
            )

        restored = torch.clamp(img_tensor + residual, 0, 1)

    restored_np = tensor_to_np(restored)
    residual_np = tensor_to_np(
        torch.clamp(residual * 5 + 0.5, 0, 1)  # amplify for visibility
    )

    diff = np.abs(restored_np.astype(np.float32) -
                  rgb.astype(np.float32))
    diff_u8 = np.clip(diff * 5, 0, 255).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff_u8, cv2.COLOR_RGB2GRAY)
    diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_HOT)
    diff_rgb   = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

    panel = np.hstack([rgb, restored_np, residual_np, diff_rgb])
    save_image(panel, '09_m2restore_decoder_output.png',
               'Decoder (untrained): input | restored | residual(x5) | diff(x5)')
    save_image(restored_np, '09b_final_restored.png',
               'Final Restored (untrained -- ~identical to input)')

    max_diff = np.abs(restored_np.astype(float) - rgb.astype(float)).max()
    print(f"  Residual max: {residual.abs().max().item():.6f}")
    print(f"  Output vs input max diff: {max_diff:.4f}")
    print(f"  NOTE: ~identical because bridge_proj is zero-init (untrained)")

except Exception as e:
    print(f"  M2RestoreDecoder failed: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# MASTER COMPARISON GRID
# =============================================================================

print("\n[Grid] Creating master comparison grid...")
try:
    images_for_grid = [
        ('00_original_bacha.png',                'Original'),
        ('01_power_spectrum.png',                'Power Spectrum'),
        ('02_aflb_low_frequency.png',            'Low Freq (AFLB)'),
        ('03_aflb_high_frequency.png',           'High Freq (AFLB)'),
        ('04_dder_moe_output.png',               'DDER MoE Output'),
        ('05_m2restore_standalone.png',          'M2Restore Standalone'),
        ('06_sam_boundary.png',                  'SAM Boundary'),
        ('07_midas_depth.png',                   'MiDaS Depth'),
        ('08b_fusion_heatmap.png',               'Fusion (mcdb_input)'),
        ('09b_final_restored.png',               'Final (untrained)'),
    ]

    loaded = []
    for fname, label in images_for_grid:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            im = cv2.imread(fpath)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # Crop title bar if present
            if im.shape[0] > 40 and np.mean(im[:36]) < 50:
                im = im[36:]
            # If multi-panel, take first square
            if im.shape[1] > im.shape[0] * 1.8:
                im = im[:, :im.shape[0], :]
            im = cv2.resize(im, (256, 256))
            loaded.append((im, label))
        else:
            blank = np.zeros((256, 256, 3), np.uint8)
            cv2.putText(blank, label, (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            loaded.append((blank, label))

    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle('HoCVid Full Pipeline -- bacha.png',
                 fontsize=16, color='white', fontweight='bold', y=1.01)

    for ax, (im, label) in zip(axes.flatten(), loaded):
        ax.imshow(im)
        ax.set_title(label, fontsize=10, color='white', pad=4)
        ax.axis('off')

    plt.tight_layout(pad=0.5)
    grid_path = os.path.join(OUT_DIR, '00_MASTER_GRID.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    print(f"  [Saved] 00_MASTER_GRID.png")

except Exception as e:
    print(f"  Master grid failed: {e}")
    import traceback; traceback.print_exc()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*60)
print("PIPELINE TEST COMPLETE")
print("="*60)
print(f"Output folder: {OUT_DIR}")
print("\nSaved files:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size_kb = os.path.getsize(fpath) // 1024
    print(f"  {f:50s} {size_kb:5d} KB")
print("\nNOTE: 09_m2restore_decoder_output.png shows ~identical")
print("      input/output because the decoder is not yet trained")
print("      (bridge_proj = zero-init). Train on AllWeather to")
print("      see actual restoration.")
print("="*60)
