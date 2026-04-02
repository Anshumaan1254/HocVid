"""
extract_dder_feats.py -- DDER MoE Router Feature Map Visualization
====================================================================

Hooks the MoE router OUTPUT (the actual degradation-aware feature map)
and visualizes WHERE the experts activate on the image.

Uses specific de_cls classes per image for informed routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Add local dirs
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from dder_fixed import DDERFixedModule, load_dder_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_resize(img_path, max_dim=512):
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    if max(orig_w, orig_h) > max_dim:
        ratio = max_dim / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img_tensor = TF.to_tensor(img).unsqueeze(0)
    _, _, h, w = img_tensor.shape
    multiple = 16
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

    return img_tensor.to(device), orig_w, orig_h, img


def extract_and_save_dder(img_path, tag, degradation_idx=3):
    """
    Extract and visualize MoE router feature map.

    Args:
        img_path:         Path to input image
        tag:              Name tag for output file
        degradation_idx:  DA-CLIP 6-class degradation index
                            0=motion_blur, 1=haze, 2=snow,
                            3=rain, 4=raindrop, 5=noise
    """
    print(f"Processing {tag} (degradation class={degradation_idx})...")
    img_tensor, orig_w, orig_h, img_pil = load_and_resize(img_path)

    model = DDERFixedModule(num_experts=3).to(device)

    dder_pt = os.path.join(_project_dir, 'DDER', 'dder.pt')
    load_dder_checkpoint(model, dder_pt, verbose=True)

    model.eval()

    # -- Hook the MoE router OUTPUT (not proj_refinement) --
    captured = {}

    def hook_fn(module, inp, output):
        # moe_router returns (feat_map, r_loss)
        feat_map = output[0] if isinstance(output, tuple) else output
        captured['feat'] = feat_map.detach().cpu()

    hook_handle = model.moe_router.register_forward_hook(hook_fn)

    with torch.no_grad():
        B = img_tensor.shape[0]
        # Specific degradation class, not uniform
        de_cls = torch.zeros(B, 6, device=device)
        de_cls[0, degradation_idx] = 1.0
        # Run forward to trigger hook
        model.forward_features(img_tensor, de_cls=de_cls)

    hook_handle.remove()

    if 'feat' not in captured:
        print(f"  ERROR: Hook did not capture MoE output!")
        return

    feat = captured['feat']  # (1, 512, h_p, w_p)
    print(f"  MoE router output: {feat.shape}")

    # -- Heatmap: abs mean across 512 channels --
    heat = feat.abs().mean(dim=1).squeeze()  # (h_p, w_p)

    # Normalize to [0, 1]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Upsample to original image size with smooth bilinear interpolation
    heat_tensor = heat.unsqueeze(0).unsqueeze(0)  # (1, 1, h_p, w_p)
    heat_up = F.interpolate(
        heat_tensor,
        size=(img_pil.height, img_pil.width),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()

    # Apply JET colormap via OpenCV
    heat_uint8 = (heat_up * 255).astype(np.uint8)
    colormap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2RGB)

    # Overlay on original
    orig_array = np.array(img_pil).astype(np.float32)
    colormap_float = colormap_rgb.astype(np.float32)
    overlay = cv2.addWeighted(orig_array, 0.5, colormap_float, 0.5, 0).astype(np.uint8)

    # -- Save side-by-side comparison --
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].set_title(f'Original {tag}', fontsize=14)
    axes[0].imshow(img_pil)
    axes[0].axis('off')

    axes[1].set_title(f'MoE Router Heatmap', fontsize=14)
    axes[1].imshow(heat_up, cmap='jet', vmin=0, vmax=1)
    axes[1].axis('off')

    axes[2].set_title(f'MoE Overlay (class={degradation_idx})', fontsize=14)
    axes[2].imshow(overlay)
    axes[2].axis('off')

    out_path = os.path.join(_project_dir, f'{tag}_dder_output.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")

    # Print heatmap stats
    print(f"  Heatmap stats: min={heat.min():.4f}, max={heat.max():.4f}, "
          f"mean={heat.mean():.4f}, std={heat.std():.4f}")


if __name__ == "__main__":
    horse_path = os.path.join(_project_dir, 'evoIR_aflb', 'Horse1.png')
    bacha_path = os.path.join(_project_dir, 'evoIR_aflb', 'bacha.png')

    # Horse1: outdoor scene with motion blur / general degradation
    if os.path.exists(horse_path):
        extract_and_save_dder(horse_path, "Horse1", degradation_idx=0)

    # bacha: portrait with haze/overexposure
    if os.path.exists(bacha_path):
        extract_and_save_dder(bacha_path, "bacha", degradation_idx=3)
