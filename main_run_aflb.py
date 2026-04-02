"""
main_run_aflb.py
================
HoCVid canonical AFLB frequency decomposition script.
Extracts low and high frequency features via FD module hooks,
generates power spectrum + feature visualizations.

Usage:
    python main_run_aflb.py                              # defaults to bacha.png
    python main_run_aflb.py --image Midasinput.jpeg       # any image
    python main_run_aflb.py --image evoIR_aflb/Horse1.png

Output saved to: test_output/
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add evoIR_aflb to path so we can import AdaIR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evoIR_aflb'))
from run_pretrained import AdaIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_to_multiple(img, multiple=16):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w


def run_aflb(image_path, output_dir='test_output'):
    """Run AFLB frequency decomposition on a single image."""

    basename = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[AFLB] Device: {device}")
    print(f"[AFLB] Loading model...")
    model = AdaIR().to(device)

    # Load pretrained weights if available
    ckpt_path = os.path.join(os.path.dirname(__file__), 'evoIR_aflb', 'pretrained', 'model_landsat.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        cleaned_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_sd, strict=False)
        print("[AFLB] Pretrained weights loaded.")
    else:
        print("[AFLB] No pretrained weights found, using random init.")

    model.eval()

    # Hook to extract low/high frequency features from FD module
    extracted_features = {}
    def hook_fn(module, input, output):
        low, high = output
        extracted_features['low'] = low.detach().cpu()
        extracted_features['high'] = high.detach().cpu()

    handle = model.encoder_level1_fft[0].prompt_block.fd.register_forward_hook(hook_fn)

    # Load and preprocess image
    print(f"[AFLB] Processing {image_path}...")
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    max_dim = 512
    if max(orig_w, orig_h) > max_dim:
        ratio = max_dim / max(orig_w, orig_h)
        new_w = (int(orig_w * ratio) // 16) * 16
        new_h = (int(orig_h * ratio) // 16) * 16
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"[AFLB] Resized to {new_w}x{new_h}")

    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    img_padded, _, _ = pad_to_multiple(img_tensor, 16)

    # Run inference (hook captures features)
    with torch.no_grad():
        _ = model(img_padded)

    handle.remove()

    if 'low' not in extracted_features or 'high' not in extracted_features:
        print("[AFLB] ERROR: Failed to extract features.")
        return None, None

    # Normalize features for visualization
    def norm_feat(t):
        f = t[0].mean(dim=0).numpy()
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        return f

    low_feat = norm_feat(extracted_features['low'])
    high_feat = norm_feat(extracted_features['high'])

    # Compute FFT power spectrum
    gray_tensor = TF.to_tensor(img.convert('L')).unsqueeze(0)
    fft = torch.fft.fftshift(torch.fft.fft2(gray_tensor))
    power_spectrum = torch.log(1 + torch.abs(fft)).squeeze().numpy()

    print(f"[AFLB] Generating visualizations...")

    # --- Combined 3-panel ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(power_spectrum, cmap='magma')
    axes[0].set_title('Original Image Power Spectrum', fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(low_feat, cmap='viridis')
    axes[1].set_title('AFLB Low Frequency Features', fontsize=12)
    axes[1].axis('off')
    axes[2].imshow(high_feat, cmap='viridis')
    axes[2].set_title('AFLB High Frequency Features', fontsize=12)
    axes[2].axis('off')
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'aflb_freq_maps_{basename}.png')
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"[AFLB] Saved: {combined_path}")

    # --- Individual: Power Spectrum ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(power_spectrum, cmap='magma')
    ax.set_title('Power Spectrum', fontsize=14)
    ax.axis('off')
    ps_path = os.path.join(output_dir, f'aflb_power_spectrum_{basename}.png')
    plt.savefig(ps_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[AFLB] Saved: {ps_path}")

    # --- Individual: Low Frequency ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(low_feat, cmap='viridis')
    ax.set_title('AFLB Low Frequency Features', fontsize=14)
    ax.axis('off')
    lf_path = os.path.join(output_dir, f'aflb_low_freq_{basename}.png')
    plt.savefig(lf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[AFLB] Saved: {lf_path}")

    # --- Individual: High Frequency ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(high_feat, cmap='viridis')
    ax.set_title('AFLB High Frequency Features', fontsize=14)
    ax.axis('off')
    hf_path = os.path.join(output_dir, f'aflb_high_freq_{basename}.png')
    plt.savefig(hf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[AFLB] Saved: {hf_path}")

    return low_feat, high_feat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFLB frequency decomposition')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    args = parser.parse_args()

    # Default: search for bacha.png
    if args.image is None:
        for candidate in ['bacha.png', 'evoIR_aflb/bacha.png']:
            if os.path.exists(candidate):
                args.image = candidate
                break
        else:
            print("Error: No image specified and bacha.png not found.")
            sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found.")
        sys.exit(1)

    run_aflb(args.image)
    print("[AFLB] DONE")
