"""
main_run_sam.py
===============
HoCVid canonical SAM automatic mask segmentation script.
Generates colored mask overlay visualization for any input image.

Usage:
    python main_run_sam.py                          # defaults to bacha.png
    python main_run_sam.py --image path/to/img.png  # any image
    python main_run_sam.py --image evoIR_aflb/Horse1.png

Output saved to: test_output/sam_output_<name>.png
"""

import os
import sys
import argparse
import colorsys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import pipeline


def run_sam(image_path, output_dir='test_output'):
    """Run SAM automatic mask generation on a single image."""

    device = 0 if torch.cuda.is_available() else -1
    print(f"[SAM] Device: {'cuda' if device == 0 else 'cpu'}")

    # Load image
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    basename = os.path.splitext(os.path.basename(image_path))[0]
    print(f"[SAM] Loaded: {image_path} ({w}x{h})")

    # Load SAM automatic mask generation pipeline
    print("[SAM] Loading SAM model...")
    generator = pipeline(
        "mask-generation",
        model="facebook/sam-vit-base",
        device=device,
    )
    print("[SAM] Model loaded.")

    # Generate masks
    print("[SAM] Running automatic mask generation...")
    outputs = generator(image, points_per_batch=64)

    # Extract masks and scores, converting tensors to numpy
    masks = []
    for m in outputs["masks"]:
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        masks.append(np.array(m, dtype=bool))

    scores = []
    for s in outputs["scores"]:
        if isinstance(s, torch.Tensor):
            s = s.item()
        scores.append(float(s))

    print(f"[SAM] Generated {len(masks)} masks")

    # Build colored overlay on original image
    img_arr = np.array(image).astype(np.float32) / 255.0
    overlay = img_arr.copy()
    alpha = 0.45

    # Sort by score (lowest first so highest-confidence draws on top)
    order = np.argsort(scores)

    # Generate distinct colors via golden ratio
    np.random.seed(42)
    for rank, idx in enumerate(order):
        mask = masks[idx]
        # Resize mask to image dims if needed
        if mask.shape != (h, w):
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            mask = np.array(mask_pil) > 127

        hue = (rank * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(
            hue,
            0.65 + np.random.random() * 0.3,
            0.7 + np.random.random() * 0.25
        )
        color = np.array([r, g, b])
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    # Save side-by-side comparison
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title(
        f'SAM Auto-Segmentation ({len(masks)} masks)',
        fontsize=14, fontweight='bold'
    )
    axes[1].axis('off')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'sam_output_{basename}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAM] Saved: {out_path}")

    # Standalone overlay
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(overlay)
    ax2.axis('off')
    standalone = os.path.join(output_dir, f'sam_masks_{basename}.png')
    plt.savefig(standalone, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[SAM] Saved: {standalone}")

    return masks, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM automatic segmentation')
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

    run_sam(args.image)
    print("[SAM] DONE")
