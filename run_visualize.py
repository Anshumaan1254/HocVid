import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# Import the top-level model we just created
from model import HocVidModel

def pad_to_multiple(img, multiple=16):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    # Returns padded image, and original height and width
    return img, h, w

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load image
img_path = os.path.join('evoIR_aflb', 'Horse1.png')
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    sys.exit(1)

img_pil = Image.open(img_path).convert('RGB')
orig_w, orig_h = img_pil.size
img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)
img_padded, orig_h, orig_w = pad_to_multiple(img_tensor, 16)

# 2. Init model
print("Initializing model...")
model = HocVidModel().to(device)
model.eval()

# 3. Forward pass
# We leave SAM and MiDaS as None - the model gracefully degradation to zeros for them,
# which tests our SPMAdapter baseline!
print("Running forward pass...")
with torch.no_grad():
    mcdb_input, r_loss = model(img_padded, de_cls=None, sam_boundary=None, midas_depth=None)

print(f"MCDB input shape: {mcdb_input.shape}")

# 4. Process feature map
# mcdb_input is (B, 512, H, W). We take the mean across all 512 channels
# to get a single 2D activation/attention heat map.
feat_map = mcdb_input.squeeze(0).mean(dim=0).cpu().numpy() # (H, W)

# Crop out the padding so it exactly matches the original image dimensions
feat_map = feat_map[:orig_h, :orig_w]

# Normalize to [0, 1] for visualization
feat_min = feat_map.min()
feat_max = feat_map.max()
feat_map_norm = (feat_map - feat_min) / (feat_max - feat_min + 1e-8)

# 5. Overlay on input image using matplotlib
plt.figure(figsize=(15, 6))

# Plot 1: Original Image
plt.subplot(1, 3, 1)
plt.imshow(img_pil)
plt.title('Input Image')
plt.axis('off')

# Plot 2: Just the activation heatmap
plt.subplot(1, 3, 2)
plt.imshow(feat_map_norm, cmap='jet')
plt.title('MCDB Features (Mean of 512 Channels)')
plt.axis('off')

# Plot 3: Blended Overlay
plt.subplot(1, 3, 3)
plt.imshow(img_pil)
plt.imshow(feat_map_norm, cmap='jet', alpha=0.6)  # 60% opacity for heatmap
plt.title('Feature Map Overlaid on Input')
plt.axis('off')

plt.tight_layout()
out_path = 'mcdb_feature_overlay.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {out_path}")

# Copy to conversation artifacts
import shutil
artifact_dir = r"C:\Users\Anshu\.gemini\antigravity\brain\b83df88f-ee52-4609-b66f-6a7d4e6c6575"
os.makedirs(artifact_dir, exist_ok=True)
shutil.copy(out_path, os.path.join(artifact_dir, 'mcdb_feature_overlay.png'))
print(f"Copied to artifacts dir.")
