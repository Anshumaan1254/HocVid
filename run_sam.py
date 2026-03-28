import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load SAM2 Model as requested
from transformers import Sam2Model, Sam2Processor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load image
img_path = 'Midasinput.jpeg'
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    sys.exit(1)

image = Image.open(img_path).convert('RGB')
print(f"Loaded image: {image.size}")

# 2. Load SAM2
MODEL_NAME = "facebook/sam2-hiera-large"

print(f"Loading processor for {MODEL_NAME}...")
try:
    processor = Sam2Processor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading processor: {e}")
    sys.exit(1)

print(f"Loading model {MODEL_NAME}...")
model = Sam2Model.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print("✅ SAM2 model loaded successfully!")

# 3. Process image and run inference
print("Running SAM2 forward pass...")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    
# For an unprompted model, the pure structural boundaries and object relationships
# are contained within the dense image embeddings.
embeds = outputs.image_embeddings
if isinstance(embeds, (list, tuple)):
    embeds = embeds[-1]  # Take the final embedding layer

print(f"Extracting SAM2 dense embeddings mapping. Shape: {embeds.shape}")

# 4. PCA for Visualization
# The feature map is highly dimensional (typically 256 channels). 
# We use PCA to compress these channels down to 3 (RGB) to beautifully visualize 
# the structural boundaries SAM has learned.
embeds_squeezed = embeds.squeeze(0).cpu()  # (C, H, W)
C, H, W = embeds_squeezed.shape
flat_embeds = embeds_squeezed.view(C, -1).permute(1, 0)  # (H*W, C)

# Center the data
flat_embeds = flat_embeds - flat_embeds.mean(dim=0)

# Compute top 3 Principal Components
U, S, V = torch.pca_lowrank(flat_embeds, q=3)
pca_features = U[:, :3]  # (H*W, 3)

# Normalize PCs to [0, 1] for RGB visualization
pca_min = pca_features.min(dim=0)[0]
pca_max = pca_features.max(dim=0)[0]
pca_norm = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)

# Reshape back to spatial grid
pca_tensor = pca_norm.view(1, H, W, 3).permute(0, 3, 1, 2)  # (1, 3, H, W)

# Resize to match original image explicitly
pca_resized = torch.nn.functional.interpolate(
    pca_tensor,
    size=(image.size[1], image.size[0]),  # (H, W)
    mode="bicubic",
    align_corners=False
)
pca_img_ready = pca_resized.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 3)

# 5. Plot and save
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(pca_img_ready)
plt.title('SAM2 Latent Feature Space (PCA mapped to RGB)')
plt.axis('off')

plt.tight_layout()
out_path = 'sam_features_midasinput.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {out_path}")

# Copy to conversation artifacts
import shutil
artifact_dir = r"C:\Users\Anshu\.gemini\antigravity\brain\b83df88f-ee52-4609-b66f-6a7d4e6c6575"
os.makedirs(artifact_dir, exist_ok=True)
shutil.copy(out_path, os.path.join(artifact_dir, 'sam_features_midasinput.png'))
shutil.copy(out_path, os.path.join('output', 'sam_output_midasinput.png'))
print("Copied to artifacts dir and output dir.")
