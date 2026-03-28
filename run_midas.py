import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load image
img_path = os.path.join('evoIR_aflb', 'bacha.png')
if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    sys.exit(1)

image = Image.open(img_path).convert('RGB')
print(f"Loaded image: {image.size}")

# 2. Load MiDaS
MODEL_NAME = "Intel/dpt-hybrid-midas"
print(f"Loading processor for {MODEL_NAME}...")
try:
    processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading processor: {e}")
    sys.exit(1)

print(f"Loading model {MODEL_NAME}...")
model = DPTForDepthEstimation.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True)
model = model.to(device)
model.eval()
print("✅ MiDaS model loaded successfully!")

# 3. Process image and run inference
print("Running MiDaS forward pass...")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# 4. Interpolate to original size and prepare for visualization
print("Processing output...")
# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# Convert to numpy array and normalize
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")

# 5. Plot and save
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
# MiDaS convention is often inverted or magma colormap
plt.imshow(formatted, cmap='magma')
plt.title('MiDaS Depth Map')
plt.axis('off')

plt.tight_layout()
out_path = 'midas_depth_bacha.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {out_path}")

# Copy to conversation artifacts
import shutil
artifact_dir = r"C:\Users\Anshu\.gemini\antigravity\brain\b83df88f-ee52-4609-b66f-6a7d4e6c6575"
os.makedirs(artifact_dir, exist_ok=True)
shutil.copy(out_path, os.path.join(artifact_dir, 'midas_depth_bacha.png'))
print("Copied to artifacts dir.")
