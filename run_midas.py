import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DPTImageProcessor, DPTForDepthEstimation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load image -- search common locations
img_path = 'bacha.png'
if not os.path.exists(img_path):
    for candidate in ['evoIR_aflb/bacha.png', 'input/bacha.png']:
        if os.path.exists(candidate):
            img_path = candidate
            break
    else:
        print(f"Error: bacha.png not found in any expected location.")
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
print("MiDaS model loaded successfully!")

# 3. Process image and run inference
print("Running MiDaS forward pass...")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# 4. Interpolate to original size and prepare for visualization
print("Processing output...")
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# Convert to numpy array and normalize with full min-max range
output = prediction.squeeze().cpu().numpy()
output_norm = (output - output.min()) / (output.max() - output.min() + 1e-8)
formatted = (output_norm * 255).astype("uint8")

# 5. Plot and save
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
im = plt.imshow(formatted, cmap='inferno')
plt.colorbar(im, label='Depth (relative)')
plt.title('MiDaS Depth Map')
plt.axis('off')

plt.tight_layout()
out_path = 'midas_depth_bacha.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {out_path}")
