import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

print("Loading model and dependencies (this might take a few seconds)...")
# Suppress the verbose output from `pretrained_model.py` execution block
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w', encoding='utf-8')
from model import HocVidModel
sys.stdout = old_stdout

def pad_to_multiple(img, multiple=16):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dder_weights = os.path.join(os.path.dirname(__file__), 'DDER', 'dder.pt')
    
    # 1. Initialize model
    model = HocVidModel(dder_weights_path=dder_weights)
    model = model.to(device)
    model.eval()
    print("Model loaded with EvoIR AFLB + DDER pipeline!")

    # 2. Load bacha.png
    input_path = os.path.join(os.path.dirname(__file__), 'evoIR_aflb', 'bacha.png')
    output_path = os.path.join(os.path.dirname(__file__), 'newbacha.png')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size

    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    img_padded, h_orig, w_orig = pad_to_multiple(img_tensor, 16)

    # 3. Forward Pass — full AFLB → Adapter → DDER pipeline
    print("Executing full pipeline inference (AFLB -> Adapter -> DDER)...")
    with torch.no_grad():
        output = model(img_padded)  # Returns final [B, 3, H, W] image

    # 4. Save Image
    print("Saving the output image...")
    restored = output[:, :, :h_orig, :w_orig].clamp(0, 1)
    out_pil = TF.to_pil_image(restored.squeeze(0).cpu())
    if out_pil.size != (orig_w, orig_h):
        out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)
        
    out_pil.save(output_path, quality=95)
    print(f"Successfully saved output to {output_path}!")

if __name__ == '__main__':
    main()
