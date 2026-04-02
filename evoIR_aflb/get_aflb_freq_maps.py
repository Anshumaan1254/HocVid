import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

# Import model architecture
from run_pretrained import AdaIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_to_multiple(img, multiple=16):
    import torch.nn.functional as F
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w

def main():
    print("Loading model...")
    model = AdaIR().to(device)
    
    ckpt_path = os.path.join('pretrained', 'model_landsat.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        cleaned_sd = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_sd, strict=False)
        print("Pretrained weights loaded.")
    else:
        print("No pretrained weights found, using random weights!")
        
    model.eval()

    # Hook to get low and high frequency features
    extracted_features = {}
    def hook_fn(module, input, output):
        low, high = output
        extracted_features['low'] = low.detach().cpu()
        extracted_features['high'] = high.detach().cpu()

    # Register hook to the first Frequency Division (FD) module
    # In run_pretrained.py, the model has encoder_level1_fft = nn.Sequential(*[FFTransformerBlock(dim) for _ in range(num_fft_blocks[0])])
    # The FFTransformerBlock contains prompt_block = FMMPreWork(...)
    # FMMPreWork contains fd = FD(...)
    handle = model.encoder_level1_fft[0].prompt_block.fd.register_forward_hook(hook_fn)

    # Load image
    print("Processing bacha.png...")
    img = Image.open('bacha.png').convert('RGB')
    orig_w, orig_h = img.size
    max_dim = 512
    if max(orig_w, orig_h) > max_dim:
        ratio = max_dim / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        new_w = (new_w // 16) * 16
        new_h = (new_h // 16) * 16
        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"Resized image to {new_w}x{new_h}")
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    img_padded, _, _ = pad_to_multiple(img_tensor, 16)
    
    # Run inference
    with torch.no_grad():
        _ = model(img_padded)
        
    handle.remove()

    if 'low' in extracted_features and 'high' in extracted_features:
        # Per-channel normalization before visualization
        def norm_feat(t):
            f = t[0].mean(dim=0).numpy()
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            return f
        low_feat = norm_feat(extracted_features['low'])
        high_feat = norm_feat(extracted_features['high'])
        
        # also compute global fft freq map for comparison
        gray_tensor = TF.to_tensor(img.convert('L')).unsqueeze(0)
        fft = torch.fft.fftshift(torch.fft.fft2(gray_tensor))
        global_freq_map = torch.log(1 + torch.abs(fft)).squeeze().numpy()

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title('Original Image Power Spectrum')
        plt.imshow(global_freq_map, cmap='magma')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('AFLB Low Frequency Features')
        plt.imshow(low_feat, cmap='viridis')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('AFLB High Frequency Features')
        plt.imshow(high_feat, cmap='viridis')
        plt.axis('off')
        
        # Save combined panel
        import os as _os
        out_dir = _os.path.join(_os.path.dirname(__file__), '..', 'test_output')
        _os.makedirs(out_dir, exist_ok=True)

        out_path = _os.path.join(out_dir, 'bacha_AFLB_freq_maps.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved combined: {out_path}")

        # Save individual images
        # Power Spectrum
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(global_freq_map, cmap='magma')
        ax.set_title('Power Spectrum', fontsize=14)
        ax.axis('off')
        ps_path = _os.path.join(out_dir, 'aflb_power_spectrum_bacha.png')
        plt.savefig(ps_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {ps_path}")

        # Low Frequency
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(low_feat, cmap='viridis')
        ax.set_title('AFLB Low Frequency Features', fontsize=14)
        ax.axis('off')
        lf_path = _os.path.join(out_dir, 'aflb_low_freq_bacha.png')
        plt.savefig(lf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {lf_path}")

        # High Frequency
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(high_feat, cmap='viridis')
        ax.set_title('AFLB High Frequency Features', fontsize=14)
        ax.axis('off')
        hf_path = _os.path.join(out_dir, 'aflb_high_freq_bacha.png')
        plt.savefig(hf_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {hf_path}")
    else:
        print("Failed to extract features.")

if __name__ == "__main__":
    main()
