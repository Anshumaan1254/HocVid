import os, sys, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# Suppress AFLB verbose output
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
    print(f"Device: {device}")

    input_path = os.path.join(os.path.dirname(__file__), 'ddertest.jpeg')
    dder_weights = os.path.join(os.path.dirname(__file__), 'DDER', 'dder.pt')

    print("Initializing model...")
    model = HocVidModel(dder_weights_path=dder_weights)
    model = model.to(device)
    model.eval()

    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Input: {orig_w}x{orig_h}")

    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    img_padded, h_orig, w_orig = pad_to_multiple(img_tensor, 16)

    # ============================================================
    # 1) AFLB OUTPUT — Run AdaIR alone
    # ============================================================
    print("Stage 1: Running AFLB...")
    with torch.no_grad():
        aflb_restored = model.aflb(img_padded)
    
    aflb_out = aflb_restored[:, :, :h_orig, :w_orig].clamp(0, 1)
    aflb_pil = TF.to_pil_image(aflb_out.squeeze(0).cpu())
    if aflb_pil.size != (orig_w, orig_h):
        aflb_pil = aflb_pil.resize((orig_w, orig_h), Image.LANCZOS)
    aflb_pil.save('aflb_output.png', quality=95)
    print(f"  -> aflb_output.png (AFLB restored image)")

    # ============================================================
    # 2) DDER OUTPUT — Run DDER standalone on raw image (as on Colab)
    # ============================================================
    print("Stage 2: Running DDER on raw image (as trained on Colab)...")
    with torch.no_grad():
        input_224 = F.interpolate(img_padded, size=(224, 224), mode='bilinear', align_corners=False)
        dder_raw = model.dder(input_224)  # [B, 3, 224, 224]
    
    # Save at 224x224 (native DDER output)
    dder_224 = dder_raw.clamp(0, 1)
    dder_224_pil = TF.to_pil_image(dder_224.squeeze(0).cpu())
    dder_224_pil.save('dder_output_224.png', quality=95)
    print(f"  -> dder_output_224.png (DDER native 224x224 output)")
    
    # Also upscale to original size
    dder_upscaled = F.interpolate(dder_raw, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
    dder_out = dder_upscaled.clamp(0, 1)
    dder_pil = TF.to_pil_image(dder_out.squeeze(0).cpu())
    if dder_pil.size != (orig_w, orig_h):
        dder_pil = dder_pil.resize((orig_w, orig_h), Image.LANCZOS)
    dder_pil.save('dder_output.png', quality=95)
    print(f"  -> dder_output.png (DDER upscaled to original size)")

    # ============================================================
    # 3) DDER FEATURE MAP — Heatmap visualization (like Colab)
    # ============================================================
    print("Generating DDER feature heatmap...")
    with torch.no_grad():
        x_tokens = model.dder.visual_backbone.conv1(input_224)
        h_p, w_p = x_tokens.shape[-2:]
        x_tokens = x_tokens.reshape(1, 768, -1).permute(0, 2, 1)
        
        pos = model.dder.visual_backbone.positional_embedding
        cls_pos = pos[:1, :]
        sp = pos[1:, :].reshape(1, 7, 7, 768).permute(0, 3, 1, 2)
        sp = F.interpolate(sp, size=(h_p, w_p), mode='bicubic', align_corners=False)
        sp = sp.permute(0, 2, 3, 1).reshape(-1, 768)
        cur_pos = torch.cat([cls_pos, sp], dim=0).to(device)
        
        cls_tok = model.dder.visual_backbone.class_embedding.to(device, dtype=input_224.dtype)
        x_tokens = torch.cat([cls_tok.expand(1, 1, -1), x_tokens], dim=1)
        x_tokens = x_tokens + cur_pos.to(input_224.dtype)
        x_tokens = model.dder.visual_backbone.ln_pre(x_tokens)
        x_tokens = model.dder.visual_backbone.transformer(x_tokens)
        
        # Spatial features (skip CLS token)
        spatial = x_tokens[:, 1:, :].permute(0, 2, 1).reshape(1, 768, h_p, w_p)
        
        # Create heatmap: average over channels -> normalize -> colormap
        feat_avg = spatial.mean(dim=1, keepdim=True)  # [1, 1, h_p, w_p]
        feat_up = F.interpolate(feat_avg, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        feat_np = feat_up.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-8)
    
    import numpy as np
    # Apply jet colormap manually
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    heatmap = cm.jet(feat_np)[:, :, :3]  # [H, W, 3] in 0-1
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay on original image
    orig_np = np.array(img.resize((orig_w, orig_h)))
    overlay = (0.4 * orig_np + 0.6 * heatmap).astype(np.uint8)
    
    Image.fromarray(heatmap).save('dder_heatmap.png', quality=95)
    Image.fromarray(overlay).save('dder_heatmap_overlay.png', quality=95)
    print(f"  -> dder_heatmap.png (raw feature heatmap)")
    print(f"  -> dder_heatmap_overlay.png (heatmap overlaid on input)")

    # ============================================================
    # 4) COMBINED (full pipeline)
    # ============================================================
    print("Running full pipeline (AFLB + DDER + Fusion)...")
    with torch.no_grad():
        combined = model(img_padded)
    
    combined_out = combined[:, :, :h_orig, :w_orig].clamp(0, 1)
    combined_pil = TF.to_pil_image(combined_out.squeeze(0).cpu())
    if combined_pil.size != (orig_w, orig_h):
        combined_pil = combined_pil.resize((orig_w, orig_h), Image.LANCZOS)
    combined_pil.save('dderout1.png', quality=95)
    print(f"  -> dderout1.png (combined pipeline output)")

    print("\nDone! All outputs saved.")

if __name__ == '__main__':
    main()
