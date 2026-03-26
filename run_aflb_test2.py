import os, sys, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from model import HocVidModel
from huggingface_hub import hf_hub_download

def pad_to_multiple(img, multiple=16):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

input_path = os.path.join(os.path.dirname(__file__), 'evoIR_aflb', 'Horse1.png')

print("Initializing model...")
model = HocVidModel().to(device)
model.eval()

print("Downloading HF weights...")
pretrained_dir = os.path.join(os.path.dirname(__file__), 'evoIR_aflb', 'pretrained')
os.makedirs(pretrained_dir, exist_ok=True)
ckpt_path = os.path.join(pretrained_dir, 'model_landsat.pth')

if not os.path.exists(ckpt_path):
    print("Downloading model_landsat.pth from leonmakise/EvoIR...")
    ckpt_path = hf_hub_download(
        repo_id='leonmakise/EvoIR',
        filename='model_landsat.pth',
        local_dir=pretrained_dir
    )

print("Loading weights into AFLB...")
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
cleaned_sd = {k.replace('module.', ''): v for k, v in sd.items()}
result = model.aflb.load_state_dict(cleaned_sd, strict=False)
print(f"AFLB Weights loaded! Matched: {len(cleaned_sd) - len(result.unexpected_keys)}")

img = Image.open(input_path).convert('RGB')
orig_w, orig_h = img.size
print(f"Input: {orig_w}x{orig_h}")

img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
img_padded, h_orig, w_orig = pad_to_multiple(img_tensor, 16)

print("Stage 1: Running AFLB and extracting Frequency Map...")
with torch.no_grad():
    aflb_restored = model.aflb(img_padded)
    freq_map = aflb_restored - img_padded

# Normalize the frequency map for visualization (min-max normalization)
freq_map_min = freq_map.min()
freq_map_max = freq_map.max()
freq_map_vis = (freq_map - freq_map_min) / (freq_map_max - freq_map_min + 1e-8)

aflb_out = freq_map_vis[:, :, :h_orig, :w_orig].clamp(0, 1)
aflb_pil = TF.to_pil_image(aflb_out.squeeze(0).cpu())
if aflb_pil.size != (orig_w, orig_h):
    aflb_pil = aflb_pil.resize((orig_w, orig_h), Image.LANCZOS)

output_path = os.path.join(os.path.dirname(__file__), 'aflb_freq_map_horse1.png')
aflb_pil.save(output_path, quality=95)
print(f"  -> {output_path}")

# Copy into artifacts folder
import shutil
artifact_dir = r"c:\Users\Anshu\.gemini\antigravity\brain\8fdb473a-9318-45ae-91fb-be10b75a35b6"
if os.path.exists(artifact_dir):
    shutil.copy(output_path, os.path.join(artifact_dir, 'aflb_freq_map_horse1.png'))

