"""
train_dder_corrected.py -- Corrected DDER Training Script for Google Colab
==========================================================================

This script contains the corrected AllWeatherDataset and training loop
with all bug fixes applied. Copy this into Colab cells or run standalone.

Fixes applied:
  1. AllWeatherDataset returns (degraded, clean, label) with proper label mapping
  2. Training loop converts label -> one-hot de_cls (B, 6) and passes to model
  3. Uses full DDERModule from dder.py (not the passthrough placeholder)
  4. Loss = L1(output, clean) + 0.01 * MoE_router_balance_loss
  5. Proper AllWeather directory structure validation

DA-CLIP uses 6 universal degradation classes. AllWeather maps as:
  snow=0, raindrop=1, rainHaze=2 (mapped to first 3 of 6 classes)

Usage on Colab:
  1. Upload fixed DDER/, MOE_CLIPbias.py, dder.py to /content/
  2. Download AllWeather dataset fully (10.9GB)
  3. Run this script
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as T

try:
    from pytorch_msssim import ssim as ssim_fn
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False


# ===========================================================
# 1. AllWeather Dataset (CORRECTED)
# ===========================================================
class AllWeatherDataset(Dataset):
    """
    AllWeather dataset for DDER training.
    
    Expected directory structure:
      {root_dir}/{split}/snow/input/    <- degraded images
      {root_dir}/{split}/snow/gt/       <- clean ground truth
      {root_dir}/{split}/raindrop/input/
      {root_dir}/{split}/raindrop/gt/
      {root_dir}/{split}/rainHaze/input/
      {root_dir}/{split}/rainHaze/gt/
    
    Returns: (degraded_tensor, clean_tensor, label_int)
      where label is 0=snow, 1=raindrop, 2=rainHaze
    """
    
    # AllWeather -> DA-CLIP 6-class mapping
    CATEGORIES = {'snow': 0, 'raindrop': 1, 'rainHaze': 2}

    def __init__(self, root_dir, split='train', size=128):
        self.root_dir = root_dir
        self.split = split
        # NOTE: No CLIP normalization -- model works in [0,1] range
        self.transform = T.Compose([
            T.RandomCrop(size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

        self.data_info = []
        for cat, label in self.CATEGORIES.items():
            input_dir = os.path.join(root_dir, split, cat, 'input')
            gt_dir = os.path.join(root_dir, split, cat, 'gt')

            if not os.path.exists(input_dir):
                print(f"  WARNING: {input_dir} not found -- skipping {cat}")
                continue
            
            files = sorted(os.listdir(input_dir))
            for f in files:
                self.data_info.append({
                    'input': os.path.join(input_dir, f),
                    'gt': os.path.join(gt_dir, f),
                    'label': label,
                })
        
        print(f"AllWeatherDataset({split}): {len(self.data_info)} image pairs loaded")
        if len(self.data_info) == 0:
            raise RuntimeError(
                f"No images found! Check that AllWeather is fully downloaded at {root_dir}. "
                f"Expected structure: {root_dir}/{split}/snow/input/*.jpg"
            )

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]
        input_img = Image.open(info['input']).convert('RGB')
        gt_img = Image.open(info['gt']).convert('RGB')

        # Apply same random crop/flip to both images
        state = torch.get_rng_state()
        input_tensor = self.transform(input_img)
        torch.set_rng_state(state)
        gt_tensor = self.transform(gt_img)

        return input_tensor, gt_tensor, info['label']


# ===========================================================
# 2. Training Loop (CORRECTED)
# ===========================================================
def train(
    model,
    train_loader,
    device,
    num_epochs=50,
    lr=2e-4,
    weight_decay=1e-4,
    moe_loss_weight=0.01,
    save_dir='/content/checkpoints/dder/',
    num_degradations=6,
):
    """
    Corrected DDER training loop.
    
    Key fixes vs original notebook:
      - Converts integer labels to one-hot de_cls (B, 6) 
      - Passes de_cls to model forward
      - Uses proper loss: L1 + 0.01 * router_loss
      - model.forward() returns (output, router_loss) in train mode
    """
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (degraded, clean, labels) in enumerate(train_loader):
            degraded = degraded.to(device)
            clean = clean.to(device)
            labels = labels.to(device)
            
            # == KEY FIX: Convert integer labels -> one-hot de_cls (B, 6) ==
            de_cls = F.one_hot(labels, num_classes=num_degradations).float()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # == KEY FIX: Pass de_cls to model ==
                output, router_loss = model(degraded, de_cls=de_cls)
                
                # == KEY FIX: Combined loss with SSIM ==
                l1_loss = F.l1_loss(output, clean)
                if _SSIM_AVAILABLE:
                    ssim_val = ssim_fn(output, clean, data_range=1.0,
                                      size_average=True)
                    img_loss = 0.84 * l1_loss + 0.16 * (1.0 - ssim_val)
                else:
                    img_loss = l1_loss
                total_loss = img_loss + moe_loss_weight * router_loss
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch[{epoch+1}/{num_epochs}] Batch[{batch_idx}] "
                      f"L1={img_loss.item():.4f} MoE={router_loss.item():.4f} "
                      f"Total={total_loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_path = os.path.join(save_dir, f'epoch_{epoch+1:03d}.pt')
        torch.save(ckpt, save_path)
        
        if (epoch + 1) % 5 == 0:
            print(f"Checkpoint saved: {save_path}")


# ===========================================================
# 3. Main Entry Point
# ===========================================================
if __name__ == '__main__':
    import sys
    
    # Add DDER paths
    _dder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DDER')
    sys.path.insert(0, _dder_dir)
    sys.path.insert(0, os.path.join(_dder_dir, 'daclip'))
    
    from DDER.dder import DDERModule
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    
    # Initialize model
    model = DDERModule(num_experts=3).to(device)
    
    # Load pretrained weights
    dder_pt = os.path.join(_dder_dir, 'dder.pt')
    if os.path.exists(dder_pt):
        from dder_fixed import load_dder_checkpoint
        load_dder_checkpoint(model, dder_pt, verbose=True)
    
    # Freeze ViT backbone
    for param in model.visual_backbone.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")
    
    # Dataset (update path for your Colab environment)
    allweather_dir = '/content/allweather/'
    dataset = AllWeatherDataset(allweather_dir, split='train', size=128)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    # Train
    train(model, loader, device, num_epochs=50)
