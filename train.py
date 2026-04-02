import argparse
import torch
from model import HocVidModel

def main():
    parser = argparse.ArgumentParser(description="Train HocVid")
    parser.add_argument("--dder_weights", type=str, default="DDER/dder.pt",
                        help="Path to DDER pretrained weights (dder.pt)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    
    print(f"====================================")
    print(f"Initializing HocVid model with AFLB + DDER...")
    model = HocVidModel(dder_weights_path=args.dder_weights)
    
    # Verification: All DDER params FROZEN in all training stages
    frozen_dder_params = sum(1 for p in model.dder.parameters() if not p.requires_grad)
    total_dder_params = sum(1 for p in model.dder.parameters())
    
    print(f"DDER Parameters: {frozen_dder_params} frozen out of {total_dder_params} total.")
    if frozen_dder_params != total_dder_params:
        print("WARNING: Not all DDER parameters are frozen!")
    else:
        print("SUCCESS: All DDER parameters frozen.")
    
    # Verification: All AFLB params FROZEN
    frozen_aflb_params = sum(1 for p in model.aflb.parameters() if not p.requires_grad)
    total_aflb_params = sum(1 for p in model.aflb.parameters())
    print(f"AFLB Parameters: {frozen_aflb_params} frozen out of {total_aflb_params} total.")
    
    # Trainable params (adapter only)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")
        
    print(f"====================================")
    print("Simulating a training step forward pass...")
    dummy_input = torch.randn(args.batch_size, 3, 224, 224) 
    # Create a dummy one-hot de_cls for smoke test (B, 6) — DA-CLIP 6-class system
    import torch.nn.functional as F_func
    dummy_labels = torch.zeros(args.batch_size, dtype=torch.long)  # class 0
    de_cls = F_func.one_hot(dummy_labels, num_classes=6).float()
    
    model.train()
    out, r_loss = model(dummy_input, de_cls=de_cls)
    print(f"Model output shape: {out.shape}")
    print(f"Router loss: {r_loss.item():.6f}")
    print(f"====================================")
    
if __name__ == '__main__':
    main()
