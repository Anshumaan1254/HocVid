import os
import sys
import torch

def check_dder_weights(dder_pt_path):
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("CHECK 1: DDER weight loading verification")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if not os.path.exists(dder_pt_path):
        print(f"File not found: {dder_pt_path}")
        sys.exit(1)
    
    file_size_mb = os.path.getsize(dder_pt_path) / (1024 * 1024)
    print(f"→ dder.pt size: {file_size_mb:.2f} MB")
        
    weights = torch.load(dder_pt_path, map_location='cpu', weights_only=False)
    if isinstance(weights, dict) and 'model_state_dict' in weights:
        weights = weights['model_state_dict']
    elif isinstance(weights, dict) and 'state_dict' in weights:
        weights = weights['state_dict']
    elif not isinstance(weights, dict):
        print(f"Weights format unrecognized: {type(weights)}")
        weights = {}

    keys = list(weights.keys())
    print(f"→ Total keys in dder.pt: {len(keys)}")
    print(f"→ First 20 keys: {keys[:20]}")
    
    expected = ['moe', 'gate', 'visual', 'router', 'expert', 'proj']
    matched = [k for k in keys if any(e in k.lower() for e in expected)]
    missing = [e for e in expected if not any(e in k.lower() for k in keys)]
    
    print(f"→ MATCHED keys (by concept): {len(matched)}")
    print(f"→ MISSING concepts: {missing}")
    
    if len(matched) == 0:
        print("\n[!] WARNING: No expected key patterns found in dder.pt.")
    else:
        print(f"\n[✓] Found {len(matched)} keys matching DDER patterns.")

def check_daclip_folder(daclip_path):
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("CHECK 2: DDER/daclip/ folder verification")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if not os.path.exists(daclip_path):
        print(f"Directory not found: {daclip_path}")
        sys.exit(1)
    
    print(f"→ Files inside DDER/daclip/:")
    file_count = 0
    for root, dirs, files in os.walk(daclip_path):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), daclip_path)
            print(f"   {rel}")
            file_count += 1
    
    print(f"→ Total files: {file_count}")
    
    # Check for critical files
    model_py = os.path.join(daclip_path, 'open_clip', 'model.py')
    tokenizer_py = os.path.join(daclip_path, 'open_clip', 'tokenizer.py')
    
    if os.path.exists(model_py):
        print("[✓] open_clip/model.py found")
    else:
        print("[!] open_clip/model.py NOT found")
        
    if os.path.exists(tokenizer_py):
        print("[✓] open_clip/tokenizer.py found")
    else:
        print("[!] open_clip/tokenizer.py NOT found")

def check_aflb():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("CHECK 3: AFLB (EvoIR) verification")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    aflb_model = os.path.join(os.path.dirname(__file__), 'evoIR_aflb', 'pretrained_model.py')
    if os.path.exists(aflb_model):
        print(f"[✓] AFLB model file found: {aflb_model}")
    else:
        print(f"[!] AFLB model file NOT found: {aflb_model}")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    dder_pt_path = os.path.join(base, 'DDER', 'dder.pt')
    daclip_path = os.path.join(base, 'DDER', 'daclip')
    
    check_dder_weights(dder_pt_path)
    check_daclip_folder(daclip_path)
    check_aflb()
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Diagnostics complete.")
