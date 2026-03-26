import torch

sd = torch.load('DDER/dder.pt', map_location='cpu', weights_only=False)
print(f"Type: {type(sd)}")

if isinstance(sd, dict) and 'model_state_dict' in sd:
    sd = sd['model_state_dict']

keys = list(sd.keys())

with open('dder_keys.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total keys: {len(keys)}\n\n")
    for k in keys:
        v = sd[k]
        if hasattr(v, 'shape'):
            f.write(f"  {k}: {v.shape}\n")
        else:
            f.write(f"  {k}: {type(v)}\n")

print(f"Written {len(keys)} keys to dder_keys.txt")
