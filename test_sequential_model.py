import torch
from model import HocVidModel

print("Initializing HocVidModel...")
model = HocVidModel()
# We don't bother strictly loading weights for this shape check since we just need mathematical verification
model.eval()

dummy_input = torch.rand(1, 3, 256, 256, device='cpu')
print(f"Input shape: {dummy_input.shape}")

print("Running forward pass...")
with torch.no_grad():
    out = model(dummy_input)

print(f"Final output shape (Degradation Map): {out.shape}")

# Verify no NaN values
has_nan = torch.isnan(out).any().item()
print(f"Contains NaNs: {has_nan}")

if not has_nan and out.shape == dummy_input.shape:
    print("SUCCESS! Sequential Extraction Pipeline operational.")
else:
    print("FAILURE!")
