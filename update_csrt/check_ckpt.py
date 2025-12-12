import torch

ckpt = torch.load('checkpoints/checkpoint_best.pth', map_location='cpu', weights_only=False)
print("Checkpoint keys:", list(ckpt.keys()))
if 'model_state_dict' in ckpt:
    print("\nModel state_dict keys (first 10):")
    print(list(ckpt['model_state_dict'].keys())[:10])
else:
    print("\nModel is directly saved, keys (first 10):")
    print(list(ckpt.keys())[:10])
