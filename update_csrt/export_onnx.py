"""
Export trained models to ONNX format for C++ inference

This script exports:
1. VGG16 conv4_3 feature extractor (pretrained)
2. CorrProject network (trained)
3. AdaptiveGating network (trained)
"""

import torch
import torch.onnx
import os
from feature_extractor import FeatureExtractor
from corr_project import CorrProject, AdaptiveGating

def export_vgg16(output_path):
    """Export VGG16 conv4_3 to ONNX"""
    print(f"Exporting VGG16 conv4_3 to {output_path}...")
    
    # Create VGG16 extractor
    extractor = FeatureExtractor(backbone='vgg16', layer='conv4_3', pretrained=True)
    extractor.eval()
    
    # Dummy input (1, 3, 127, 127) - template size
    dummy_input = torch.randn(1, 3, 127, 127)
    
    # Export
    torch.onnx.export(
        extractor.features,  # Only export the features module
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✓ VGG16 exported successfully")

def export_corr_projection(checkpoint_path, output_path):
    """Export CorrProjection to ONNX"""
    print(f"Exporting CorrProjection to {output_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    corr_proj = CorrProject(
        input_dim=512,  # VGG16 conv4_3
        hidden_dims=[256, 64],
        output_dim=31   # HOG (21) + ColorNames (10)
    )
    
    # Load weights
    state_dict = checkpoint['model']
    corr_proj_state = {k.replace('corr_project.', ''): v 
                       for k, v in state_dict.items() 
                       if k.startswith('corr_project.')}
    corr_proj.load_state_dict(corr_proj_state)
    corr_proj.eval()
    
    # Dummy input (1, 512, 15, 15) - VGG16 conv4_3 output for 127x127 input
    dummy_input = torch.randn(1, 512, 15, 15)
    
    # Export
    torch.onnx.export(
        corr_proj,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✓ CorrProjection exported successfully")

def export_adaptive_gating(checkpoint_path, output_path):
    """Export AdaptiveGating to ONNX"""
    print(f"Exporting AdaptiveGating to {output_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    adaptive_gating = AdaptiveGating(
        feature_dim=31,
        alpha_min=0.3,
        alpha_max=0.9
    )
    
    # Load weights
    state_dict = checkpoint['model']
    gating_state = {k.replace('hybrid_filter.gating.', ''): v 
                    for k, v in state_dict.items() 
                    if k.startswith('hybrid_filter.gating.')}
    adaptive_gating.load_state_dict(gating_state)
    adaptive_gating.eval()
    
    # Dummy input: features (1x31x31x31 for response map)
    dummy_features = torch.randn(1, 31, 31, 31)
    
    # Export
    torch.onnx.export(
        adaptive_gating,
        dummy_features,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['alpha'],
        dynamic_axes={
            'features': {0: 'batch_size'}
        }
    )
    print(f"✓ AdaptiveGating exported successfully")

def main():
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Checkpoint path
    checkpoint_path = "checkpoints/checkpoint_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    print("="*80)
    print("Exporting models to ONNX format")
    print("="*80)
    
    # Export VGG16
    export_vgg16(os.path.join(models_dir, "vgg16_conv4_3.onnx"))
    
    # Export CorrProjection
    export_corr_projection(
        checkpoint_path, 
        os.path.join(models_dir, "corr_project.onnx")
    )
    
    # Export AdaptiveGating
    export_adaptive_gating(
        checkpoint_path,
        os.path.join(models_dir, "adaptive_gating.onnx")
    )
    
    print("="*80)
    print("All models exported successfully!")
    print("="*80)
    print(f"Models saved to: {os.path.abspath(models_dir)}/")
    print("  - vgg16_conv4_3.onnx")
    print("  - corr_project.onnx")
    print("  - adaptive_gating.onnx")

if __name__ == "__main__":
    main()
