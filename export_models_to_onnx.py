"""
Export PyTorch models to ONNX format for C++ inference
"""
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# Add update_csrt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from update_csrt.feature_extractor import FeatureExtractor
from update_csrt.corr_project import CorrProject, AdaptiveGating
from update_csrt.config import Config


def export_vgg16_conv4_3():
    """Export VGG16 conv4_3 features to ONNX"""
    print("Exporting VGG16 conv4_3...")
    
    # Load VGG16
    vgg16 = models.vgg16(pretrained=True)
    
    # Extract features up to conv4_3 (layer 23)
    feature_layers = nn.Sequential(*list(vgg16.features.children())[:24])
    feature_layers.eval()
    
    # Dummy input (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, 127, 127)
    
    # Export
    output_path = "update_csrt/models/vgg16_conv4_3.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        feature_layers,
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
    
    print(f"✓ VGG16 exported to {output_path}")
    
    # Test export
    with torch.no_grad():
        output_pytorch = feature_layers(dummy_input)
        print(f"  Output shape: {output_pytorch.shape}")


def export_corr_project(checkpoint_path):
    """Export CorrProject network to ONNX"""
    print("Exporting CorrProject network...")
    
    config = Config()
    
    # Load trained model
    model = CorrProject(config.deep_feature_channels, config.num_channels)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'corr_project' in checkpoint:
            model.load_state_dict(checkpoint['corr_project'])
            print(f"  Loaded weights from {checkpoint_path}")
        else:
            print("  Warning: No trained weights found, using random initialization")
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")
        print("  Using random initialization")
    
    model.eval()
    
    # Dummy input (batch, channels=512, height, width)
    dummy_input = torch.randn(1, 512, 15, 15)  # Typical VGG conv4_3 output size
    
    # Export
    output_path = "update_csrt/models/corr_project.onnx"
    
    torch.onnx.export(
        model,
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
    
    print(f"✓ CorrProject exported to {output_path}")
    
    # Test export
    with torch.no_grad():
        output_pytorch = model(dummy_input)
        print(f"  Output shape: {output_pytorch.shape}")


def export_adaptive_gating(checkpoint_path):
    """Export AdaptiveGating network to ONNX"""
    print("Exporting AdaptiveGating network...")
    
    config = Config()
    
    # Load trained model
    model = AdaptiveGating(config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'adaptive_gating' in checkpoint:
            model.load_state_dict(checkpoint['adaptive_gating'])
            print(f"  Loaded weights from {checkpoint_path}")
        else:
            print("  Warning: No trained weights found, using random initialization")
    else:
        print(f"  Warning: Checkpoint not found at {checkpoint_path}")
        print("  Using random initialization")
    
    model.eval()
    
    # Dummy input (context features)
    dummy_input = torch.randn(1, 4)  # 4 context features
    
    # Export
    output_path = "update_csrt/models/adaptive_gating.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['context'],
        output_names=['alpha']
    )
    
    print(f"✓ AdaptiveGating exported to {output_path}")
    
    # Test export
    with torch.no_grad():
        output_pytorch = model(dummy_input)
        print(f"  Alpha value: {output_pytorch.item():.4f}")


def main():
    print("================================================================================")
    print("PyTorch to ONNX Model Export")
    print("================================================================================")
    
    # Find latest checkpoint
    checkpoint_dir = "update_csrt/checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Models will be exported with random weights")
        print("Please train the model first using train_update_csrt.py")
    
    # Export all models
    try:
        export_vgg16_conv4_3()
        export_corr_project(checkpoint_path)
        export_adaptive_gating(checkpoint_path)
        
        print("\n================================================================================")
        print("✓ All models exported successfully!")
        print("================================================================================")
        print("\nExported files:")
        print("  - update_csrt/models/vgg16_conv4_3.onnx")
        print("  - update_csrt/models/corr_project.onnx")
        print("  - update_csrt/models/adaptive_gating.onnx")
        print("\nYou can now use these ONNX models in the C++ implementation.")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
