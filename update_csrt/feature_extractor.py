"""
Deep Feature Extractor using VGG16/ResNet/MobileNet
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Literal


class FeatureExtractor(nn.Module):
    """Extract deep features from backbone CNN"""
    
    def __init__(
        self, 
        backbone: Literal['vgg16', 'resnet50', 'mobilenetv2'] = 'vgg16',
        layer: str = 'conv4_3',
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone
        self.layer_name = layer
        
        if backbone == 'vgg16':
            vgg = models.vgg16(pretrained=pretrained)
            if layer == 'conv4_3':
                # VGG16 conv4_3: features[22] (28x28x512 for 224x224 input)
                self.features = nn.Sequential(*list(vgg.features[:23]))
                self.output_dim = 512
            elif layer == 'conv5_3':
                # VGG16 conv5_3: features[29] (14x14x512 for 224x224 input)
                self.features = nn.Sequential(*list(vgg.features[:30]))
                self.output_dim = 512
            else:
                raise ValueError(f"Unsupported VGG layer: {layer}")
                
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            if layer == 'layer3':
                # ResNet layer3: (28x28x1024 for 224x224 input)
                self.features = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                    resnet.layer1, resnet.layer2, resnet.layer3
                )
                self.output_dim = 1024
            elif layer == 'layer4':
                # ResNet layer4: (14x14x2048 for 224x224 input)
                self.features = nn.Sequential(
                    resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                    resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                )
                self.output_dim = 2048
            else:
                raise ValueError(f"Unsupported ResNet layer: {layer}")
                
        elif backbone == 'mobilenetv2':
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            # MobileNetV2 bottleneck_14: features[14] (14x14x96)
            self.features = nn.Sequential(*list(mobilenet.features[:15]))
            self.output_dim = 96
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone weights (optional - can be fine-tuned later)
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image tensor, normalized to [0, 1]
        Returns:
            features: (B, C, H', W') feature maps
        """
        return self.features(x)
    
    def extract_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract features and apply binary mask
        
        Args:
            x: (B, 3, H, W) RGB image
            mask: (B, 1, H, W) binary mask {0, 1}
        Returns:
            masked_features: (B, C, H', W') features with background zeroed
        """
        features = self.forward(x)
        
        # Resize mask to match feature spatial dimensions
        B, C, H, W = features.shape
        mask_resized = torch.nn.functional.interpolate(
            mask, size=(H, W), mode='nearest'
        )
        
        # Apply mask: zero out background
        masked_features = features * mask_resized
        
        return masked_features


if __name__ == '__main__':
    # Test feature extractor
    extractor = FeatureExtractor(backbone='vgg16', layer='conv4_3')
    print(f"Feature extractor: {extractor.backbone_name} {extractor.layer_name}")
    print(f"Output dimension: {extractor.output_dim}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    mask = torch.randint(0, 2, (2, 1, 224, 224)).float()
    
    features = extractor(x)
    print(f"Features shape: {features.shape}")
    
    masked_features = extractor.extract_masked(x, mask)
    print(f"Masked features shape: {masked_features.shape}")
