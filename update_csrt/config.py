"""
Configuration for PCSRT Tracker
"""

class PCSRTConfig:
    """Hyperparameters for PCSRT"""
    
    # Loss weights
    lambda_peak = 1.0       # Peak response loss
    lambda_smooth = 0.1     # Smoothness regularization
    lambda_reg = 1e-4       # Weight decay
    
    # Feature extraction
    backbone = 'vgg16'      # 'vgg16', 'resnet50', 'mobilenetv2'
    feature_layer = 'conv4_3'  # For VGG16: conv4_3 (28x28x512)
    
    # CorrProject architecture
    feature_dim = 512       # Input feature dimension
    hidden_dims = [256, 64] # Hidden layer dimensions
    output_dim = 31         # Match CSRT (13 HOG + 10 CN + padding)
    
    # Training
    learning_rate = 1e-4
    batch_size = 8
    num_epochs = 50
    
    # Tracker parameters
    search_region_scale = 2.0  # Scale factor for search region
    target_size = 28           # Resize target patch to 28x28
    
    # Adaptive fusion
    alpha_min = 0.3         # Minimum weight for h_csrt
    alpha_max = 0.9         # Maximum weight for h_csrt
    alpha_adaptive = True   # Use learned adaptive gating
    
    # Update frequency
    update_deep_every = 1   # Update h_deep every N frames (1 = every frame)
    
    # Segmentation mask
    mask_threshold = 0.5    # Threshold for binary mask
    mask_smoothing = 3      # Kernel size for mask smoothing
    
    # Dataset
    dataset_root = 'otb100/OTB-dataset/OTB100'
    sequences = None        # None = use all sequences
    
    def __repr__(self):
        return f"PCSRTConfig(backbone={self.backbone}, alpha_adaptive={self.alpha_adaptive})"
