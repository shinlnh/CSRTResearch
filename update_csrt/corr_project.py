"""
CorrProject: Project deep features to correlation filter space
"""

import torch
import torch.nn as nn


class CorrProject(nn.Module):
    """
    Project deep features h_deep to correlation space h'_deep
    Compatible with CSRT filter dimension (31 channels)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: list = [256, 64],
        output_dim: int = 31
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with BatchNorm and ReLU
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no BN/activation)
        layers.append(
            nn.Conv2d(prev_dim, output_dim, kernel_size=1)
        )
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, h_deep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_deep: (B, C_in, H, W) deep features
        Returns:
            h_proj: (B, C_out, H, W) projected features
        """
        return self.projection(h_deep)


class AdaptiveGating(nn.Module):
    """
    Learn adaptive weight α for fusion: h = α*h_csrt + (1-α)*h'_deep
    """
    
    def __init__(self, feature_dim: int = 31, alpha_min: float = 0.3, alpha_max: float = 0.9):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # Simple gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) input features (can be h_csrt or response map)
        Returns:
            alpha: (B, 1) gating weight in [alpha_min, alpha_max]
        """
        gate_val = self.gate(features)  # (B, 1)
        # Scale to [alpha_min, alpha_max]
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * gate_val
        return alpha


class HybridFilter(nn.Module):
    """
    Combine h_csrt and h'_deep with adaptive or fixed gating
    """
    
    def __init__(
        self,
        feature_dim: int = 31,
        adaptive: bool = True,
        alpha_fixed: float = 0.5,
        alpha_min: float = 0.3,
        alpha_max: float = 0.9
    ):
        super().__init__()
        self.adaptive = adaptive
        self.alpha_fixed = alpha_fixed
        
        if adaptive:
            self.gating = AdaptiveGating(feature_dim, alpha_min, alpha_max)
        
    def forward(
        self, 
        h_csrt: torch.Tensor, 
        h_proj: torch.Tensor,
        context: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_csrt: (B, C, H, W) CSRT correlation filter
            h_proj: (B, C, H, W) projected deep filter
            context: (B, C, H, W) context for adaptive gating (e.g., response map)
        Returns:
            h_final: (B, C, H, W) fused filter
            alpha: (B, 1) gating weight used
        """
        if self.adaptive:
            # Use context (or h_csrt if context not provided) for gating
            gating_input = context if context is not None else h_csrt
            alpha = self.gating(gating_input)  # (B, 1)
            alpha = alpha.view(-1, 1, 1, 1)    # (B, 1, 1, 1) for broadcasting
        else:
            B = h_csrt.size(0)
            alpha = torch.full((B, 1, 1, 1), self.alpha_fixed, device=h_csrt.device)
        
        # Weighted fusion
        h_final = alpha * h_csrt + (1 - alpha) * h_proj
        
        return h_final, alpha.squeeze()


if __name__ == '__main__':
    # Test CorrProject
    proj = CorrProject(input_dim=512, hidden_dims=[256, 64], output_dim=31)
    h_deep = torch.randn(2, 512, 28, 28)
    h_proj = proj(h_deep)
    print(f"CorrProject: {h_deep.shape} -> {h_proj.shape}")
    
    # Test AdaptiveGating
    gating = AdaptiveGating(feature_dim=31)
    features = torch.randn(2, 31, 28, 28)
    alpha = gating(features)
    print(f"AdaptiveGating: alpha = {alpha}")
    
    # Test HybridFilter
    hybrid = HybridFilter(feature_dim=31, adaptive=True)
    h_csrt = torch.randn(2, 31, 28, 28)
    h_proj = torch.randn(2, 31, 28, 28)
    h_final, alpha = hybrid(h_csrt, h_proj)
    print(f"HybridFilter: h_final shape = {h_final.shape}, alpha = {alpha}")
