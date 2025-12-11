"""
Loss functions for PCSRT training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakLoss(nn.Module):
    """
    Peak response loss: maximize response at target center
    """
    
    def __init__(self, use_heatmap: bool = True):
        super().__init__()
        self.use_heatmap = use_heatmap
    
    def forward(
        self,
        response: torch.Tensor,
        target: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            response: (B, H, W) predicted response map
            target: (B, H, W) target Gaussian heatmap or None (use center)
        Returns:
            loss: scalar loss value
        """
        B, H, W = response.shape
        
        if self.use_heatmap and target is not None:
            # MSE between response and target heatmap
            loss = F.mse_loss(response, target)
        else:
            # Negative peak at center (maximize response[cy, cx])
            cy, cx = H // 2, W // 2
            loss = -response[:, cy, cx].mean()
        
        return loss


class SmoothnessLoss(nn.Module):
    """
    Smoothness regularization on response map
    Penalize large gradients
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
        # Sobel kernels for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(1))  # (1, 1, 3, 3)
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(1))  # (1, 1, 3, 3)
    
    def forward(self, response: torch.Tensor) -> torch.Tensor:
        """
        Args:
            response: (B, H, W) response map
        Returns:
            loss: smoothness loss
        """
        # Add channel dimension for conv2d
        response_4d = response.unsqueeze(1)  # (B, 1, H, W)
        
        # Compute gradients using Sobel filters
        grad_x = F.conv2d(response_4d, self.sobel_x, padding=1)
        grad_y = F.conv2d(response_4d, self.sobel_y, padding=1)
        
        # L2 norm of gradients
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # Average over spatial dimensions and batch
        loss = self.weight * grad_magnitude.mean()
        
        return loss


class RegularizationLoss(nn.Module):
    """
    L2 regularization on model parameters
    """
    
    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Args:
            model: PyTorch model (e.g., CorrProject)
        Returns:
            loss: L2 regularization loss
        """
        l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)**2
        
        return self.weight * l2_reg


class CompositeLoss(nn.Module):
    """
    Combined loss: Peak + Smoothness + Regularization
    """
    
    def __init__(
        self,
        lambda_peak: float = 1.0,
        lambda_smooth: float = 0.1,
        lambda_reg: float = 1e-4,
        use_heatmap: bool = True
    ):
        super().__init__()
        self.lambda_peak = lambda_peak
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg
        
        self.peak_loss = PeakLoss(use_heatmap=use_heatmap)
        self.smooth_loss = SmoothnessLoss(weight=1.0)  # weight applied in forward
        self.reg_loss = RegularizationLoss(weight=1.0)
    
    def forward(
        self,
        response: torch.Tensor,
        target: torch.Tensor,
        model: nn.Module
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            response: (B, H, W) predicted response map
            target: (B, H, W) target heatmap
            model: model for regularization
        Returns:
            total_loss: weighted sum of losses
            loss_dict: individual loss components
        """
        # Compute individual losses
        l_peak = self.peak_loss(response, target)
        l_smooth = self.smooth_loss(response)
        l_reg = self.reg_loss(model)
        
        # Weighted sum
        total_loss = (
            self.lambda_peak * l_peak +
            self.lambda_smooth * l_smooth +
            self.lambda_reg * l_reg
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'peak': l_peak.item(),
            'smooth': l_smooth.item(),
            'reg': l_reg.item()
        }
        
        return total_loss, loss_dict


if __name__ == '__main__':
    # Test losses
    B, H, W = 4, 28, 28
    
    # Create dummy response and target
    response = torch.randn(B, H, W, requires_grad=True)
    target = torch.zeros(B, H, W)
    target[:, H//2, W//2] = 1.0  # Peak at center
    
    # Test PeakLoss
    print("Test PeakLoss:")
    peak_loss = PeakLoss(use_heatmap=True)
    loss_peak = peak_loss(response, target)
    print(f"  Peak loss (heatmap): {loss_peak.item():.4f}")
    
    peak_loss_center = PeakLoss(use_heatmap=False)
    loss_center = peak_loss_center(response)
    print(f"  Peak loss (center): {loss_center.item():.4f}")
    
    # Test SmoothnessLoss
    print("\nTest SmoothnessLoss:")
    smooth_loss = SmoothnessLoss(weight=0.1)
    loss_smooth = smooth_loss(response)
    print(f"  Smoothness loss: {loss_smooth.item():.4f}")
    
    # Test RegularizationLoss
    print("\nTest RegularizationLoss:")
    model = nn.Linear(10, 10)
    reg_loss = RegularizationLoss(weight=1e-4)
    loss_reg = reg_loss(model)
    print(f"  Regularization loss: {loss_reg.item():.6f}")
    
    # Test CompositeLoss
    print("\nTest CompositeLoss:")
    composite = CompositeLoss(lambda_peak=1.0, lambda_smooth=0.1, lambda_reg=1e-4)
    total_loss, loss_dict = composite(response, target, model)
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Loss dict: {loss_dict}")
    
    # Test backward
    print("\nTest backward:")
    total_loss.backward()
    print(f"  Response grad norm: {response.grad.norm().item():.4f}")
