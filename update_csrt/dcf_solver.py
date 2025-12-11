"""
DCF (Discriminative Correlation Filter) Solver with ADMM
Constraint: h = m ⊙ h (filter masked by binary segmentation)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class DCFSolver:
    """
    Solve DCF with spatial mask constraint using ADMM
    """
    
    def __init__(
        self,
        feature_size: tuple = (28, 28),
        num_channels: int = 31,
        lambda_reg: float = 1e-4,
        admm_iterations: int = 5,
        admm_penalty: float = 10.0
    ):
        self.feature_size = feature_size
        self.num_channels = num_channels
        self.lambda_reg = lambda_reg
        self.admm_iterations = admm_iterations
        self.admm_penalty = admm_penalty
        
    def create_gaussian_target(self, size: tuple, sigma: float = 2.0) -> torch.Tensor:
        """
        Create Gaussian-shaped desired response
        
        Args:
            size: (H, W) spatial size
            sigma: Gaussian bandwidth
        Returns:
            target: (H, W) Gaussian response centered at (H/2, W/2)
        """
        H, W = size
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        
        center_y, center_x = H / 2, W / 2
        target = torch.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        return target
    
    def solve_unconstrained(
        self, 
        features: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Solve standard DCF without mask constraint
        
        Args:
            features: (B, C, H, W) training features
            target: (B, H, W) desired response
        Returns:
            filter_h: (C, H, W) learned correlation filter
        """
        B, C, H, W = features.shape
        device = features.device
        
        # FFT of features and target
        F_features = torch.fft.rfft2(features, dim=(-2, -1))  # (B, C, H, W/2+1)
        F_target = torch.fft.rfft2(target, dim=(-2, -1))      # (B, H, W/2+1)
        
        # Compute numerator: sum over batch
        # numerator = sum_b conj(F_features_b) * F_target_b
        numerator = torch.sum(
            F_features.conj() * F_target.unsqueeze(1),  # (B, C, H, W/2+1)
            dim=0
        )  # (C, H, W/2+1)
        
        # Compute denominator: sum over batch and channels
        # denominator = sum_b sum_c |F_features_bc|^2 + lambda
        denominator = torch.sum(
            F_features.abs()**2,
            dim=(0, 1)  # sum over batch and channels
        ) + self.lambda_reg  # (H, W/2+1)
        
        # Solve in frequency domain
        F_filter = numerator / denominator.unsqueeze(0)  # (C, H, W/2+1)
        
        # IFFT back to spatial domain
        filter_h = torch.fft.irfft2(F_filter, s=(H, W), dim=(-2, -1))  # (C, H, W)
        
        return filter_h
    
    def solve_with_mask_admm(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve DCF with mask constraint: h = m ⊙ h using ADMM
        
        ADMM formulation:
        minimize: ||y - f*h||^2 + λ||h||^2
        subject to: h = m ⊙ h_m
        
        Args:
            features: (B, C, H, W) training features
            target: (B, H, W) desired response
            mask: (B, 1, H, W) or (1, 1, H, W) binary mask {0, 1}
        Returns:
            h_m: (C, H, W) masked filter
            h_c: (C, H, W) correlation filter
        """
        B, C, H, W = features.shape
        device = features.device
        
        # Initialize variables
        h_c = self.solve_unconstrained(features, target)  # (C, H, W)
        h_m = h_c.clone()
        u = torch.zeros_like(h_c)  # Dual variable
        
        # Average mask over batch if needed
        if mask.size(0) > 1:
            mask_avg = mask.mean(dim=0, keepdim=True)  # (1, 1, H, W)
        else:
            mask_avg = mask
        mask_binary = (mask_avg > 0.5).float().squeeze(0).squeeze(0)  # (H, W)
        
        rho = self.admm_penalty
        
        # ADMM iterations
        for _ in range(self.admm_iterations):
            # Update h_c: minimize ||y - f*h_c||^2 + ρ/2||h_c - m⊙h_m + u||^2
            # This has closed-form solution in frequency domain
            
            # FFT
            F_features = torch.fft.rfft2(features, dim=(-2, -1))
            F_target = torch.fft.rfft2(target, dim=(-2, -1))
            F_constraint = torch.fft.rfft2(mask_binary.unsqueeze(0) * h_m - u, dim=(-2, -1))
            
            # Numerator
            numerator = torch.sum(
                F_features.conj() * F_target.unsqueeze(1),
                dim=0
            ) + rho * F_constraint
            
            # Denominator
            denominator = torch.sum(
                F_features.abs()**2,
                dim=(0, 1)
            ) + self.lambda_reg + rho
            
            # Solve
            F_h_c = numerator / denominator.unsqueeze(0)
            h_c = torch.fft.irfft2(F_h_c, s=(H, W), dim=(-2, -1))
            
            # Update h_m: minimize ρ/2||h_c - m⊙h_m + u||^2
            # Solution: h_m = (h_c + u) / m (where m=1)
            h_m = mask_binary.unsqueeze(0) * (h_c + u)
            
            # Update dual variable u
            u = u + (h_c - mask_binary.unsqueeze(0) * h_m)
        
        return h_m, h_c
    
    def apply_filter(
        self,
        features: torch.Tensor,
        filter_h: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply correlation filter to get response map
        
        Args:
            features: (B, C, H, W) input features
            filter_h: (C, H, W) correlation filter
        Returns:
            response: (B, H, W) response map
        """
        # FFT
        F_features = torch.fft.rfft2(features, dim=(-2, -1))  # (B, C, H, W/2+1)
        F_filter = torch.fft.rfft2(filter_h, dim=(-2, -1))     # (C, H, W/2+1)
        
        # Correlation in frequency domain: element-wise multiply and sum over channels
        F_response = torch.sum(
            F_features * F_filter.conj().unsqueeze(0),
            dim=1
        )  # (B, H, W/2+1)
        
        # IFFT back to spatial domain
        response = torch.fft.irfft2(F_response, s=features.shape[-2:], dim=(-2, -1))
        
        return response


if __name__ == '__main__':
    # Test DCF solver
    solver = DCFSolver(feature_size=(28, 28), num_channels=31)
    
    # Create dummy data
    B, C, H, W = 4, 31, 28, 28
    features = torch.randn(B, C, H, W)
    target = solver.create_gaussian_target((H, W), sigma=2.0).unsqueeze(0).expand(B, -1, -1)
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    
    print("Testing unconstrained DCF...")
    h = solver.solve_unconstrained(features, target)
    print(f"Filter shape: {h.shape}")
    
    response = solver.apply_filter(features[:1], h)
    print(f"Response shape: {response.shape}")
    print(f"Response max at: {response[0].argmax().item()}, expected center: {H//2 * W + W//2}")
    
    print("\nTesting ADMM with mask constraint...")
    h_m, h_c = solver.solve_with_mask_admm(features, target, mask)
    print(f"h_m shape: {h_m.shape}")
    print(f"h_c shape: {h_c.shape}")
    print(f"Constraint violation: {torch.mean((h_c - mask[0,0].unsqueeze(0) * h_m)**2).item():.6f}")
