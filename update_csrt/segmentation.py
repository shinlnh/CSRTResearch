"""
Binary Mask Segmentation for CSRT
Graph-based segmentation from HOG/ColorNames response consistency
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


class MaskGenerator:
    """
    Generate binary segmentation mask from feature response consistency
    Similar to CSRT's mask generation using graph-based clustering
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        smoothing_kernel: int = 3,
        min_mask_ratio: float = 0.1
    ):
        self.threshold = threshold
        self.smoothing_kernel = smoothing_kernel
        self.min_mask_ratio = min_mask_ratio
    
    def create_mask_from_response(
        self,
        response: torch.Tensor,
        sigma: float = 2.0
    ) -> torch.Tensor:
        """
        Create binary mask from response map using thresholding
        
        Args:
            response: (B, H, W) response map
            sigma: Gaussian sigma for smoothing
        Returns:
            mask: (B, 1, H, W) binary mask {0, 1}
        """
        B, H, W = response.shape
        device = response.device
        
        # Normalize response to [0, 1]
        response_norm = (response - response.min()) / (response.max() - response.min() + 1e-8)
        
        # Apply Gaussian smoothing
        if self.smoothing_kernel > 1:
            kernel_size = self.smoothing_kernel
            response_norm = F.avg_pool2d(
                response_norm.unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ).squeeze(1)
        
        # Threshold to create binary mask
        mask = (response_norm > self.threshold).float().unsqueeze(1)  # (B, 1, H, W)
        
        # Ensure minimum mask coverage (avoid degenerate cases)
        mask_ratio = mask.mean(dim=(-2, -1), keepdim=True)  # (B, 1, 1, 1)
        if (mask_ratio < self.min_mask_ratio).any():
            # Fallback: use top K% pixels
            k = int(H * W * self.min_mask_ratio)
            for b in range(B):
                if mask_ratio[b] < self.min_mask_ratio:
                    response_flat = response_norm[b].flatten()
                    threshold_k = torch.kthvalue(response_flat, len(response_flat) - k).values
                    mask[b] = (response_norm[b] > threshold_k).float().unsqueeze(0)
        
        return mask
    
    def create_mask_from_features(
        self,
        features: torch.Tensor,
        filter_h: torch.Tensor
    ) -> torch.Tensor:
        """
        Create mask from feature-filter response
        
        Args:
            features: (B, C, H, W) input features
            filter_h: (C, H, W) correlation filter
        Returns:
            mask: (B, 1, H, W) binary mask
        """
        # Compute response
        F_features = torch.fft.rfft2(features, dim=(-2, -1))
        F_filter = torch.fft.rfft2(filter_h, dim=(-2, -1))
        F_response = torch.sum(F_features * F_filter.conj().unsqueeze(0), dim=1)
        response = torch.fft.irfft2(F_response, s=features.shape[-2:], dim=(-2, -1))
        
        return self.create_mask_from_response(response)
    
    def create_circular_mask(
        self,
        size: tuple,
        center: tuple,
        radius: float,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Create circular binary mask (fallback/initialization)
        
        Args:
            size: (H, W) mask size
            center: (cy, cx) center coordinates
            radius: circle radius
            device: torch device
        Returns:
            mask: (1, 1, H, W) circular mask
        """
        H, W = size
        cy, cx = center
        
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
        mask = (dist <= radius).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return mask
    
    def refine_mask_morphology(
        self,
        mask: torch.Tensor,
        iterations: int = 2
    ) -> torch.Tensor:
        """
        Refine mask using morphological operations
        
        Args:
            mask: (B, 1, H, W) binary mask
            iterations: number of morphological iterations
        Returns:
            refined_mask: (B, 1, H, W) refined mask
        """
        if iterations == 0:
            return mask
        
        B, _, H, W = mask.shape
        refined = mask.clone()
        
        # Convert to numpy for OpenCV morphology
        for b in range(B):
            mask_np = (mask[b, 0].cpu().numpy() * 255).astype(np.uint8)
            
            # Morphological closing (fill small holes)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Morphological opening (remove small noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=1)
            
            refined[b, 0] = torch.from_numpy(mask_np / 255.0).to(mask.device)
        
        return refined
    
    def create_mask_from_bbox(
        self,
        size: tuple,
        bbox: tuple,
        margin: float = 0.1,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Create mask from bounding box with margin
        
        Args:
            size: (H, W) mask size
            bbox: (x, y, w, h) bounding box
            margin: margin ratio to expand bbox
            device: torch device
        Returns:
            mask: (1, 1, H, W) mask
        """
        H, W = size
        x, y, w, h = bbox
        
        # Expand bbox by margin
        w_margin = w * margin
        h_margin = h * margin
        x1 = max(0, int(x - w_margin))
        y1 = max(0, int(y - h_margin))
        x2 = min(W, int(x + w + w_margin))
        y2 = min(H, int(y + h + h_margin))
        
        mask = torch.zeros(1, 1, H, W, device=device)
        mask[0, 0, y1:y2, x1:x2] = 1.0
        
        return mask


if __name__ == '__main__':
    # Test mask generator
    generator = MaskGenerator(threshold=0.5, smoothing_kernel=3)
    
    # Test 1: Create mask from response
    print("Test 1: Mask from response")
    response = torch.randn(2, 28, 28)
    mask = generator.create_mask_from_response(response)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.mean():.3f}")
    
    # Test 2: Circular mask
    print("\nTest 2: Circular mask")
    circular = generator.create_circular_mask((28, 28), (14, 14), radius=10.0)
    print(f"Circular mask shape: {circular.shape}")
    print(f"Circular mask ratio: {circular.mean():.3f}")
    
    # Test 3: Bbox mask
    print("\nTest 3: Bbox mask")
    bbox_mask = generator.create_mask_from_bbox((28, 28), (5, 5, 18, 18), margin=0.1)
    print(f"Bbox mask shape: {bbox_mask.shape}")
    print(f"Bbox mask ratio: {bbox_mask.mean():.3f}")
    
    # Test 4: Morphological refinement
    print("\nTest 4: Morphological refinement")
    noisy_mask = torch.rand(1, 1, 28, 28) > 0.5
    refined = generator.refine_mask_morphology(noisy_mask.float(), iterations=2)
    print(f"Before refinement: {noisy_mask.float().mean():.3f}")
    print(f"After refinement: {refined.mean():.3f}")
