"""
PCSRT Tracker - Progressive CSRT with Deep Features
Main tracker implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
from pathlib import Path

from .feature_extractor import FeatureExtractor
from .corr_project import CorrProject, HybridFilter
from .dcf_solver import DCFSolver
from .segmentation import MaskGenerator
from .utils import create_gaussian_label, extract_patch


class PCSRTTracker(nn.Module):
    """
    Progressive CSRT Tracker with Deep Features
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extractor (deep features)
        self.feature_extractor = FeatureExtractor(
            backbone=config.backbone,
            layer=config.feature_layer,
            pretrained=True
        )
        
        # CorrProject: map deep features to correlation space
        self.corr_project = CorrProject(
            input_dim=self.feature_extractor.output_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim
        )
        
        # Hybrid filter fusion
        self.hybrid_filter = HybridFilter(
            feature_dim=config.output_dim,
            adaptive=config.alpha_adaptive,
            alpha_fixed=0.5,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max
        )
        
        # DCF solver (for h_csrt baseline)
        self.dcf_solver = DCFSolver(
            feature_size=(config.target_size // 8, config.target_size // 8),  # Approx feature map size
            num_channels=config.output_dim,
            lambda_reg=config.lambda_reg
        )
        
        # Mask generator
        self.mask_generator = MaskGenerator(
            threshold=config.mask_threshold,
            smoothing_kernel=config.mask_smoothing
        )
        
        # State variables
        self.initialized = False
        self.target_pos = None
        self.target_sz = None
        self.h_csrt = None
        self.h_deep = None
        self.mask = None
        self.frame_count = 0
        
    def initialize(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ):
        """
        Initialize tracker with first frame
        
        Args:
            image: (H, W, 3) BGR image
            bbox: (4,) [x, y, w, h]
        """
        x, y, w, h = bbox
        self.target_pos = np.array([y + h/2, x + w/2])  # (cy, cx)
        self.target_sz = np.array([h, w])
        self.frame_count = 0
        
        # Extract initial patch
        patch = self._extract_patch(image, self.target_pos, self.target_sz)
        patch_tensor = self._preprocess_image(patch)
        
        # Extract deep features
        with torch.no_grad():
            deep_features = self.feature_extractor(patch_tensor)  # (1, C, H, W)
        
        # Create initial mask (circular or bbox-based)
        H, W = deep_features.shape[-2:]
        self.mask = self.mask_generator.create_circular_mask(
            (H, W), (H/2, W/2), radius=min(H, W) / 4,
            device=deep_features.device
        )
        
        # Extract masked deep features
        masked_features = deep_features * self.mask
        
        # Project to correlation space
        h_deep_proj = self.corr_project(masked_features)  # (1, 31, H, W)
        
        # Create Gaussian target
        target = self.dcf_solver.create_gaussian_target((H, W), sigma=2.0)
        target = target.unsqueeze(0).to(deep_features.device)
        
        # Solve for h_csrt (baseline correlation filter)
        # For now, use same features (simplified - should use HOG/ColorNames)
        self.h_csrt = self.dcf_solver.solve_unconstrained(
            h_deep_proj, target
        ).squeeze(0)  # (31, H, W)
        
        # Solve for h_deep with mask constraint
        h_m, h_c = self.dcf_solver.solve_with_mask_admm(
            h_deep_proj, target, self.mask
        )
        self.h_deep = h_m  # Use masked filter
        
        self.initialized = True
    
    def track(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Track object in new frame
        
        Args:
            image: (H, W, 3) BGR image
        Returns:
            bbox: (4,) [x, y, w, h]
            confidence: tracking confidence score
        """
        if not self.initialized:
            raise RuntimeError("Tracker not initialized")
        
        self.frame_count += 1
        
        # Extract search region
        search_patch = self._extract_patch(
            image, self.target_pos, self.target_sz * self.config.search_region_scale
        )
        search_tensor = self._preprocess_image(search_patch)
        
        # Extract deep features
        with torch.no_grad():
            deep_features = self.feature_extractor(search_tensor)
            
            # Project to correlation space
            h_deep_proj = self.corr_project(deep_features)
            
            # Fuse h_csrt and h_deep
            h_final, alpha = self.hybrid_filter(
                self.h_csrt.unsqueeze(0),
                self.h_deep.unsqueeze(0),
                context=h_deep_proj
            )
            h_final = h_final.squeeze(0)
            
            # Compute response map
            response = self.dcf_solver.apply_filter(
                h_deep_proj, h_final
            ).squeeze(0)  # (H, W)
        
        # Find peak location
        peak_idx = response.argmax()
        H, W = response.shape
        peak_y = peak_idx // W
        peak_x = peak_idx % W
        
        # Convert to image coordinates
        # (Simplified - should account for scale and translation)
        scale_y = self.target_sz[0] * self.config.search_region_scale / H
        scale_x = self.target_sz[1] * self.config.search_region_scale / W
        
        dy = (peak_y - H/2) * scale_y
        dx = (peak_x - W/2) * scale_x
        
        # Update position
        self.target_pos[0] += dy
        self.target_pos[1] += dx
        
        # Update filters (every N frames)
        if self.frame_count % self.config.update_deep_every == 0:
            self._update_filters(deep_features, response, h_deep_proj)
        
        # Compute bbox
        y, x = self.target_pos
        h, w = self.target_sz
        bbox = np.array([x - w/2, y - h/2, w, h])
        
        # Compute confidence from response peak
        confidence = float(response.max().item())
        
        return bbox, confidence
    
    def _update_filters(
        self,
        deep_features: torch.Tensor,
        response: torch.Tensor,
        h_deep_proj: torch.Tensor
    ):
        """Update correlation filters and mask"""
        # Update mask from response
        self.mask = self.mask_generator.create_mask_from_response(
            response.unsqueeze(0)
        )
        
        # Apply mask to features
        masked_features = deep_features * self.mask
        h_deep_masked = self.corr_project(masked_features)
        
        # Create target
        H, W = response.shape
        target = self.dcf_solver.create_gaussian_target((H, W), sigma=2.0)
        target = target.unsqueeze(0).to(response.device)
        
        # Update h_csrt
        self.h_csrt = self.dcf_solver.solve_unconstrained(
            h_deep_proj, target
        ).squeeze(0)
        
        # Update h_deep with mask constraint
        h_m, _ = self.dcf_solver.solve_with_mask_admm(
            h_deep_masked, target, self.mask
        )
        self.h_deep = h_m
    
    def _extract_patch(
        self,
        image: np.ndarray,
        center: np.ndarray,
        size: np.ndarray
    ) -> np.ndarray:
        """Extract and resize patch from image"""
        cy, cx = center
        h, w = size
        
        patch = extract_patch(
            image,
            center=(cy, cx),
            size=(int(h), int(w)),
            output_size=(self.config.target_size, self.config.target_size)
        )
        
        return patch
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to tensor and normalize"""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To tensor and normalize to [0, 1]
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        if next(self.parameters()).is_cuda:
            tensor = tensor.cuda()
        
        return tensor


if __name__ == '__main__':
    from config import PCSRTConfig
    
    print("Testing PCSRT Tracker...")
    config = PCSRTConfig()
    tracker = PCSRTTracker(config)
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox_init = np.array([100, 100, 80, 80])
    
    # Initialize
    print("Initializing tracker...")
    tracker.initialize(frame, bbox_init)
    print("Tracker initialized!")
    
    # Track
    print("Tracking...")
    bbox_pred, confidence = tracker.track(frame)
    print(f"Predicted bbox: {bbox_pred}")
    print(f"Confidence: {confidence:.4f}")
