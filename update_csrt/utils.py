"""
Utility functions for PCSRT
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes
    
    Args:
        bbox1, bbox2: (4,) [x, y, w, h]
    Returns:
        iou: Intersection over Union
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-8)
    return iou


def create_gaussian_label(
    size: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    sigma: float = 2.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Create Gaussian label for target
    
    Args:
        size: (H, W)
        center: (cy, cx), if None use center of image
        sigma: Gaussian bandwidth
        device: torch device
    Returns:
        label: (H, W) Gaussian heatmap
    """
    H, W = size
    if center is None:
        cy, cx = H / 2, W / 2
    else:
        cy, cx = center
    
    y, x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    label = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    
    return label


def extract_patch(
    image: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[int, int],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extract and resize patch from image
    
    Args:
        image: (H, W, 3) image
        center: (cy, cx) center coordinates
        size: (h, w) patch size
        output_size: (h_out, w_out) resize size, if None keep original
    Returns:
        patch: extracted patch
    """
    H, W = image.shape[:2]
    cy, cx = center
    h, w = size
    
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    
    # Handle boundary
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    # Extract patch
    patch = image[y1:y2, x1:x2]
    
    # Pad if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    
    # Resize if needed
    if output_size is not None:
        patch = cv2.resize(patch, output_size)
    
    return patch


def visualize_response(
    image: np.ndarray,
    response: np.ndarray,
    bbox: Optional[np.ndarray] = None,
    title: str = "Response Map"
) -> np.ndarray:
    """
    Visualize response map overlaid on image
    
    Args:
        image: (H, W, 3) BGR image
        response: (H, W) response map
        bbox: (4,) [x, y, w, h] bounding box to draw
        title: plot title
    Returns:
        vis: (H, W, 3) visualization image
    """
    # Normalize response to [0, 1]
    response_norm = (response - response.min()) / (response.max() - response.min() + 1e-8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap((response_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Overlay
    vis = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # Draw bbox if provided
    if bbox is not None:
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return vis


def compute_success_curve(
    ious: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute success curve from IoU values
    
    Args:
        ious: (N,) IoU values for all frames
        thresholds: (M,) IoU thresholds, if None use linspace(0, 1, 51)
    Returns:
        thresholds: (M,) thresholds used
        success_rates: (M,) success rate at each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 51)
    
    success_rates = np.array([
        np.mean(ious >= thresh) for thresh in thresholds
    ])
    
    return thresholds, success_rates


def compute_auc(success_rates: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Compute Area Under Curve for success plot
    
    Args:
        success_rates: (M,) success rates
        thresholds: (M,) thresholds
    Returns:
        auc: area under curve
    """
    auc = np.trapz(success_rates, thresholds)
    return auc


def plot_success_curve(
    results: dict,
    save_path: Optional[str] = None
):
    """
    Plot success curves for multiple trackers
    
    Args:
        results: {tracker_name: {'thresholds': ..., 'success_rates': ..., 'auc': ...}}
        save_path: path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    for name, data in results.items():
        thresholds = data['thresholds']
        success_rates = data['success_rates']
        auc = data.get('auc', compute_auc(success_rates, thresholds))
        
        plt.plot(thresholds, success_rates, linewidth=2, label=f"{name} [{auc:.3f}]")
    
    plt.xlabel('Overlap threshold', fontsize=12)
    plt.ylabel('Success rate', fontsize=12)
    plt.title('Success plots of OPE', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc='upper right', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved success plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test IoU
    bbox1 = np.array([10, 10, 50, 50])
    bbox2 = np.array([30, 30, 50, 50])
    iou = compute_iou(bbox1, bbox2)
    print(f"IoU: {iou:.3f}")
    
    # Test Gaussian label
    label = create_gaussian_label((28, 28), sigma=2.0)
    print(f"Gaussian label shape: {label.shape}, max: {label.max():.3f}")
    
    # Test success curve
    ious = np.random.rand(100)
    thresholds, success_rates = compute_success_curve(ious)
    auc = compute_auc(success_rates, thresholds)
    print(f"AUC: {auc:.3f}")
    
    print("All tests passed!")
