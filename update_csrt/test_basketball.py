"""
Quick test on Basketball sequence
Outputs AUC and Precision@20px
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from tracker import PCSRTTracker
from config import PCSRTConfig

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter = (xi2 - xi1) * (yi2 - yi1)
    union = w1*h1 + w2*h2 - inter
    return inter / union

def compute_center_distance(box1, box2):
    """Compute center distance"""
    c1x, c1y = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    c2x, c2y = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    return np.sqrt((c1x-c2x)**2 + (c1y-c2y)**2)

def load_groundtruth(gt_file):
    """Load ground truth annotations"""
    gt = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try comma or space or tab separated
            if ',' in line:
                parts = [float(x) for x in line.replace('\t', ',').split(',') if x.strip()]
            else:
                parts = [float(x) for x in line.split()]
            if len(parts) >= 4:
                # Handle format: x,y,w,h
                gt.append([parts[0], parts[1], parts[2], parts[3]])
    return np.array(gt)

def main():
    # Basketball sequence path
    seq_path = Path("../otb100/OTB-dataset/OTB100/Basketball/Basketball")
    
    if not seq_path.exists():
        print(f"Error: Basketball sequence not found at {seq_path}")
        print(f"Absolute: {seq_path.absolute()}")
        return
    
    print("="*80)
    print("Testing Updated CSRT Tracker on Basketball")
    print("="*80)
    
    # Load ground truth
    gt_file = seq_path / "groundtruth_rect.txt"
    if not gt_file.exists():
        print(f"Error: Ground truth not found at {gt_file}")
        return
    
    gt_boxes = load_groundtruth(gt_file)
    print(f"\nLoaded {len(gt_boxes)} ground truth boxes")
    
    # Load images
    img_dir = seq_path / "img"
    if not img_dir.exists():
        print(f"Error: Image directory not found at {img_dir}")
        return
    
    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.png"))
    
    print(f"Found {len(img_files)} images")
    
    num_frames = min(len(gt_boxes), len(img_files))
    print(f"Will process {num_frames} frames\n")
    
    # Initialize tracker
    print("Initializing tracker...")
    config = PCSRTConfig()
    
    # Use CPU
    device = torch.device('cpu')
    tracker = PCSRTTracker(config).to(device).eval()
    
    # First frame
    frame = cv2.imread(str(img_files[0]))
    if frame is None:
        print(f"Error: Cannot read first frame: {img_files[0]}")
        return
    
    print(f"Frame size: {frame.shape}")
    print(f"Initial bbox: {gt_boxes[0]}")
    
    # Initialize with ground truth
    init_bbox = gt_boxes[0]
    tracker.initialize(frame, init_bbox)
    
    print("\nTracking...")
    
    # Track remaining frames
    overlaps = []
    distances = []
    
    for i in range(1, num_frames):
        frame = cv2.imread(str(img_files[i]))
        if frame is None:
            print(f"Warning: Cannot read frame {i}")
            break
        
        # Track
        pred_bbox, confidence = tracker.track(frame)
        gt_bbox = gt_boxes[i]
        
        # Compute metrics
        iou = compute_iou(pred_bbox, gt_bbox)
        dist = compute_center_distance(pred_bbox, gt_bbox)
        
        overlaps.append(iou)
        distances.append(dist)
        
        # Progress
        if (i+1) % 50 == 0 or i == num_frames - 1:
            print(f"  Frame {i+1}/{num_frames} - IoU: {iou:.3f}, Dist: {dist:.1f}px")
    
    # Compute final metrics
    overlaps = np.array(overlaps)
    distances = np.array(distances)
    
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    
    # AUC (Area Under Curve for success plot)
    thresholds = np.linspace(0, 1, 50)
    success_rates = [(overlaps > t).mean() for t in thresholds]
    auc = np.mean(success_rates)
    
    # Precision @ 20px
    precision_20 = (distances <= 20).mean()
    
    # Additional metrics
    mean_iou = overlaps.mean()
    mean_dist = distances.mean()
    
    print(f"Frames tracked:  {len(overlaps)}")
    print(f"AUC:             {auc:.3f}")
    print(f"Precision@20px:  {precision_20:.3f}")
    print(f"Mean IoU:        {mean_iou:.3f}")
    print(f"Mean Distance:   {mean_dist:.1f}px")
    print("="*80)

if __name__ == "__main__":
    main()
