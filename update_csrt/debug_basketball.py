"""
Debug Basketball - visualize tracking
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent))

from tracker import PCSRTTracker
from config import PCSRTConfig

def load_groundtruth(gt_file):
    """Load ground truth annotations"""
    gt = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                parts = [float(x) for x in line.replace('\t', ',').split(',') if x.strip()]
            else:
                parts = [float(x) for x in line.split()]
            if len(parts) >= 4:
                gt.append([parts[0], parts[1], parts[2], parts[3]])
    return np.array(gt)

def draw_box(img, box, color, label=""):
    """Draw bounding box"""
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    if label:
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    seq_path = Path("../otb100/OTB-dataset/OTB100/Basketball/Basketball")
    
    # Load data
    gt_file = seq_path / "groundtruth_rect.txt"
    gt_boxes = load_groundtruth(gt_file)
    
    img_dir = seq_path / "img"
    img_files = sorted(img_dir.glob("*.jpg"))
    
    print("="*80)
    print("Debug Basketball Tracking (First 10 frames)")
    print("="*80)
    
    # Initialize tracker
    config = PCSRTConfig()
    device = torch.device('cpu')
    tracker = PCSRTTracker(config).to(device).eval()
    
    # First frame
    frame = cv2.imread(str(img_files[0]))
    init_bbox = gt_boxes[0]
    
    print(f"\nFrame 0:")
    print(f"  GT bbox: {init_bbox}")
    print(f"  Image shape: {frame.shape}")
    
    tracker.initialize(frame, init_bbox)
    
    # Create output directory
    output_dir = Path("debug_frames")
    output_dir.mkdir(exist_ok=True)
    
    # Save first frame
    vis = frame.copy()
    draw_box(vis, init_bbox, (0, 255, 0), "GT")
    cv2.imwrite(str(output_dir / "frame_000.jpg"), vis)
    
    # Track 10 frames
    for i in range(1, min(11, len(img_files))):
        frame = cv2.imread(str(img_files[i]))
        gt_bbox = gt_boxes[i]
        
        # Track
        pred_bbox, confidence = tracker.track(frame)
        
        print(f"\nFrame {i}:")
        print(f"  GT bbox:   {gt_bbox}")
        print(f"  Pred bbox: {pred_bbox}")
        print(f"  Confidence: {confidence:.3f}")
        
        # Visualize
        vis = frame.copy()
        draw_box(vis, gt_bbox, (0, 255, 0), "GT")
        draw_box(vis, pred_bbox, (0, 0, 255), f"Pred ({confidence:.2f})")
        
        # Add text
        cv2.putText(vis, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imwrite(str(output_dir / f"frame_{i:03d}.jpg"), vis)
    
    print(f"\n✓ Debug frames saved to {output_dir}/")
    print("Check frame_000.jpg to frame_010.jpg to see tracking results")

if __name__ == "__main__":
    main()
