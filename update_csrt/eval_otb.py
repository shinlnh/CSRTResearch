"""
Quick OTB-100 Test using Python tracker
Outputs AUC and Precision@20px
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import glob

# Add parent directory
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
            # Try comma or space separated
            if ',' in line:
                parts = [float(x) for x in line.split(',')]
            else:
                parts = [float(x) for x in line.split()]
            if len(parts) >= 4:
                gt.append(parts[:4])
    return np.array(gt)

def track_sequence(seq_path):
    """Track one sequence"""
    seq_name = seq_path.name
    print(f"\nTracking {seq_name}...")
    
    # Find ground truth
    gt_files = list(seq_path.glob("**/groundtruth*.txt"))
    if not gt_files:
        print(f"  SKIP: No groundtruth found")
        return None
    
    gt_boxes = load_groundtruth(gt_files[0])
    
    # Find images
    img_patterns = ["**/img/*.jpg", "**/img/*.png", "**/*.jpg", "**/*.png"]
    img_files = []
    for pattern in img_patterns:
        img_files = sorted(list(seq_path.glob(pattern)))
        if img_files:
            break
    
    if not img_files:
        print(f"  SKIP: No images found")
        return None
    
    num_frames = min(len(gt_boxes), len(img_files))
    print(f"  Frames: {num_frames}")
    
    # Initialize tracker
    config = PCSRTConfig()
    tracker = PCSRTTracker(config).eval()  # Run on CPU
    
    # First frame
    frame = cv2.imread(str(img_files[0]))
    if frame is None:
        print(f"  SKIP: Cannot read first frame")
        return None
    
    init_bbox = gt_boxes[0]
    tracker.init(frame, init_bbox)
    
    # Track
    overlaps = []
    distances = []
    
    for i in range(1, num_frames):
        frame = cv2.imread(str(img_files[i]))
        if frame is None:
            break
        
        pred_bbox = tracker.track(frame)
        gt_bbox = gt_boxes[i]
        
        iou = compute_iou(pred_bbox, gt_bbox)
        dist = compute_center_distance(pred_bbox, gt_bbox)
        
        overlaps.append(iou)
        distances.append(dist)
        
        if (i+1) % 50 == 0:
            print(f"    Frame {i+1}/{num_frames}", end='\r')
    
    print()
    
    # Compute metrics
    overlaps = np.array(overlaps)
    distances = np.array(distances)
    
    # AUC (success plot)
    thresholds = np.linspace(0, 1, 50)
    success_rates = [(overlaps > t).mean() for t in thresholds]
    auc = np.mean(success_rates)
    
    # Precision @ 20px
    precision_20 = (distances <= 20).mean()
    
    print(f"  AUC: {auc:.3f}, P@20: {precision_20:.3f}")
    
    return {
        'name': seq_name,
        'auc': auc,
        'precision_20': precision_20,
        'overlaps': overlaps,
        'distances': distances
    }

def main():
    otb_path = Path("../otb100/OTB-dataset/OTB100")
    
    if not otb_path.exists():
        print(f"Error: OTB100 dataset not found at {otb_path}")
        return
    
    print("="*80)
    print("OTB-100 Evaluation (Python)")
    print("="*80)
    
    # Get all sequences
    sequences = sorted([d for d in otb_path.iterdir() if d.is_dir()])
    print(f"\nFound {len(sequences)} sequences\n")
    
    # Track all
    results = []
    for seq_path in sequences:
        result = track_sequence(seq_path)
        if result:
            results.append(result)
    
    if not results:
        print("\nNo valid results!")
        return
    
    # Overall metrics
    mean_auc = np.mean([r['auc'] for r in results])
    mean_precision = np.mean([r['precision_20'] for r in results])
    
    print("\n" + "="*80)
    print("OTB-100 Results")
    print("="*80)
    print(f"Sequences: {len(results)}")
    print(f"AUC:       {mean_auc:.3f}")
    print(f"P@20px:    {mean_precision:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
