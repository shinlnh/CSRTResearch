"""
Test/Evaluate PCSRT on OTB benchmark
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import json

import config
from config import PCSRTConfig
import tracker
from tracker import PCSRTTracker
import dataset
from dataset import OTBSequence
import utils
from utils import compute_iou, compute_success_curve, compute_auc, plot_success_curve


def evaluate_sequence(
    tracker: PCSRTTracker,
    sequence: OTBSequence,
    visualize: bool = False
) -> dict:
    """
    Evaluate tracker on single sequence
    
    Returns:
        results: {
            'sequence': str,
            'ious': list,
            'avg_iou': float,
            'success_rate': float (at threshold 0.5),
            'fps': float
        }
    """
    # Initialize with first frame
    frame0, bbox0 = sequence.get_frame(0)
    tracker.initialize(frame0, bbox0)
    
    ious = []
    times = []
    
    for i in range(len(sequence)):
        frame, bbox_gt = sequence.get_frame(i)
        
        if i == 0:
            # First frame: perfect match
            bbox_pred = bbox0
            time_elapsed = 0.0
        else:
            # Track
            import time
            t_start = time.time()
            bbox_pred, confidence = tracker.track(frame)
            time_elapsed = time.time() - t_start
        
        # Compute IoU
        iou = compute_iou(bbox_pred, bbox_gt)
        ious.append(iou)
        times.append(time_elapsed)
        
        # Visualize
        if visualize:
            vis = frame.copy()
            
            # Draw GT (green)
            x, y, w, h = bbox_gt.astype(int)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis, 'GT', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw prediction (red)
            x, y, w, h = bbox_pred.astype(int)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis, f'Pred (IoU: {iou:.3f})', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.imshow(f'Tracking: {sequence.name}', vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                break
    
    if visualize:
        cv2.destroyAllWindows()
    
    # Compute metrics
    ious_array = np.array(ious)
    avg_iou = np.mean(ious_array)
    success_rate = np.mean(ious_array >= 0.5)
    fps = 1.0 / (np.mean(times[1:]) + 1e-8) if len(times) > 1 else 0.0
    
    results = {
        'sequence': sequence.name,
        'ious': ious,
        'avg_iou': avg_iou,
        'success_rate': success_rate,
        'fps': fps,
        'num_frames': len(sequence)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate PCSRT on OTB')
    parser.add_argument('--dataset-root', type=str,
                       default='../otb100/OTB-dataset/OTB100',
                       help='Path to OTB dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained checkpoint')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sequences', nargs='+', default=None,
                       help='Specific sequences to test (default: all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize tracking results')
    parser.add_argument('--save-results', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device
    save_dir = Path(args.save_results)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Create model
    print("Creating model...")
    tracker = PCSRTTracker(config).to(device)
    tracker.load_state_dict(checkpoint['model'])
    tracker.eval()
    
    # Load sequences
    print("Loading sequences...")
    dataset_root = Path(args.dataset_root)
    
    if args.sequences:
        sequence_names = args.sequences
    else:
        # All sequences
        sequence_names = [d.name for d in dataset_root.iterdir() if d.is_dir()]
    
    sequences = []
    for name in sequence_names:
        seq_path = dataset_root / name
        try:
            seq = OTBSequence(seq_path)
            sequences.append(seq)
        except Exception as e:
            print(f"Warning: Failed to load {name}: {e}")
    
    print(f"Testing on {len(sequences)} sequences")
    print("=" * 80)
    
    # Evaluate each sequence
    all_results = []
    all_ious = []
    
    for sequence in sequences:
        print(f"\nEvaluating: {sequence.name} ({len(sequence)} frames)")
        
        with torch.no_grad():
            results = evaluate_sequence(tracker, sequence, visualize=args.visualize)
        
        all_results.append(results)
        all_ious.extend(results['ious'])
        
        print(f"  Avg IoU: {results['avg_iou']:.4f}")
        print(f"  Success@0.5: {results['success_rate']:.4f}")
        print(f"  FPS: {results['fps']:.2f}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("Overall Results")
    print("=" * 80)
    
    all_ious_array = np.array(all_ious)
    overall_avg_iou = np.mean(all_ious_array)
    overall_success = np.mean(all_ious_array >= 0.5)
    overall_fps = np.mean([r['fps'] for r in all_results])
    
    print(f"Average IoU: {overall_avg_iou:.4f}")
    print(f"Success@0.5: {overall_success:.4f}")
    print(f"Average FPS: {overall_fps:.2f}")
    
    # Compute success curve
    thresholds, success_rates = compute_success_curve(all_ious_array)
    auc = compute_auc(success_rates, thresholds)
    print(f"AUC: {auc:.4f}")
    
    # Save results
    results_dict = {
        'checkpoint': str(args.checkpoint),
        'config': config.__dict__,
        'overall': {
            'avg_iou': float(overall_avg_iou),
            'success_rate': float(overall_success),
            'auc': float(auc),
            'fps': float(overall_fps)
        },
        'sequences': all_results,
        'success_curve': {
            'thresholds': thresholds.tolist(),
            'success_rates': success_rates.tolist()
        }
    }
    
    results_file = save_dir / 'pcsrt_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Plot success curve
    plot_results = {
        'PCSRT': {
            'thresholds': thresholds,
            'success_rates': success_rates,
            'auc': auc
        }
    }
    plot_success_curve(plot_results, save_path=save_dir / 'success_plot.png')
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
