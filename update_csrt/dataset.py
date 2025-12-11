"""
OTB Dataset Loader for PCSRT Training
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from typing import List, Tuple, Optional
import random


class OTBSequence:
    """Single OTB sequence handler"""
    
    def __init__(self, sequence_path: Path):
        self.sequence_path = sequence_path
        self.name = sequence_path.name
        
        # Load groundtruth
        gt_path = sequence_path / 'groundtruth_rect.txt'
        if not gt_path.exists():
            gt_path = sequence_path / 'groundtruth.txt'
        
        self.groundtruth = self._load_groundtruth(gt_path)
        
        # Load frame paths
        img_dir = sequence_path / 'img'
        if not img_dir.exists():
            # Some sequences have images directly in sequence folder
            img_dir = sequence_path
        
        self.frame_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        
        # Verify frame count matches groundtruth
        assert len(self.frame_paths) == len(self.groundtruth), \
            f"Frame count mismatch: {len(self.frame_paths)} frames vs {len(self.groundtruth)} gt"
    
    def _load_groundtruth(self, gt_path: Path) -> np.ndarray:
        """Load groundtruth bounding boxes"""
        gt = []
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip().replace(',', ' ')
                if not line:
                    continue
                coords = [float(x) for x in line.split()]
                if len(coords) == 4:
                    gt.append(coords)  # [x, y, w, h]
                elif len(coords) == 8:
                    # Convert 4 corners to bbox
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    x, y = min(x_coords), min(y_coords)
                    w = max(x_coords) - x
                    h = max(y_coords) - y
                    gt.append([x, y, w, h])
        
        return np.array(gt, dtype=np.float32)
    
    def __len__(self):
        return len(self.frame_paths)
    
    def get_frame(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get frame and groundtruth bbox
        
        Returns:
            frame: (H, W, 3) BGR image
            bbox: (4,) [x, y, w, h]
        """
        frame = cv2.imread(str(self.frame_paths[idx]))
        bbox = self.groundtruth[idx]
        return frame, bbox


class OTBTrackingDataset(Dataset):
    """
    OTB dataset for training PCSRT
    Generates training pairs: (template, search_region, target_response)
    """
    
    def __init__(
        self,
        dataset_root: str,
        sequences: Optional[List[str]] = None,
        search_scale: float = 2.0,
        target_size: int = 224,
        exemplar_size: int = 127,
        max_frame_gap: int = 100,
        samples_per_sequence: int = 100
    ):
        self.dataset_root = Path(dataset_root)
        self.search_scale = search_scale
        self.target_size = target_size
        self.exemplar_size = exemplar_size
        self.max_frame_gap = max_frame_gap
        self.samples_per_sequence = samples_per_sequence
        
        # Load sequences
        self.sequences = self._load_sequences(sequences)
        print(f"Loaded {len(self.sequences)} sequences from OTB")
        
        # Generate training pairs
        self.training_pairs = self._generate_pairs()
        print(f"Generated {len(self.training_pairs)} training pairs")
    
    def _load_sequences(self, sequence_names: Optional[List[str]]) -> List[OTBSequence]:
        """Load OTB sequences"""
        sequences = []
        
        if sequence_names is None:
            # Load all sequences
            sequence_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        else:
            sequence_dirs = [self.dataset_root / name for name in sequence_names]
        
        for seq_dir in sequence_dirs:
            try:
                seq = OTBSequence(seq_dir)
                if len(seq) > 10:  # Skip very short sequences
                    sequences.append(seq)
            except Exception as e:
                print(f"Warning: Failed to load sequence {seq_dir.name}: {e}")
        
        return sequences
    
    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Generate training pairs: (sequence_idx, template_frame_idx, search_frame_idx)
        """
        pairs = []
        
        for seq_idx, sequence in enumerate(self.sequences):
            num_frames = len(sequence)
            
            # Sample pairs from this sequence
            for _ in range(self.samples_per_sequence):
                # Random template frame
                template_idx = random.randint(0, num_frames - 1)
                
                # Random search frame within max_frame_gap
                min_search = max(0, template_idx - self.max_frame_gap)
                max_search = min(num_frames - 1, template_idx + self.max_frame_gap)
                search_idx = random.randint(min_search, max_search)
                
                pairs.append((seq_idx, template_idx, search_idx))
        
        return pairs
    
    def _crop_and_resize(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        scale: float,
        output_size: int
    ) -> np.ndarray:
        """
        Crop region around bbox and resize
        
        Args:
            image: (H, W, 3) image
            bbox: (4,) [x, y, w, h]
            scale: scale factor for crop region
            output_size: output size (square)
        Returns:
            crop: (output_size, output_size, 3) cropped and resized image
        """
        H, W = image.shape[:2]
        x, y, w, h = bbox
        
        # Compute crop region
        cx, cy = x + w/2, y + h/2
        crop_size = max(w, h) * scale
        
        x1 = int(cx - crop_size/2)
        y1 = int(cy - crop_size/2)
        x2 = int(cx + crop_size/2)
        y2 = int(cy + crop_size/2)
        
        # Handle boundary
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - W)
        pad_bottom = max(0, y2 - H)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)
        
        # Crop
        crop = image[y1:y2, x1:x2]
        
        # Check if crop is empty (can happen with bad bboxes)
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            # Return a black image if crop failed
            return np.zeros((output_size, output_size, 3), dtype=np.uint8)
        
        # Pad if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        
        # Resize
        crop = cv2.resize(crop, (output_size, output_size))
        
        return crop
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get training sample
        
        Returns:
            sample: {
                'template': (3, 127, 127) template image,
                'search': (3, 224, 224) search region,
                'bbox': (4,) relative bbox in search region [x, y, w, h],
                'sequence': str sequence name
            }
        """
        seq_idx, template_idx, search_idx = self.training_pairs[idx]
        sequence = self.sequences[seq_idx]
        
        # Get template frame
        template_frame, template_bbox = sequence.get_frame(template_idx)
        template_crop = self._crop_and_resize(
            template_frame, template_bbox, scale=1.5, output_size=self.exemplar_size
        )
        
        # Get search frame
        search_frame, search_bbox = sequence.get_frame(search_idx)
        search_crop = self._crop_and_resize(
            search_frame, search_bbox, scale=self.search_scale, output_size=self.target_size
        )
        
        # Convert to tensor and normalize
        template_tensor = torch.from_numpy(template_crop).permute(2, 0, 1).float() / 255.0
        search_tensor = torch.from_numpy(search_crop).permute(2, 0, 1).float() / 255.0
        
        # Compute relative bbox in search region
        # (This is simplified - should account for actual transformation)
        rel_bbox = torch.tensor([0.5, 0.5, 0.3, 0.3], dtype=torch.float32)  # Placeholder
        
        return {
            'template': template_tensor,
            'search': search_tensor,
            'bbox': rel_bbox,
            'sequence': sequence.name
        }


if __name__ == '__main__':
    # Test dataset
    dataset_root = '../otb100/OTB-dataset/OTB100'
    
    if Path(dataset_root).exists():
        print("Testing OTBTrackingDataset...")
        dataset = OTBTrackingDataset(
            dataset_root=dataset_root,
            sequences=None,  # Use all sequences
            samples_per_sequence=10
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test single sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Template shape: {sample['template'].shape}")
        print(f"  Search shape: {sample['search'].shape}")
        print(f"  Bbox: {sample['bbox']}")
        print(f"  Sequence: {sample['sequence']}")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))
        print(f"\nBatch:")
        print(f"  Template batch shape: {batch['template'].shape}")
        print(f"  Search batch shape: {batch['search'].shape}")
    else:
        print(f"Dataset not found at {dataset_root}")
        print("Please update dataset_root path")
