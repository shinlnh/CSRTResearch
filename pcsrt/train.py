"""
Training script for PCSRT
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from config import PCSRTConfig
from tracker import PCSRTTracker
from dataset import OTBTrackingDataset
from loss import CompositeLoss
from utils import create_gaussian_label


def train_one_epoch(
    model: PCSRTTracker,
    dataloader: DataLoader,
    criterion: CompositeLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter
):
    """Train for one epoch"""
    model.train()
    model.feature_extractor.eval()  # Keep feature extractor frozen
    
    total_loss = 0.0
    total_peak = 0.0
    total_smooth = 0.0
    total_reg = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        search = batch['search'].to(device)  # (B, 3, 224, 224)
        B = search.size(0)
        
        # Extract features
        with torch.no_grad():
            deep_features = model.feature_extractor(search)  # (B, C, H, W)
        
        # Project to correlation space
        h_proj = model.corr_project(deep_features)  # (B, 31, H, W)
        
        # Create Gaussian target
        H, W = h_proj.shape[-2:]
        target = create_gaussian_label((H, W), sigma=2.0, device=device)
        target = target.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
        
        # Solve DCF to get h_csrt baseline
        h_csrt_list = []
        for b in range(B):
            h_csrt_b = model.dcf_solver.solve_unconstrained(
                h_proj[b:b+1], target[b:b+1]
            )
            h_csrt_list.append(h_csrt_b)
        h_csrt = torch.cat(h_csrt_list, dim=0)  # (B, 31, H, W)
        
        # Fuse with projected features
        h_final, alpha = model.hybrid_filter(h_csrt, h_proj)
        
        # Compute response
        response = torch.zeros(B, H, W, device=device)
        for b in range(B):
            response_b = model.dcf_solver.apply_filter(
                h_proj[b:b+1], h_final[b]
            )
            response[b] = response_b
        
        # Compute loss
        loss, loss_dict = criterion(response, target, model.corr_project)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total']
        total_peak += loss_dict['peak']
        total_smooth += loss_dict['smooth']
        total_reg += loss_dict['reg']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'peak': f"{loss_dict['peak']:.4f}",
            'alpha': f"{alpha.mean().item():.3f}"
        })
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('train/loss', loss_dict['total'], global_step)
        writer.add_scalar('train/peak_loss', loss_dict['peak'], global_step)
        writer.add_scalar('train/smooth_loss', loss_dict['smooth'], global_step)
        writer.add_scalar('train/reg_loss', loss_dict['reg'], global_step)
        writer.add_scalar('train/alpha_mean', alpha.mean().item(), global_step)
    
    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_peak = total_peak / n_batches
    avg_smooth = total_smooth / n_batches
    avg_reg = total_reg / n_batches
    
    return {
        'loss': avg_loss,
        'peak': avg_peak,
        'smooth': avg_smooth,
        'reg': avg_reg
    }


def main():
    parser = argparse.ArgumentParser(description='Train PCSRT')
    parser.add_argument('--dataset-root', type=str, 
                       default='../otb100/OTB-dataset/OTB100',
                       help='Path to OTB dataset')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup
    device = args.device
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Config
    config = PCSRTConfig()
    config.dataset_root = args.dataset_root
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.lr
    
    print("=" * 80)
    print("PCSRT Training Configuration")
    print("=" * 80)
    print(f"Dataset: {config.dataset_root}")
    print(f"Backbone: {config.backbone} ({config.feature_layer})")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Device: {device}")
    print(f"Alpha adaptive: {config.alpha_adaptive}")
    print("=" * 80)
    
    # Dataset
    print("\nLoading dataset...")
    dataset = OTBTrackingDataset(
        dataset_root=config.dataset_root,
        sequences=config.sequences,
        search_scale=config.search_region_scale,
        target_size=config.target_size,
        samples_per_sequence=100
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Model
    print("Creating model...")
    model = PCSRTTracker(config).to(device)
    
    # Loss
    criterion = CompositeLoss(
        lambda_peak=config.lambda_peak,
        lambda_smooth=config.lambda_smooth,
        lambda_reg=config.lambda_reg,
        use_heatmap=True
    )
    
    # Optimizer (only train CorrProject and HybridFilter)
    trainable_params = list(model.corr_project.parameters()) + \
                      list(model.hybrid_filter.parameters())
    optimizer = optim.Adam(trainable_params, lr=config.learning_rate)
    
    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch, writer
        )
        
        # Scheduler step
        scheduler.step()
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Peak: {train_metrics['peak']:.4f}")
        print(f"  Smooth: {train_metrics['smooth']:.4f}")
        print(f"  Reg: {train_metrics['reg']:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': config,
            'train_metrics': train_metrics
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / 'checkpoint_latest.pth')
        
        # Save best
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            torch.save(checkpoint, save_dir / 'checkpoint_best.pth')
            print(f"  → Saved best model (loss: {best_loss:.4f})")
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
