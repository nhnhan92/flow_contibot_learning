#!/usr/bin/env python3
"""
Training script for Diffusion Policy on Pick-Place task

Usage:
    python train/train.py --config train/config.yaml
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from copy import deepcopy

# Add parent directory to path
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, PROJECT_DIR)

from train.dataset import PickPlaceDataset
from train.model import DiffusionPolicy


class EMAModel:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters after evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_epoch(model, dataloader, optimizer, device, ema=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move to device
        obs_state = batch['obs_state'].to(device)
        obs_image = batch['obs_image'].to(device)
        actions = batch['actions'].to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = model(obs_state, obs_image, actions, train=True)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update()

        # Track loss
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            obs_state = batch['obs_state'].to(device)
            obs_image = batch['obs_image'].to(device)
            actions = batch['actions'].to(device)

            loss = model(obs_state, obs_image, actions, train=True)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train/config.yaml', help='Config file')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--wandb_project', type=str, default='pickplace-diffusion', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("   DIFFUSION POLICY TRAINING - PICK-PLACE")
    print("="*60)
    print(f"\nConfig: {args.config}")
    print(f"Device: {args.device}")

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    print("\nLoading dataset...")
    exclude_episodes = config.get('exclude_episodes', [])
    if exclude_episodes:
        print(f"Excluding episodes: {exclude_episodes}")

    dataset = PickPlaceDataset(
        dataset_path=config['dataset_path'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        action_horizon=config['action_horizon'],
        image_size=tuple(config['image_size']),
        exclude_episodes=exclude_episodes,
    )
    print(f"Total samples: {len(dataset)}")
    print(f"State range - min: {dataset.state_min[:3]}, max: {dataset.state_max[:3]}")
    print(f"Action range - min: {dataset.action_min[:3]}, max: {dataset.action_max[:3]}")

    # Train/val split
    val_ratio = config.get('val_ratio', 0.1)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False,
    )

    # Model
    print("\nInitializing model...")
    model = DiffusionPolicy(
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        action_dim=config['action_dim'],
        state_dim=config['state_dim'],
        vision_feature_dim=config['vision_feature_dim'],
        state_feature_dim=config['state_feature_dim'],
        num_diffusion_iters=config['num_diffusion_iters'],
        num_inference_steps=config.get('num_inference_steps', 16),
        use_resnet=config.get('use_resnet', True),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.95, 0.999),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * 0.1,
    )

    # EMA
    ema = EMAModel(model, decay=config.get('ema_decay', 0.999))

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        ema.shadow = checkpoint['ema']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # W&B
    use_wandb = config.get('use_wandb', False)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
            resume='allow' if args.resume else None,
        )
        wandb.watch(model, log='gradients', log_freq=100)

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    best_val_loss = float('inf')
    patience = 100  # Stop if no improvement for 100 epochs
    patience_counter = 0
    min_improvement = 1e-6  # Minimum improvement to reset patience

    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, ema)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Validate with EMA
        ema.apply_shadow()
        val_loss_ema = validate(model, val_loader, device)
        ema.restore()

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Loss (EMA): {val_loss_ema:.4f}")
        print(f"  LR: {current_lr:.6f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_loss_ema': val_loss_ema,
                'lr': current_lr,
            })

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.shadow,
                'config': config,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_loss_ema': val_loss_ema,
                # Save normalization stats (CRITICAL for deployment!)
                'state_min': dataset.state_min,
                'state_max': dataset.state_max,
                'state_range': dataset.state_range,
                'action_min': dataset.action_min,
                'action_max': dataset.action_max,
                'action_range': dataset.action_range,
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

        # Save best model and check for early stopping
        if val_loss_ema < best_val_loss - min_improvement:
            improvement = best_val_loss - val_loss_ema
            best_val_loss = val_loss_ema
            patience_counter = 0  # Reset counter
            best_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.shadow,
                'config': config,
                'val_loss_ema': val_loss_ema,
                # Save normalization stats (CRITICAL for deployment!)
                'state_min': dataset.state_min,
                'state_max': dataset.state_max,
                'state_range': dataset.state_range,
                'action_min': dataset.action_min,
                'action_max': dataset.action_max,
                'action_range': dataset.action_range,
            }, best_path)
            print(f"  ✅ New best! Saved: {best_path} (improved by {improvement:.6f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs.")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                break

    # Save final model
    final_path = output_dir / 'final_model.pt'
    torch.save({
        'epoch': config['num_epochs'] - 1,
        'model': model.state_dict(),
        'ema': ema.shadow,
        'config': config,
        # Save normalization stats (CRITICAL for deployment!)
        'state_min': dataset.state_min,
        'state_max': dataset.state_max,
        'state_range': dataset.state_range,
        'action_min': dataset.action_min,
        'action_max': dataset.action_max,
        'action_range': dataset.action_range,
    }, final_path)
    print(f"\n✅ Training complete! Final model: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
