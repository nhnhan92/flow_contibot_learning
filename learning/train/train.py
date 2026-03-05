#!/usr/bin/env python3
"""
Training script for Diffusion Policy on UR5e + Flowbot soft manipulator task

Usage:
    python train/train.py --config train/config.yaml
"""

import os
import sys
import math
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from copy import deepcopy
from EMA_model import EMAModel
# Add parent directory to path
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, LEARNING_DIR)

from train.dataset import PickPlaceDataset
from train.model import DiffusionPolicy

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
    parser.add_argument('--dataset', type=str, default=None, help='Dataset path')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Output directory for checkpoints and logs')
    parser.add_argument('--config', type=str, default='/config/config_train_flowbot.yaml', help='Config file')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--wandb_project', type=str, default='pickplace-diffusion', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    args = parser.parse_args()
    config_path = Path(LEARNING_DIR + args.config)
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*30)
    print("   DIFFUSION POLICY TRAINING - FLOWBOT")
    print("="*30)
    print(f"\nConfig: {config_path}")
    print(f"Device: {args.device}")

    # Create output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(LEARNING_DIR) / config['output_dir']
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
        dataset_path=args.dataset if args.dataset is not None else config['dataset_path'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        action_horizon=config['action_horizon'],
        image_size=tuple(config['image_size']),
        exclude_episodes=exclude_episodes,
        tcp_dims=config.get('tcp_dims', 3),
    )
    print(f"Total samples: {len(dataset)}")
    print(f"State  XYZ range - min: {dataset.state_min[:3]}, max: {dataset.state_max[:3]}")
    print(f"State  PWM range - min: {dataset.state_min[6:]}, max: {dataset.state_max[6:]}")
    print(f"Action XYZ range - min: {dataset.action_min[:3]}, max: {dataset.action_max[:3]}")
    print(f"Action PWM range - min: {dataset.action_min[6:]}, max: {dataset.action_max[6:]}")

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
    use_film = config.get('use_film_unet', True)
    use_ss   = config.get('use_spatial_softmax', None)   # None → auto
    n_kp     = config.get('num_keypoints', 32)
    pool_name = 'SpatialSoftmax' if (use_ss or (use_ss is None and use_film)) else 'AvgPool'
    print(f"UNet variant  : {'FiLM (ConditionalUNet1D)' if use_film else 'Simple (DiffusionUNet1D)'}")
    print(f"Vision pooling: {pool_name}" + (f" ({n_kp} kp → {n_kp*2}D/frame)" if 'Spatial' in pool_name else ''))
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
        use_film_unet=use_film,
        film_step_embed_dim=config.get('film_step_embed_dim', 256),
        film_kernel_size=config.get('film_kernel_size', 5),
        use_spatial_softmax=use_ss,
        num_keypoints=n_kp,
        crop_pad=config.get('random_crop_pad', 0),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer (AdamW with PyTorch-default betas, matching Stanford)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        betas=(0.9, 0.999),
    )

    # LR schedule: linear warmup → cosine decay
    #
    #   Epoch 0 … warmup_epochs-1 : LR linearly ramps 0 → peak_lr
    #   Epoch warmup_epochs … end : LR cosine-decays peak_lr → peak_lr * lr_min_factor
    #
    # Using LambdaLR so the multiplier is relative to the base lr set above.
    num_epochs    = config['num_epochs']
    warmup_epochs = config.get('warmup_epochs', 500)
    lr_min_factor = config.get('lr_min_factor', 0.01)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: 0/W → 1/W → ... → W/W = 1.0
            return float(epoch + 1) / float(warmup_epochs)
        else:
            # Cosine decay: 1.0 → lr_min_factor
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_factor + (1.0 - lr_min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"LR schedule   : warmup {warmup_epochs} epochs → cosine decay "
          f"(min factor {lr_min_factor})")

    # EMA — applied to UNet (diffusion_model) only, with adaptive decay
    ema_max = config.get('ema_max_value', 0.9999)
    ema = EMAModel(model.diffusion_model, max_value=ema_max)
    print(f"EMA           : UNet-only, adaptive decay → max {ema_max}")

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Full resume: epoch={start_epoch}, optimizer & scheduler restored")
        else:
            print(f"⚠️  Checkpoint has no optimizer/scheduler — weights loaded, "
                  f"optimizer starts fresh (epoch 0)")
            start_epoch = 0

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
    print("\n" + "="*30)
    print("Starting training...")
    print("="*30)

    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 200)
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
                'ema': ema.state_dict(),
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
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.state_dict(),
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
            # Don't count patience during warmup — LR is still ramping up
            if epoch < warmup_epochs:
                print(f"  No improvement (warmup phase, patience paused)")
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
