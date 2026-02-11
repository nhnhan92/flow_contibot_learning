#!/usr/bin/env python3
"""
Evaluation and Inference Script for Diffusion Policy

Usage:
    # Test on validation set
    python train/eval.py --checkpoint train/checkpoints/best_model.pt --mode eval

    # Generate single prediction
    python train/eval.py --checkpoint train/checkpoints/best_model.pt --mode predict
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, PROJECT_DIR)

from train.dataset import PickPlaceDataset
from train.model import DiffusionPolicy


class DiffusionPolicyInference:
    """Wrapper for inference with trained Diffusion Policy"""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.config = checkpoint['config']

        # Create model
        self.model = DiffusionPolicy(
            obs_horizon=self.config['obs_horizon'],
            pred_horizon=self.config['pred_horizon'],
            action_dim=self.config['action_dim'],
            state_dim=self.config['state_dim'],
            vision_feature_dim=self.config['vision_feature_dim'],
            state_feature_dim=self.config['state_feature_dim'],
            num_diffusion_iters=self.config['num_diffusion_iters'],
            num_inference_steps=self.config.get('num_inference_steps', 16),
            use_resnet=self.config.get('use_resnet', True),
        ).to(self.device)

        # Load weights (use EMA if available)
        if 'ema' in checkpoint:
            print("Loading EMA weights...")
            # Load EMA weights
            for name, param in self.model.named_parameters():
                if name in checkpoint['ema']:
                    param.data = checkpoint['ema'][name]
        else:
            print("Loading model weights...")
            self.model.load_state_dict(checkpoint['model'])

        self.model.eval()

        print(f"Model loaded! Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'val_loss_ema' in checkpoint:
            print(f"Validation Loss (EMA): {checkpoint['val_loss_ema']:.4f}")

    @torch.no_grad()
    def predict(self, obs_state, obs_image):
        """
        Generate action predictions

        Args:
            obs_state: (B, obs_horizon, state_dim) or (obs_horizon, state_dim)
            obs_image: (B, obs_horizon, C, H, W) or (obs_horizon, C, H, W)

        Returns:
            actions: (B, pred_horizon, action_dim) or (pred_horizon, action_dim)
        """
        # Add batch dimension if needed
        single_sample = False
        if obs_state.ndim == 2:
            obs_state = obs_state.unsqueeze(0)
            obs_image = obs_image.unsqueeze(0)
            single_sample = True

        # Move to device
        obs_state = obs_state.to(self.device)
        obs_image = obs_image.to(self.device)

        # Predict
        actions = self.model(obs_state, obs_image, train=False)

        # Remove batch dimension if single sample
        if single_sample:
            actions = actions.squeeze(0)

        return actions.cpu()


def evaluate_dataset(policy, dataset, num_samples=None):
    """Evaluate policy on dataset"""

    print("\nEvaluating on dataset...")

    # Create subset if needed
    if num_samples is not None and num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Metrics
    total_l1_error = 0
    total_l2_error = 0
    num_samples_total = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        obs_state = batch['obs_state']
        obs_image = batch['obs_image']
        gt_actions = batch['actions']

        # Predict
        pred_actions = policy.predict(obs_state, obs_image)

        # Compute errors
        l1_error = torch.abs(pred_actions - gt_actions).mean()
        l2_error = torch.sqrt(((pred_actions - gt_actions) ** 2).mean())

        total_l1_error += l1_error.item() * len(gt_actions)
        total_l2_error += l2_error.item() * len(gt_actions)
        num_samples_total += len(gt_actions)

    # Average errors
    avg_l1 = total_l1_error / num_samples_total
    avg_l2 = total_l2_error / num_samples_total

    print(f"\nResults:")
    print(f"  L1 Error: {avg_l1:.4f}")
    print(f"  L2 Error: {avg_l2:.4f}")

    return {
        'l1_error': avg_l1,
        'l2_error': avg_l2,
    }


def visualize_prediction(policy, dataset, sample_idx=None):
    """Visualize a single prediction"""

    if sample_idx is None:
        sample_idx = np.random.randint(len(dataset))

    print(f"\nVisualizing sample {sample_idx}...")

    # Get sample
    sample = dataset[sample_idx]
    obs_state = sample['obs_state']
    obs_image = sample['obs_image']
    gt_actions = sample['actions']

    # Predict
    pred_actions = policy.predict(obs_state, obs_image)

    # Convert to numpy
    obs_state_np = obs_state.numpy()
    gt_actions_np = gt_actions.numpy()
    pred_actions_np = pred_actions.numpy()

    # Plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Sample {sample_idx} - Prediction vs Ground Truth', fontsize=16)

    # Plot actions (7D: 6D robot pose + 1D gripper)
    action_labels = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'Gripper']
    for i in range(7):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.plot(gt_actions_np[:, i], label='Ground Truth', linewidth=2)
        ax.plot(pred_actions_np[:, i], label='Predicted', linewidth=2, linestyle='--')
        ax.set_title(f'Action {action_labels[i]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    # Plot observation images
    ax = fig.add_subplot(gs[1, 3])
    img = obs_image[0].permute(1, 2, 0).numpy()
    img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title('Observation t-1')
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 0])
    img = obs_image[1].permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title('Observation t')
    ax.axis('off')

    # Plot error heatmap
    ax = fig.add_subplot(gs[2, 1:3])
    error = np.abs(pred_actions_np - gt_actions_np)
    im = ax.imshow(error.T, aspect='auto', cmap='hot')
    ax.set_title('Absolute Error')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Dimension')
    ax.set_yticks(range(7))
    ax.set_yticklabels(action_labels)
    plt.colorbar(im, ax=ax)

    # Add statistics
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    stats_text = f"Statistics:\n\n"
    for i, label in enumerate(action_labels):
        mae = np.mean(np.abs(pred_actions_np[:, i] - gt_actions_np[:, i]))
        stats_text += f"{label}: {mae:.4f}\n"
    ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150)
    print("Saved: prediction_visualization.png")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'predict', 'visualize'],
                        help='Evaluation mode')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset path (uses config if not provided)')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to evaluate')
    parser.add_argument('--sample_idx', type=int, default=None, help='Sample index for visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    print("="*60)
    print("   DIFFUSION POLICY EVALUATION")
    print("="*60)

    # Load policy
    policy = DiffusionPolicyInference(args.checkpoint, device=args.device)

    if args.mode in ['eval', 'visualize']:
        # Load dataset
        dataset_path = args.dataset or policy.config['dataset_path']
        print(f"\nLoading dataset: {dataset_path}")

        dataset = PickPlaceDataset(
            dataset_path=dataset_path,
            obs_horizon=policy.config['obs_horizon'],
            pred_horizon=policy.config['pred_horizon'],
            action_horizon=policy.config['action_horizon'],
            image_size=tuple(policy.config['image_size']),
            exclude_episodes=policy.config.get('exclude_episodes', []),
        )
        print(f"Total samples: {len(dataset)}")

        if args.mode == 'eval':
            evaluate_dataset(policy, dataset, num_samples=args.num_samples)

        elif args.mode == 'visualize':
            visualize_prediction(policy, dataset, sample_idx=args.sample_idx)

    elif args.mode == 'predict':
        print("\nPrediction mode - load your own observations and call policy.predict()")
        print("Example:")
        print("  obs_state = torch.randn(2, 7)  # (obs_horizon, state_dim)")
        print("  obs_image = torch.randn(2, 3, 96, 96)  # (obs_horizon, C, H, W)")
        print("  actions = policy.predict(obs_state, obs_image)")
        print("  print(actions.shape)  # (pred_horizon, action_dim)")


if __name__ == '__main__':
    main()
