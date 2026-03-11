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

from train.dataset import DiffusionDataset
from train.model import DiffusionPolicy


class DiffusionPolicyInference:
    """Wrapper for inference with trained Diffusion Policy"""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.config = checkpoint['config']
        # Store full checkpoint so deploy scripts can access normalization stats
        # via policy.checkpoint['state_min'], policy.checkpoint['action_range'], etc.
        self.checkpoint = checkpoint

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
            use_film_unet=self.config.get('use_film_unet', True),
            film_step_embed_dim=self.config.get('film_step_embed_dim', 256),
            film_kernel_size=self.config.get('film_kernel_size', 5),
            use_spatial_softmax=self.config.get('use_spatial_softmax', None),
            num_keypoints=self.config.get('num_keypoints', 32),
            crop_pad=self.config.get('random_crop_pad', 0),  # disabled at eval time automatically
        ).to(self.device)

        # Load weights — prefer EMA shadow weights for the UNet denoiser
        if 'ema' in checkpoint:
            ema_state = checkpoint['ema']
            # EMA is scoped to diffusion_model (UNet) only.
            # shadow dict keys are relative to diffusion_model (e.g. 'input_proj.weight').
            shadow = ema_state['shadow'] if isinstance(ema_state, dict) else ema_state
            n_loaded = 0
            for name, param in self.model.diffusion_model.named_parameters():
                if name in shadow:
                    param.data = shadow[name]
                    n_loaded += 1
            # Load live weights for vision/state encoders (not covered by EMA)
            full_state = checkpoint.get('model')
            if full_state is not None:
                encoder_state = {k: v for k, v in full_state.items()
                                 if not k.startswith('diffusion_model.')}
                self.model.load_state_dict(
                    {**self.model.state_dict(), **encoder_state}
                )
            print(f"Loaded EMA weights ({n_loaded} UNet params) + live encoder weights")
        else:
            print("Loading model weights (no EMA found)...")
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

    # Plot actions (9D: 6D robot TCP pose + 3D flowbot PWM)
    action_labels = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'PWM1', 'PWM2', 'PWM3']
    for i in range(9):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.plot(gt_actions_np[:, i], label='Ground Truth', linewidth=2)
        ax.plot(pred_actions_np[:, i], label='Predicted', linewidth=2, linestyle='--')
        ax.set_title(f'Action {action_labels[i]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)



    # Plot error heatmap
    ax = fig.add_subplot(gs[2, 1])
    error = np.abs(pred_actions_np - gt_actions_np)
    im = ax.imshow(error.T, aspect='auto', cmap='hot')
    ax.set_title('Absolute Error')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Action Dimension')
    ax.set_yticks(range(9))
    ax.set_yticklabels(action_labels)
    plt.colorbar(im, ax=ax)

        # Plot observation images
    ax = fig.add_subplot(gs[2, 2])
    img = obs_image[0].permute(1, 2, 0).numpy()
    img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title('Observation t-1')
    ax.axis('off')

    ax = fig.add_subplot(gs[2, 3])
    img = obs_image[1].permute(1, 2, 0).numpy()
    img = (img + 1) / 2
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title('Observation t')
    ax.axis('off')

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
    parser.add_argument('--dataset', type=str, default=None, help='Dataset path (uses config if not provided)')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--sample_idx', type=int, default=None, help='Sample index for visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    print("="*30)
    print("   DIFFUSION POLICY EVALUATION")
    print("="*30)

    # Load policy
    policy = DiffusionPolicyInference(args.checkpoint, device=args.device)

    # Load dataset
    dataset_path = args.dataset or policy.config['dataset_path']
    print(f"\nLoading dataset: {dataset_path}")

    dataset = DiffusionDataset(
        dataset_path=dataset_path,
        obs_horizon=policy.config['obs_horizon'],
        pred_horizon=policy.config['pred_horizon'],
        action_horizon=policy.config['action_horizon'],
        image_size=tuple(policy.config['image_size']),
        exclude_episodes=policy.config.get('exclude_episodes', []),
    )
    print(f"Total samples: {len(dataset)}")

    visualize_prediction(policy, dataset, sample_idx=args.sample_idx)


if __name__ == '__main__':
    main()
