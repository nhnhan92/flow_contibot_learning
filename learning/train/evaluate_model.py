#!/usr/bin/env python3
"""
Evaluate trained Diffusion Policy model on dataset (Flowbot task)

This script evaluates the model by:
1. Running inference on validation data
2. Computing prediction errors (position, PWM)
3. Visualizing predicted vs actual trajectories

Usage:
    python train/evaluate_model.py --checkpoint train/checkpoints/best_model.pt --dataset_path data/demo_data/dataset.zarr
"""

import os
import sys
import argparse
import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids Tkinter threading issues
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, PROJECT_DIR)

from train.dataset import PickPlaceDataset
from train.eval import DiffusionPolicyInference


def evaluate_model(checkpoint_path, dataset_path, num_episodes=5):
    """Evaluate model on dataset"""

    print("="*30)
    print("MODEL EVALUATION - FLOWBOT")
    print("="*30)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")

    # Load policy
    print("\nLoading policy...")
    policy = DiffusionPolicyInference(checkpoint_path)
    config = policy.config
    dataset_path = dataset_path or policy.config['dataset_path']
    # Load dataset
    print("\nLoading dataset...")
    dataset = PickPlaceDataset(
        dataset_path=dataset_path,
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        action_horizon=config['action_horizon'],
        image_size=tuple(config['image_size']),
    )
    print(f"Total samples: {len(dataset)}")

    # Load zarr to get episode boundaries
    zarr_root = zarr.open(dataset_path, mode='r')
    episode_ends = zarr_root['meta/episode_ends'][:]
    num_total_episodes = len(episode_ends)

    print(f"Total episodes: {num_total_episodes}")
    print(f"Evaluating on first {num_episodes} episodes...")

    # Evaluation metrics
    all_position_errors = []
    all_pwm_errors = []     # Mean absolute error across 3 PWM channels

    # Evaluate each episode
    for ep_idx in range(min(num_episodes, num_total_episodes)):
        print(f"\n{'='*30}")
        print(f"Episode {ep_idx}")
        print(f"{'='*30}")

        start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end_idx = int(episode_ends[ep_idx])
        ep_length = end_idx - start_idx

        print(f"Episode length: {ep_length} steps")

        position_errors = []
        pwm_errors = []

        predicted_actions_list = []
        actual_actions_list = []

        sample_stride = config['action_horizon']

        for step in range(0, ep_length - config['pred_horizon'], sample_stride):
            sample_idx = start_idx + step

            try:
                sample = dataset[sample_idx]
            except IndexError:
                break

            obs_state = sample['obs_state']                               # (obs_horizon, 9)
            obs_image = sample['obs_image']                               # (obs_horizon, 3, H, W)
            actual_actions_normalized = sample['actions'].cpu().numpy()   # (pred_horizon, 9)

            predicted_actions_normalized = policy.predict(obs_state, obs_image).numpy()  # (pred_horizon, 9)

            # Denormalize both: x = (x_norm + 1) * 0.5 * range + min
            predicted_actions = (predicted_actions_normalized + 1.0) * 0.5 * dataset.action_range + dataset.action_min
            actual_actions = (actual_actions_normalized + 1.0) * 0.5 * dataset.action_range + dataset.action_min

            # Position error (first 3 dims: XYZ)
            pos_error = np.linalg.norm(predicted_actions[:, :3] - actual_actions[:, :3], axis=1)

            # PWM error: mean absolute error across 3 channels (dims 6,7,8)
            pwm_error = np.abs(predicted_actions[:, 6:] - actual_actions[:, 6:]).mean(axis=1)

            position_errors.extend(pos_error.tolist())
            pwm_errors.extend(pwm_error.tolist())

            predicted_actions_list.append(predicted_actions)
            actual_actions_list.append(actual_actions)

        position_errors = np.array(position_errors)
        pwm_errors = np.array(pwm_errors)

        print(f"\nPosition Error (meters):")
        print(f"  Mean: {position_errors.mean():.4f} ({position_errors.mean()*100:.2f} cm)")
        print(f"  Std:  {position_errors.std():.4f}")
        print(f"  Max:  {position_errors.max():.4f} ({position_errors.max()*100:.2f} cm)")

        print(f"\nFlowbot PWM Error (mean abs across 3 channels):")
        print(f"  Mean: {pwm_errors.mean():.2f}")
        print(f"  Std:  {pwm_errors.std():.2f}")
        print(f"  Max:  {pwm_errors.max():.2f}")

        all_position_errors.extend(position_errors.tolist())
        all_pwm_errors.extend(pwm_errors.tolist())

        # if ep_idx == 0:
        visualize_episode(predicted_actions_list, actual_actions_list, ep_idx)

    # Overall statistics
    print(f"\n{'='*30}")
    print(f"OVERALL STATISTICS (across {num_episodes} episodes)")
    print(f"{'='*30}")

    all_position_errors = np.array(all_position_errors)
    all_pwm_errors = np.array(all_pwm_errors)

    print(f"\nPosition Error (meters):")
    print(f"  Mean:   {all_position_errors.mean():.4f} ({all_position_errors.mean()*100:.2f} cm)")
    print(f"  Median: {np.median(all_position_errors):.4f} ({np.median(all_position_errors)*100:.2f} cm)")
    print(f"  Std:    {all_position_errors.std():.4f}")
    print(f"  Max:    {all_position_errors.max():.4f} ({all_position_errors.max()*100:.2f} cm)")

    print(f"\nFlowbot PWM Error:")
    print(f"  Mean:   {all_pwm_errors.mean():.2f}")
    print(f"  Median: {np.median(all_pwm_errors):.2f}")
    print(f"  Std:    {all_pwm_errors.std():.2f}")
    print(f"  Max:    {all_pwm_errors.max():.2f}")

    # Quality assessment
    print(f"\n{'='*30}")
    print(f"QUALITY ASSESSMENT")
    print(f"{'='*30}")

    mean_pos_error_cm = all_position_errors.mean() * 100
    mean_pwm_error = all_pwm_errors.mean()

    if mean_pos_error_cm < 1.0 and mean_pwm_error < 2.0:
        print("✅ EXCELLENT: Model predictions are very accurate!")
    elif mean_pos_error_cm < 2.0 and mean_pwm_error < 4.0:
        print("✅ GOOD: Model predictions are reasonably accurate")
    elif mean_pos_error_cm < 5.0 and mean_pwm_error < 8.0:
        print("⚠️  ACCEPTABLE: Model predictions have some errors")
    else:
        print("❌ POOR: Model predictions have significant errors")
        print("   Consider collecting more data or adjusting training hyperparameters")

def visualize_episode(predicted_actions_list, actual_actions_list, ep_idx):
    """Visualize predicted vs actual trajectories"""

    predicted = np.concatenate(predicted_actions_list, axis=0)  # (N, 9)
    actual = np.concatenate(actual_actions_list, axis=0)         # (N, 9)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Episode {ep_idx}: Predicted vs Actual', fontsize=16)

    timesteps = np.arange(len(predicted))

    # Plot 1: XYZ positions
    ax = axes[0, 0]
    ax.plot(timesteps[:300], predicted[:300, 0], 'r-', label='Pred X', alpha=0.7)
    ax.plot(timesteps[:300], actual[:300, 0],    'r--', label='Actual X')
    ax.plot(timesteps[:300], predicted[:300, 1], 'g-', label='Pred Y', alpha=0.7)
    ax.plot(timesteps[:300], actual[:300, 1],    'g--', label='Actual Y')
    ax.plot(timesteps[:300], predicted[:300, 2], 'b-', label='Pred Z', alpha=0.7)
    ax.plot(timesteps[:300], actual[:300, 2],    'b--', label='Actual Z')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position (m)')
    ax.set_title('UR5e TCP Position Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Flowbot PWM signals
    ax = axes[0, 1]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, color in enumerate(colors):
        ax.plot(timesteps, predicted[:, 6+i], color=color, linestyle='-',
                label=f'Pred PWM{i+1}', alpha=0.7)
        ax.plot(timesteps, actual[:, 6+i],    color=color, linestyle='--',
                label=f'Actual PWM{i+1}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('PWM Value')
    ax.set_title('Flowbot PWM Signals')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 3: Position error over time
    ax = axes[1, 0]
    pos_error = np.linalg.norm(predicted[:, :3] - actual[:, :3], axis=1) * 100  # cm
    ax.plot(timesteps, pos_error, 'r-', linewidth=2)
    ax.axhline(y=pos_error.mean(), color='b', linestyle='--',
               label=f'Mean: {pos_error.mean():.2f} cm')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Error (cm)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: PWM error distribution
    ax = axes[1, 1]
    pwm_error = np.abs(predicted[:, 6:] - actual[:, 6:]).mean(axis=1)
    ax.hist(pwm_error, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=pwm_error.mean(), color='r', linestyle='--',
               label=f'Mean: {pwm_error.mean():.2f}')
    ax.set_xlabel('PWM Error (mean across 3 channels)')
    ax.set_ylabel('Count')
    ax.set_title('PWM Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = f'evaluation_ep{ep_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--num_episodes', type=int, default=2, help='Number of episodes to evaluate')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found: {args.dataset_path}")
        return

    evaluate_model(args.checkpoint, args.dataset_path, args.num_episodes)

    print(f"\n{'='*30}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*30}\n")


if __name__ == '__main__':
    main()
