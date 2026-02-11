#!/usr/bin/env python3
"""
Evaluate trained Diffusion Policy model on dataset

This script evaluates the model by:
1. Running inference on validation data
2. Computing prediction errors (position, gripper)
3. Analyzing gripper state prediction accuracy
4. Visualizing predicted vs actual trajectories

Usage:
    python train/evaluate_model.py --checkpoint train/checkpoints/best_model.pt --dataset_path data/fix_pose_data/dataset.zarr
"""

import os
import sys
import argparse
import numpy as np
import torch
import zarr
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

    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")

    # Load policy
    print("\nLoading policy...")
    policy = DiffusionPolicyInference(checkpoint_path)
    config = policy.config
    device = policy.device

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

    # Load zarr to get episodes
    zarr_root = zarr.open(dataset_path, 'r')
    episode_ends = zarr_root['meta/episode_ends'][:]
    num_total_episodes = len(episode_ends)

    print(f"Total episodes: {num_total_episodes}")
    print(f"Evaluating on first {num_episodes} episodes...")

    # Evaluation metrics
    all_position_errors = []
    all_gripper_errors = []
    all_gripper_correct = []

    # Evaluate each episode
    for ep_idx in range(min(num_episodes, num_total_episodes)):
        print(f"\n{'='*80}")
        print(f"Episode {ep_idx}")
        print(f"{'='*80}")

        # Get episode range
        start_idx = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end_idx = int(episode_ends[ep_idx])
        ep_length = end_idx - start_idx

        print(f"Episode length: {ep_length} steps")
        print(f"Sample range: [{start_idx}, {end_idx})")

        # Collect predictions and ground truth
        position_errors = []
        gripper_errors = []
        gripper_correct = []

        predicted_actions_list = []
        actual_actions_list = []

        # Sample every N steps to avoid overlap
        sample_stride = config['action_horizon']

        for step in range(0, ep_length - config['pred_horizon'], sample_stride):
            sample_idx = start_idx + step

            # Get sample from dataset
            try:
                sample = dataset[sample_idx]
            except IndexError:
                break

            # Prepare inputs (dataset returns normalized tensors)
            obs_state = sample['obs_state']  # (obs_horizon, 7)
            obs_image = sample['obs_image']  # (obs_horizon, 3, H, W)
            actual_actions_normalized = sample['actions'].cpu().numpy()  # (pred_horizon, 7) - NORMALIZED

            # Predict using the policy inference class (returns normalized predictions)
            predicted_actions_normalized = policy.predict(obs_state, obs_image).numpy()  # (pred_horizon, 7)

            # Denormalize BOTH predictions and actual to compare in original scale
            # Using Min-Max denormalization: x = (x_norm + 1) * 0.5 * range + min
            predicted_actions = (predicted_actions_normalized + 1.0) * 0.5 * dataset.action_range + dataset.action_min
            actual_actions = (actual_actions_normalized + 1.0) * 0.5 * dataset.action_range + dataset.action_min

            # Compute errors
            pos_error = np.linalg.norm(predicted_actions[:, :3] - actual_actions[:, :3], axis=1)  # (pred_horizon,)
            grip_error = np.abs(predicted_actions[:, 6] - actual_actions[:, 6])  # (pred_horizon,)

            # Gripper classification accuracy (threshold at 0.5)
            pred_grip_binary = (predicted_actions[:, 6] < 0.5).astype(int)
            actual_grip_binary = (actual_actions[:, 6] < 0.5).astype(int)
            grip_correct = (pred_grip_binary == actual_grip_binary).astype(float)

            position_errors.extend(pos_error.tolist())
            gripper_errors.extend(grip_error.tolist())
            gripper_correct.extend(grip_correct.tolist())

            predicted_actions_list.append(predicted_actions)
            actual_actions_list.append(actual_actions)

        # Episode statistics
        position_errors = np.array(position_errors)
        gripper_errors = np.array(gripper_errors)
        gripper_correct = np.array(gripper_correct)

        print(f"\nPosition Error (meters):")
        print(f"  Mean: {position_errors.mean():.4f} ({position_errors.mean()*100:.2f} cm)")
        print(f"  Std:  {position_errors.std():.4f}")
        print(f"  Max:  {position_errors.max():.4f} ({position_errors.max()*100:.2f} cm)")

        print(f"\nGripper Error:")
        print(f"  Mean: {gripper_errors.mean():.4f}")
        print(f"  Std:  {gripper_errors.std():.4f}")
        print(f"  Max:  {gripper_errors.max():.4f}")

        print(f"\nGripper Classification Accuracy: {gripper_correct.mean()*100:.1f}%")

        all_position_errors.extend(position_errors.tolist())
        all_gripper_errors.extend(gripper_errors.tolist())
        all_gripper_correct.extend(gripper_correct.tolist())

        # Visualize first episode
        if ep_idx == 0:
            visualize_episode(predicted_actions_list, actual_actions_list, ep_idx)

    # Overall statistics
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS (across {num_episodes} episodes)")
    print(f"{'='*80}")

    all_position_errors = np.array(all_position_errors)
    all_gripper_errors = np.array(all_gripper_errors)
    all_gripper_correct = np.array(all_gripper_correct)

    print(f"\nPosition Error (meters):")
    print(f"  Mean: {all_position_errors.mean():.4f} ({all_position_errors.mean()*100:.2f} cm)")
    print(f"  Median: {np.median(all_position_errors):.4f} ({np.median(all_position_errors)*100:.2f} cm)")
    print(f"  Std:  {all_position_errors.std():.4f}")
    print(f"  Max:  {all_position_errors.max():.4f} ({all_position_errors.max()*100:.2f} cm)")

    print(f"\nGripper Error:")
    print(f"  Mean: {all_gripper_errors.mean():.4f}")
    print(f"  Median: {np.median(all_gripper_errors):.4f}")
    print(f"  Std:  {all_gripper_errors.std():.4f}")
    print(f"  Max:  {all_gripper_errors.max():.4f}")

    print(f"\nGripper Classification Accuracy: {all_gripper_correct.mean()*100:.1f}%")

    # Quality assessment
    print(f"\n{'='*80}")
    print(f"QUALITY ASSESSMENT")
    print(f"{'='*80}")

    mean_pos_error_cm = all_position_errors.mean() * 100
    grip_accuracy = all_gripper_correct.mean() * 100

    if mean_pos_error_cm < 1.0 and grip_accuracy > 95:
        print("✅ EXCELLENT: Model predictions are very accurate!")
    elif mean_pos_error_cm < 2.0 and grip_accuracy > 90:
        print("✅ GOOD: Model predictions are reasonably accurate")
    elif mean_pos_error_cm < 5.0 and grip_accuracy > 80:
        print("⚠️  ACCEPTABLE: Model predictions have some errors")
    else:
        print("❌ POOR: Model predictions have significant errors")
        print("   Consider collecting more data or adjusting training hyperparameters")


def visualize_episode(predicted_actions_list, actual_actions_list, ep_idx):
    """Visualize predicted vs actual trajectories"""

    # Concatenate all predictions/actuals
    predicted = np.concatenate(predicted_actions_list, axis=0)  # (N, 7)
    actual = np.concatenate(actual_actions_list, axis=0)  # (N, 7)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Episode {ep_idx}: Predicted vs Actual', fontsize=16)

    # Plot 1: XYZ positions
    ax = axes[0, 0]
    timesteps = np.arange(len(predicted))
    ax.plot(timesteps, predicted[:, 0], 'r-', label='Pred X', alpha=0.7)
    ax.plot(timesteps, actual[:, 0], 'r--', label='Actual X')
    ax.plot(timesteps, predicted[:, 1], 'g-', label='Pred Y', alpha=0.7)
    ax.plot(timesteps, actual[:, 1], 'g--', label='Actual Y')
    ax.plot(timesteps, predicted[:, 2], 'b-', label='Pred Z', alpha=0.7)
    ax.plot(timesteps, actual[:, 2], 'b--', label='Actual Z')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Gripper
    ax = axes[0, 1]
    ax.plot(timesteps, predicted[:, 6], 'r-', label='Predicted', linewidth=2)
    ax.plot(timesteps, actual[:, 6], 'b--', label='Actual', linewidth=2)
    ax.axhline(y=0.5, color='k', linestyle=':', label='Threshold')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Gripper Position')
    ax.set_title('Gripper State')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Position error over time
    ax = axes[1, 0]
    pos_error = np.linalg.norm(predicted[:, :3] - actual[:, :3], axis=1) * 100  # cm
    ax.plot(timesteps, pos_error, 'r-', linewidth=2)
    ax.axhline(y=pos_error.mean(), color='b', linestyle='--', label=f'Mean: {pos_error.mean():.2f} cm')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Position Error (cm)')
    ax.set_title('Position Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Error distribution
    ax = axes[1, 1]
    ax.hist(pos_error, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=pos_error.mean(), color='r', linestyle='--', label=f'Mean: {pos_error.mean():.2f} cm')
    ax.axvline(x=np.median(pos_error), color='g', linestyle='--', label=f'Median: {np.median(pos_error):.2f} cm')
    ax.set_xlabel('Position Error (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Position Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = f'train/evaluation_ep{ep_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found: {args.dataset_path}")
        return

    evaluate_model(args.checkpoint, args.dataset_path, args.num_episodes)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
