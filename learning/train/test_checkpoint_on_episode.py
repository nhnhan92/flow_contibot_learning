#!/usr/bin/env python3
"""
Test checkpoint on a training episode to verify model predictions

Usage:
    python train/test_checkpoint_on_episode.py --checkpoint train/checkpoints/checkpoint_epoch_1000.pt --episode 5
"""

import sys
import os

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np
import matplotlib.pyplot as plt
from train.eval import DiffusionPolicyInference
from train.dataset import PickPlaceDataset


def test_checkpoint_on_episode(checkpoint_path, episode_idx=5):
    """Test model predictions on a specific training episode"""

    print("="*60)
    print("   TESTING CHECKPOINT ON TRAINING EPISODE")
    print("="*60)

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    policy = DiffusionPolicyInference(checkpoint_path)

    # Load dataset
    print("Loading dataset...")
    dataset = PickPlaceDataset(
        dataset_path=policy.config['dataset_path'],
        obs_horizon=policy.config['obs_horizon'],
        pred_horizon=policy.config['pred_horizon'],
        action_horizon=policy.config['action_horizon'],
        image_size=tuple(policy.config['image_size']),
    )

    print(f"\nDataset info:")
    print(f"  Total episodes: {len(dataset.episode_ends)}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  obs_horizon: {dataset.obs_horizon}")
    print(f"  pred_horizon: {dataset.pred_horizon}")
    print(f"  action_horizon: {dataset.action_horizon}")

    # Get episode data
    print(f"\n{'='*60}")
    print(f"Testing on Episode {episode_idx}")
    print(f"{'='*60}")

    # Find samples from this episode
    episode_start = 0 if episode_idx == 0 else dataset.episode_ends[episode_idx - 1]
    episode_end = dataset.episode_ends[episode_idx]
    episode_length = episode_end - episode_start

    print(f"\nEpisode {episode_idx}:")
    print(f"  Start index: {episode_start}")
    print(f"  End index: {episode_end}")
    print(f"  Length: {episode_length} steps")

    # Get ground truth trajectory from zarr
    poses = dataset.zarr_root['data/robot_eef_pose'][episode_start:episode_end]
    grippers = dataset.zarr_root['data/gripper_position'][episode_start:episode_end]

    print(f"\nGround truth trajectory:")
    print(f"  Z range: {poses[:, 2].min():.4f} to {poses[:, 2].max():.4f}")
    print(f"  Z descent: {poses[0, 2] - poses[:, 2].min():.4f}")
    print(f"  Gripper range: {grippers.min():.4f} to {grippers.max():.4f}")
    print(f"  Gripper closes: {'Yes' if grippers.min() < 0.5 else 'No'}")

    # Test model predictions at a few timesteps
    test_steps = [0, episode_length // 4, episode_length // 2, 3 * episode_length // 4]

    print(f"\n{'='*60}")
    print("Model Predictions vs Ground Truth")
    print(f"{'='*60}")

    for step in test_steps:
        sample_idx = episode_start + step

        # Get sample
        sample = dataset[sample_idx]
        obs_images = sample['obs_image'].unsqueeze(0).cuda()
        obs_states = sample['obs_state'].unsqueeze(0).cuda()
        gt_actions = sample['actions'].numpy()  # Ground truth actions (NORMALIZED)

        # Predict
        with torch.no_grad():
            pred_actions = policy.predict(obs_states, obs_images)
            pred_actions = pred_actions.cpu().numpy()[0]  # Remove batch dim (NORMALIZED)

        # Denormalize both for fair comparison
        gt_actions_denorm = gt_actions * dataset.action_std + dataset.action_mean
        pred_actions_denorm = pred_actions * dataset.action_std + dataset.action_mean

        # Compare first action (most important)
        first_gt = gt_actions_denorm[0]
        first_pred = pred_actions_denorm[0]

        print(f"\nStep {step}/{episode_length}:")
        print(f"  Ground Truth action:")
        print(f"    Position: [{first_gt[0]:.4f}, {first_gt[1]:.4f}, {first_gt[2]:.4f}]")
        print(f"    Gripper: {first_gt[6]:.4f}")
        print(f"  Predicted action:")
        print(f"    Position: [{first_pred[0]:.4f}, {first_pred[1]:.4f}, {first_pred[2]:.4f}]")
        print(f"    Gripper: {first_pred[6]:.4f}")

        # Calculate error
        pos_error = np.linalg.norm(first_gt[:3] - first_pred[:3])
        gripper_error = abs(first_gt[6] - first_pred[6])

        print(f"  Error:")
        print(f"    Position: {pos_error*1000:.2f} mm")
        print(f"    Gripper: {gripper_error:.4f}")

    # Rollout entire episode
    print(f"\n{'='*60}")
    print("Full Episode Rollout")
    print(f"{'='*60}")

    rollout_poses = []
    rollout_grippers = []
    gt_poses = []
    gt_grippers = []

    # Use first few samples as context
    for step in range(min(50, episode_length - dataset.pred_horizon)):
        sample_idx = episode_start + step
        sample = dataset[sample_idx]

        obs_images = sample['obs_image'].unsqueeze(0).cuda()
        obs_states = sample['obs_state'].unsqueeze(0).cuda()
        gt_actions = sample['actions'].numpy()  # NORMALIZED

        # Predict
        with torch.no_grad():
            pred_actions = policy.predict(obs_states, obs_images)
            pred_actions = pred_actions.cpu().numpy()[0]  # NORMALIZED

        # Denormalize both
        gt_actions_denorm = gt_actions * dataset.action_std + dataset.action_mean
        pred_actions_denorm = pred_actions * dataset.action_std + dataset.action_mean

        # Store predictions (first action only)
        rollout_poses.append(pred_actions_denorm[0, :3])
        rollout_grippers.append(pred_actions_denorm[0, 6])

        # Store ground truth
        gt_poses.append(gt_actions_denorm[0, :3])
        gt_grippers.append(gt_actions_denorm[0, 6])

    rollout_poses = np.array(rollout_poses)
    rollout_grippers = np.array(rollout_grippers)
    gt_poses = np.array(gt_poses)
    gt_grippers = np.array(gt_grippers)

    # Calculate metrics
    pos_errors = np.linalg.norm(rollout_poses - gt_poses, axis=1)
    gripper_errors = np.abs(rollout_grippers - gt_grippers)

    print(f"\nRollout Statistics (first 50 steps):")
    print(f"  Position error:")
    print(f"    Mean: {pos_errors.mean()*1000:.2f} mm")
    print(f"    Std: {pos_errors.std()*1000:.2f} mm")
    print(f"    Max: {pos_errors.max()*1000:.2f} mm")
    print(f"  Gripper error:")
    print(f"    Mean: {gripper_errors.mean():.4f}")
    print(f"    Std: {gripper_errors.std():.4f}")
    print(f"    Max: {gripper_errors.max():.4f}")

    # Check if prediction captures key behaviors
    print(f"\n{'='*60}")
    print("Behavior Analysis")
    print(f"{'='*60}")

    gt_z_descent = gt_poses[0, 2] - gt_poses[:, 2].min()
    pred_z_descent = rollout_poses[0, 2] - rollout_poses[:, 2].min()

    gt_gripper_closes = gt_grippers.min() < 0.5
    pred_gripper_closes = rollout_grippers.min() < 0.5

    print(f"\nZ Descent:")
    print(f"  Ground Truth: {gt_z_descent:.4f} m ({gt_z_descent*100:.1f} cm)")
    print(f"  Predicted: {pred_z_descent:.4f} m ({pred_z_descent*100:.1f} cm)")
    print(f"  Ratio: {pred_z_descent/gt_z_descent*100:.1f}%")

    print(f"\nGripper Closing:")
    print(f"  Ground Truth: {gt_grippers.min():.4f} ({'Closes' if gt_gripper_closes else 'Stays open'})")
    print(f"  Predicted: {rollout_grippers.min():.4f} ({'Closes' if pred_gripper_closes else 'Stays open'})")

    # Overall assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT")
    print(f"{'='*60}")

    if pos_errors.mean() < 0.02:  # < 2cm
        print("✅ Position predictions are GOOD (mean error < 2cm)")
    elif pos_errors.mean() < 0.05:  # < 5cm
        print("⚠️  Position predictions are OK (mean error < 5cm)")
    else:
        print("❌ Position predictions are POOR (mean error > 5cm)")

    if pred_z_descent / gt_z_descent > 0.7:
        print("✅ Z descent behavior CAPTURED (>70% of ground truth)")
    else:
        print("❌ Z descent behavior NOT CAPTURED (<70% of ground truth)")

    if pred_gripper_closes == gt_gripper_closes:
        print("✅ Gripper behavior CAPTURED")
    else:
        print("❌ Gripper behavior NOT CAPTURED")

    return {
        'pos_error_mean': pos_errors.mean(),
        'gripper_error_mean': gripper_errors.mean(),
        'z_descent_ratio': pred_z_descent / gt_z_descent,
        'gripper_correct': pred_gripper_closes == gt_gripper_closes,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--episode', type=int, default=5, help='Episode index to test')
    args = parser.parse_args()

    test_checkpoint_on_episode(args.checkpoint, args.episode)
