#!/usr/bin/env python3
"""
Test checkpoint on a training episode to verify model predictions (Flowbot task)

Action/State space (9D):
    - UR5e TCP pose: x, y, z, rx, ry, rz  (dims 0-5)
    - Flowbot PWM signals: pwm1, pwm2, pwm3 (dims 6-8)

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


def denormalize(data_norm, data_min, data_range):
    """Min-Max denormalization from [-1, 1] to original range"""
    return (data_norm + 1.0) * 0.5 * data_range + data_min


def test_checkpoint_on_episode(checkpoint_path, episode_idx=5):
    """Test model predictions on a specific training episode"""

    print("="*60)
    print("   TESTING CHECKPOINT ON TRAINING EPISODE - FLOWBOT")
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
    print(f"  Total samples:  {len(dataset)}")
    print(f"  obs_horizon:    {dataset.obs_horizon}")
    print(f"  pred_horizon:   {dataset.pred_horizon}")
    print(f"  action_horizon: {dataset.action_horizon}")
    print(f"  action_dim:     9 (pose 6D + PWM 3D)")

    # Get episode data
    print(f"\n{'='*60}")
    print(f"Testing on Episode {episode_idx}")
    print(f"{'='*60}")

    episode_start = 0 if episode_idx == 0 else dataset.episode_ends[episode_idx - 1]
    episode_end = dataset.episode_ends[episode_idx]
    episode_length = episode_end - episode_start

    print(f"\nEpisode {episode_idx}:")
    print(f"  Start index: {episode_start}")
    print(f"  End index:   {episode_end}")
    print(f"  Length:      {episode_length} steps")

    # Get ground truth trajectory from zarr
    poses = dataset.zarr_root['data/robot_eef_pose'][episode_start:episode_end]  # (T, 6)
    pwms  = dataset.zarr_root['data/pwm_signals'][episode_start:episode_end]      # (T, 3)

    print(f"\nGround truth trajectory:")
    print(f"  Z range:  [{poses[:, 2].min():.4f}, {poses[:, 2].max():.4f}]")
    print(f"  Z descent: {poses[0, 2] - poses[:, 2].min():.4f} m")
    print(f"  PWM1 range: [{pwms[:, 0].min():.0f}, {pwms[:, 0].max():.0f}]")
    print(f"  PWM2 range: [{pwms[:, 1].min():.0f}, {pwms[:, 1].max():.0f}]")
    print(f"  PWM3 range: [{pwms[:, 2].min():.0f}, {pwms[:, 2].max():.0f}]")

    # Test model predictions at a few timesteps
    test_steps = [0, episode_length // 4, episode_length // 2, 3 * episode_length // 4]

    print(f"\n{'='*60}")
    print("Model Predictions vs Ground Truth")
    print(f"{'='*60}")

    device = policy.device

    for step in test_steps:
        sample_idx = episode_start + step

        sample = dataset[sample_idx]
        obs_images = sample['obs_image'].unsqueeze(0).to(device)
        obs_states = sample['obs_state'].unsqueeze(0).to(device)
        gt_actions_norm = sample['actions'].numpy()   # (pred_horizon, 9) normalized

        with torch.no_grad():
            pred_actions_norm = policy.predict(obs_states, obs_images)
            pred_actions_norm = pred_actions_norm.cpu().numpy()[0]  # (pred_horizon, 9)

        # Denormalize both using Min-Max
        gt_actions   = denormalize(gt_actions_norm,   dataset.action_min, dataset.action_range)
        pred_actions = denormalize(pred_actions_norm, dataset.action_min, dataset.action_range)

        # Compare first predicted action step
        first_gt   = gt_actions[0]    # (9,)
        first_pred = pred_actions[0]  # (9,)

        print(f"\nStep {step}/{episode_length}:")
        print(f"  Ground Truth:")
        print(f"    Position: [{first_gt[0]:.4f}, {first_gt[1]:.4f}, {first_gt[2]:.4f}]")
        print(f"    PWM:      [{first_gt[6]:.1f}, {first_gt[7]:.1f}, {first_gt[8]:.1f}]")
        print(f"  Predicted:")
        print(f"    Position: [{first_pred[0]:.4f}, {first_pred[1]:.4f}, {first_pred[2]:.4f}]")
        print(f"    PWM:      [{first_pred[6]:.1f}, {first_pred[7]:.1f}, {first_pred[8]:.1f}]")

        pos_error = np.linalg.norm(first_gt[:3] - first_pred[:3])
        pwm_error = np.abs(first_gt[6:] - first_pred[6:]).mean()
        print(f"  Errors:")
        print(f"    Position: {pos_error*1000:.2f} mm")
        print(f"    PWM (MAE): {pwm_error:.2f}")

    # Rollout entire episode
    print(f"\n{'='*60}")
    print("Full Episode Rollout (first 50 steps)")
    print(f"{'='*60}")

    rollout_poses = []
    rollout_pwms  = []
    gt_poses = []
    gt_pwms  = []

    for step in range(min(50, episode_length - dataset.pred_horizon)):
        sample_idx = episode_start + step
        sample = dataset[sample_idx]

        obs_images = sample['obs_image'].unsqueeze(0).to(device)
        obs_states = sample['obs_state'].unsqueeze(0).to(device)
        gt_actions_norm = sample['actions'].numpy()

        with torch.no_grad():
            pred_actions_norm = policy.predict(obs_states, obs_images)
            pred_actions_norm = pred_actions_norm.cpu().numpy()[0]

        # Denormalize
        gt_actions   = denormalize(gt_actions_norm,   dataset.action_min, dataset.action_range)
        pred_actions = denormalize(pred_actions_norm, dataset.action_min, dataset.action_range)

        # Store first action step
        rollout_poses.append(pred_actions[0, :3])
        rollout_pwms.append(pred_actions[0, 6:])
        gt_poses.append(gt_actions[0, :3])
        gt_pwms.append(gt_actions[0, 6:])

    rollout_poses = np.array(rollout_poses)
    rollout_pwms  = np.array(rollout_pwms)
    gt_poses = np.array(gt_poses)
    gt_pwms  = np.array(gt_pwms)

    # Calculate metrics
    pos_errors = np.linalg.norm(rollout_poses - gt_poses, axis=1)
    pwm_errors = np.abs(rollout_pwms - gt_pwms).mean(axis=1)

    print(f"\nRollout Statistics:")
    print(f"  Position error:")
    print(f"    Mean: {pos_errors.mean()*1000:.2f} mm")
    print(f"    Std:  {pos_errors.std()*1000:.2f} mm")
    print(f"    Max:  {pos_errors.max()*1000:.2f} mm")
    print(f"  PWM error (MAE across 3 channels):")
    print(f"    Mean: {pwm_errors.mean():.2f}")
    print(f"    Std:  {pwm_errors.std():.2f}")
    print(f"    Max:  {pwm_errors.max():.2f}")

    # Behavior analysis
    print(f"\n{'='*60}")
    print("Behavior Analysis")
    print(f"{'='*60}")

    gt_z_descent   = gt_poses[0, 2] - gt_poses[:, 2].min()
    pred_z_descent = rollout_poses[0, 2] - rollout_poses[:, 2].min()

    gt_pwm_activated   = gt_pwms.sum(axis=1).max() > 0
    pred_pwm_activated = rollout_pwms.sum(axis=1).max() > 0

    print(f"\nZ Descent:")
    print(f"  Ground Truth: {gt_z_descent:.4f} m ({gt_z_descent*100:.1f} cm)")
    print(f"  Predicted:    {pred_z_descent:.4f} m ({pred_z_descent*100:.1f} cm)")
    if gt_z_descent > 0:
        print(f"  Ratio: {pred_z_descent/gt_z_descent*100:.1f}%")

    print(f"\nFlowbot Activation:")
    print(f"  Ground Truth: {'Active' if gt_pwm_activated else 'Not activated'}")
    print(f"  Predicted:    {'Active' if pred_pwm_activated else 'Not activated'}")

    # Overall assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT")
    print(f"{'='*60}")

    if pos_errors.mean() < 0.02:
        print("✅ Position predictions are GOOD (mean error < 2cm)")
    elif pos_errors.mean() < 0.05:
        print("⚠️  Position predictions are OK (mean error < 5cm)")
    else:
        print("❌ Position predictions are POOR (mean error > 5cm)")

    if gt_z_descent > 0 and pred_z_descent / gt_z_descent > 0.7:
        print("✅ Z descent behavior CAPTURED (>70% of ground truth)")
    else:
        print("❌ Z descent behavior NOT CAPTURED (<70% of ground truth)")

    if pred_pwm_activated == gt_pwm_activated:
        print("✅ Flowbot activation behavior CAPTURED")
    else:
        print("❌ Flowbot activation behavior NOT CAPTURED")

    if pwm_errors.mean() < 4.0:
        print("✅ PWM predictions are GOOD (MAE < 4)")
    else:
        print("⚠️  PWM predictions have significant error (MAE >= 4)")

    return {
        'pos_error_mean': pos_errors.mean(),
        'pwm_error_mean': pwm_errors.mean(),
        'z_descent_ratio': pred_z_descent / gt_z_descent if gt_z_descent > 0 else 0,
        'pwm_activation_correct': pred_pwm_activated == gt_pwm_activated,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--episode', type=int, default=5, help='Episode index to test')
    args = parser.parse_args()

    test_checkpoint_on_episode(args.checkpoint, args.episode)
