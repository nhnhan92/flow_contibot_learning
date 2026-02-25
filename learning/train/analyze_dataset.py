#!/usr/bin/env python3
"""
Analyze Flowbot Dataset

Analyzes the collected dataset to verify data quality before training.
Checks:
- Episode count and lengths
- Robot TCP pose distribution
- Flowbot PWM signal distribution
- Z-axis movement patterns
- Data anomalies and outliers

Usage:
    python train/analyze_dataset.py --dataset_path data/demo_data/dataset.zarr
"""

import sys
import os
import argparse
import numpy as np
import zarr
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_dataset(dataset_path, show_plots=True):
    """Analyze dataset and print statistics"""

    print("="*80)
    print("DATASET ANALYSIS - FLOWBOT")
    print("="*80)
    print(f"\nDataset path: {dataset_path}\n")

    # Load dataset
    root = zarr.open(dataset_path, mode='r')

    # Get episode information
    episode_ends = root['meta/episode_ends'][:]
    num_episodes = len(episode_ends)
    total_timesteps = int(episode_ends[-1]) if num_episodes > 0 else 0

    print(f"{'='*80}")
    print(f"BASIC STATISTICS")
    print(f"{'='*80}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Total timesteps: {total_timesteps}")

    if num_episodes == 0:
        print("\n⚠️  No episodes found in dataset!")
        return

    # Episode lengths
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_lengths = episode_ends - episode_starts

    print(f"\nEpisode lengths:")
    print(f"  Mean: {episode_lengths.mean():.1f} steps")
    print(f"  Min:  {episode_lengths.min()} steps")
    print(f"  Max:  {episode_lengths.max()} steps")
    print(f"  Std:  {episode_lengths.std():.1f} steps")

    # Load all data
    robot_poses = root['data/robot_eef_pose'][:]   # (T, 6)
    pwm_signals = root['data/pwm_signals'][:]       # (T, 3)

    # Check if camera data exists
    has_camera = 'camera_0' in root['data']
    if has_camera:
        camera_shape = root['data/camera_0'].shape
        print(f"\nCamera data: {camera_shape}")
    else:
        print("\n⚠️  No camera data found!")

    # Analyze robot TCP poses
    print(f"\n{'='*80}")
    print(f"ROBOT TCP POSE DISTRIBUTION")
    print(f"{'='*80}")

    print(f"\nPosition range (meters):")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        min_val = robot_poses[:, i].min()
        max_val = robot_poses[:, i].max()
        range_val = max_val - min_val
        print(f"  {axis}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.4f} m)")

    print(f"\nRotation range (radians):")
    for i, axis in enumerate(['RX', 'RY', 'RZ']):
        min_val = robot_poses[:, i+3].min()
        max_val = robot_poses[:, i+3].max()
        range_val = max_val - min_val
        print(f"  {axis}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.4f})")

    # Analyze Z-axis movement per episode
    print(f"\n{'='*80}")
    print(f"Z-AXIS MOVEMENT PER EPISODE")
    print(f"{'='*80}")

    z_descents = []
    z_starts = []
    z_mins = []

    for ep_idx in range(num_episodes):
        start_idx = int(episode_starts[ep_idx])
        end_idx = int(episode_ends[ep_idx])

        ep_z = robot_poses[start_idx:end_idx, 2]
        z_start = ep_z[0]
        z_min = ep_z.min()
        z_descent = z_start - z_min

        z_descents.append(z_descent)
        z_starts.append(z_start)
        z_mins.append(z_min)

    z_descents = np.array(z_descents)
    z_starts = np.array(z_starts)
    z_mins = np.array(z_mins)

    print(f"\nStarting Z positions:")
    print(f"  Mean: {z_starts.mean():.4f} m")
    print(f"  Min:  {z_starts.min():.4f} m")
    print(f"  Max:  {z_starts.max():.4f} m")
    print(f"  Std:  {z_starts.std():.4f} m")

    print(f"\nMinimum Z positions:")
    print(f"  Mean: {z_mins.mean():.4f} m")
    print(f"  Min:  {z_mins.min():.4f} m")
    print(f"  Max:  {z_mins.max():.4f} m")
    print(f"  Std:  {z_mins.std():.4f} m")

    print(f"\nZ descent per episode:")
    print(f"  Mean: {z_descents.mean():.4f} m ({z_descents.mean()*100:.2f} cm)")
    print(f"  Min:  {z_descents.min():.4f} m ({z_descents.min()*100:.2f} cm)")
    print(f"  Max:  {z_descents.max():.4f} m ({z_descents.max()*100:.2f} cm)")
    print(f"  Std:  {z_descents.std():.4f} m ({z_descents.std()*100:.2f} cm)")

    # Analyze flowbot PWM signals
    print(f"\n{'='*80}")
    print(f"FLOWBOT PWM SIGNAL DISTRIBUTION")
    print(f"{'='*80}")

    pwm_labels = ['PWM1', 'PWM2', 'PWM3']
    print(f"\nPWM signal ranges (integer values):")
    for i, label in enumerate(pwm_labels):
        min_val = pwm_signals[:, i].min()
        max_val = pwm_signals[:, i].max()
        mean_val = pwm_signals[:, i].mean()
        print(f"  {label}: min={min_val:.0f}  max={max_val:.0f}  mean={mean_val:.1f}")

    # Check PWM activation per episode
    print(f"\nPWM activation per episode (fraction of steps with any PWM > 0):")
    for ep_idx in range(num_episodes):
        start_idx = int(episode_starts[ep_idx])
        end_idx = int(episode_ends[ep_idx])
        ep_pwm = pwm_signals[start_idx:end_idx]
        active = (ep_pwm.sum(axis=1) > 0).mean()
        print(f"  Episode {ep_idx}: {active*100:.1f}% steps with flowbot active")

    # Quality checks
    print(f"\n{'='*80}")
    print(f"DATA QUALITY CHECKS")
    print(f"{'='*80}")

    issues = []

    # Check 1: Episode length consistency
    length_std_ratio = episode_lengths.std() / episode_lengths.mean()
    if length_std_ratio > 0.5:
        issues.append(f"⚠️  High variance in episode lengths (CV={length_std_ratio:.2f})")
    else:
        print(f"✅ Episode lengths are consistent (CV={length_std_ratio:.2f})")

    # Check 2: Z descent consistency
    if z_descents.mean() > 0:
        z_descent_std_ratio = z_descents.std() / z_descents.mean()
        if z_descent_std_ratio > 0.3:
            issues.append(f"⚠️  High variance in Z descent (CV={z_descent_std_ratio:.2f})")
        else:
            print(f"✅ Z descent is consistent (CV={z_descent_std_ratio:.2f})")

    # Check 3: PWM range coverage
    pwm_max_observed = pwm_signals.max()
    if pwm_max_observed < 5:
        issues.append(f"⚠️  PWM signals very low (max={pwm_max_observed:.0f}); flowbot may not have activated")
    else:
        print(f"✅ PWM signals observed up to {pwm_max_observed:.0f}")

    # Check 4: Sufficient data
    if num_episodes < 10:
        issues.append(f"⚠️  Only {num_episodes} episodes (recommend ≥20 for good generalization)")
    else:
        print(f"✅ Sufficient episodes ({num_episodes})")

    # Print issues
    if issues:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n✅ All quality checks passed!")

    # Create visualizations
    if show_plots and num_episodes > 0:
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dataset Analysis (Flowbot): {Path(dataset_path).name}', fontsize=16)

        # Plot 1: Episode lengths
        axes[0, 0].hist(episode_lengths, bins=20, edgecolor='black')
        axes[0, 0].axvline(episode_lengths.mean(), color='r', linestyle='--',
                           label=f'Mean: {episode_lengths.mean():.1f}')
        axes[0, 0].set_xlabel('Episode Length (steps)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Z positions
        axes[0, 1].hist(robot_poses[:, 2], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(z_starts.mean(), color='g', linestyle='--',
                           label=f'Mean start: {z_starts.mean():.3f}')
        axes[0, 1].axvline(z_mins.mean(), color='r', linestyle='--',
                           label=f'Mean min: {z_mins.mean():.3f}')
        axes[0, 1].set_xlabel('Z Position (m)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Z Position Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: PWM signal distributions (all 3 channels)
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for i, (label, color) in enumerate(zip(pwm_labels, colors)):
            axes[0, 2].hist(pwm_signals[:, i], bins=30, edgecolor='black',
                            alpha=0.5, color=color, label=label)
        axes[0, 2].set_xlabel('PWM Value')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Flowbot PWM Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Z descent per episode
        axes[1, 0].bar(range(num_episodes), z_descents*100, edgecolor='black')
        axes[1, 0].axhline(z_descents.mean()*100, color='r', linestyle='--',
                           label=f'Mean: {z_descents.mean()*100:.2f} cm')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Z Descent (cm)')
        axes[1, 0].set_title('Z Descent per Episode')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: XY positions
        axes[1, 1].scatter(robot_poses[:, 0], robot_poses[:, 1], alpha=0.1, s=1)
        axes[1, 1].set_xlabel('X Position (m)')
        axes[1, 1].set_ylabel('Y Position (m)')
        axes[1, 1].set_title('XY Position Coverage')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')

        # Plot 6: PWM signals over time for episode 0
        start_idx = 0
        end_idx = int(episode_ends[0])
        ep_pwm = pwm_signals[start_idx:end_idx]
        steps = np.arange(len(ep_pwm))
        for i, (label, color) in enumerate(zip(pwm_labels, colors)):
            axes[1, 2].plot(steps, ep_pwm[:, i], color=color, label=label, alpha=0.8)
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('PWM Value')
        axes[1, 2].set_title('Flowbot PWM Trajectory (Episode 0)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = Path(dataset_path).parent / 'dataset_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_path}")

        try:
            plt.show(block=False)
        except Exception:
            pass

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze flowbot dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to zarr dataset')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset not found at {args.dataset_path}")
        return

    analyze_dataset(args.dataset_path, show_plots=not args.no_plots)


if __name__ == '__main__':
    main()
