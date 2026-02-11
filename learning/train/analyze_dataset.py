#!/usr/bin/env python3
"""
Analyze Pick-Place Dataset

Analyzes the collected dataset to verify data quality before training.
Checks:
- Episode count and lengths
- Robot pose distribution
- Gripper state distribution
- Z-axis movement patterns
- Data anomalies and outliers

Usage:
    python train/analyze_dataset.py --dataset_path data/real_data/dataset.zarr
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
    print("DATASET ANALYSIS")
    print("="*80)
    print(f"\nDataset path: {dataset_path}\n")

    # Load dataset
    root = zarr.open(dataset_path, 'r')

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
    print(f"  Min: {episode_lengths.min()} steps")
    print(f"  Max: {episode_lengths.max()} steps")
    print(f"  Std: {episode_lengths.std():.1f} steps")

    # Load all data
    robot_poses = root['data/robot_eef_pose'][:]
    gripper_states = root['data/gripper_position'][:]

    # Check if camera data exists
    has_camera = 'camera_0' in root['data']
    if has_camera:
        camera_shape = root['data/camera_0'].shape
        print(f"\nCamera data: {camera_shape}")
    else:
        print("\n⚠️  No camera data found!")

    # Analyze robot poses
    print(f"\n{'='*80}")
    print(f"ROBOT POSE DISTRIBUTION")
    print(f"{'='*80}")

    print(f"\nPosition range (meters):")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        min_val = robot_poses[:, i].min()
        max_val = robot_poses[:, i].max()
        range_val = max_val - min_val
        print(f"  {axis}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.4f})")

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
    print(f"  Min: {z_starts.min():.4f} m")
    print(f"  Max: {z_starts.max():.4f} m")
    print(f"  Std: {z_starts.std():.4f} m")

    print(f"\nMinimum Z positions (grasp height):")
    print(f"  Mean: {z_mins.mean():.4f} m")
    print(f"  Min: {z_mins.min():.4f} m")
    print(f"  Max: {z_mins.max():.4f} m")
    print(f"  Std: {z_mins.std():.4f} m")

    print(f"\nZ descent per episode:")
    print(f"  Mean: {z_descents.mean():.4f} m ({z_descents.mean()*100:.2f} cm)")
    print(f"  Min: {z_descents.min():.4f} m ({z_descents.min()*100:.2f} cm)")
    print(f"  Max: {z_descents.max():.4f} m ({z_descents.max()*100:.2f} cm)")
    print(f"  Std: {z_descents.std():.4f} m ({z_descents.std()*100:.2f} cm)")

    # Analyze gripper states
    print(f"\n{'='*80}")
    print(f"GRIPPER STATE DISTRIBUTION")
    print(f"{'='*80}")

    print(f"\nGripper position range: [{gripper_states.min():.4f}, {gripper_states.max():.4f}]")

    # Binary classification: closed (<0.5) vs open (>=0.5)
    closed_states = gripper_states < 0.5
    open_states = gripper_states >= 0.5

    num_closed = closed_states.sum()
    num_open = open_states.sum()
    pct_closed = 100.0 * num_closed / len(gripper_states)
    pct_open = 100.0 * num_open / len(gripper_states)

    print(f"\nGripper state distribution (threshold=0.5):")
    print(f"  Closed (<0.5): {num_closed:6d} timesteps ({pct_closed:5.1f}%)")
    print(f"  Open  (>=0.5): {num_open:6d} timesteps ({pct_open:5.1f}%)")

    # Check gripper closing per episode
    episodes_with_closing = 0
    for ep_idx in range(num_episodes):
        start_idx = int(episode_starts[ep_idx])
        end_idx = int(episode_ends[ep_idx])
        ep_gripper = gripper_states[start_idx:end_idx]
        if ep_gripper.min() < 0.5:
            episodes_with_closing += 1

    print(f"\nEpisodes with gripper closing: {episodes_with_closing}/{num_episodes} ({100.0*episodes_with_closing/num_episodes:.1f}%)")

    # Quality checks
    print(f"\n{'='*80}")
    print(f"DATA QUALITY CHECKS")
    print(f"{'='*80}")

    issues = []

    # Check 1: Gripper distribution
    if pct_closed < 20 or pct_closed > 80:
        issues.append(f"⚠️  Imbalanced gripper states: {pct_closed:.1f}% closed (ideal: 30-70%)")
    else:
        print(f"✅ Gripper distribution is balanced ({pct_closed:.1f}% closed)")

    # Check 2: Episode length consistency
    length_std_ratio = episode_lengths.std() / episode_lengths.mean()
    if length_std_ratio > 0.5:
        issues.append(f"⚠️  High variance in episode lengths (CV={length_std_ratio:.2f})")
    else:
        print(f"✅ Episode lengths are consistent (CV={length_std_ratio:.2f})")

    # Check 3: Z descent consistency
    z_descent_std_ratio = z_descents.std() / z_descents.mean()
    if z_descent_std_ratio > 0.3:
        issues.append(f"⚠️  High variance in Z descent (CV={z_descent_std_ratio:.2f})")
    else:
        print(f"✅ Z descent is consistent (CV={z_descent_std_ratio:.2f})")

    # Check 4: All episodes have gripper closing
    if episodes_with_closing < num_episodes:
        issues.append(f"⚠️  {num_episodes - episodes_with_closing} episodes missing gripper closing")
    else:
        print(f"✅ All episodes contain gripper closing")

    # Check 5: Sufficient data
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
        fig.suptitle(f'Dataset Analysis: {Path(dataset_path).name}', fontsize=16)

        # Plot 1: Episode lengths
        axes[0, 0].hist(episode_lengths, bins=20, edgecolor='black')
        axes[0, 0].axvline(episode_lengths.mean(), color='r', linestyle='--', label=f'Mean: {episode_lengths.mean():.1f}')
        axes[0, 0].set_xlabel('Episode Length (steps)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Z positions
        axes[0, 1].hist(robot_poses[:, 2], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(z_starts.mean(), color='g', linestyle='--', label=f'Mean start: {z_starts.mean():.3f}')
        axes[0, 1].axvline(z_mins.mean(), color='r', linestyle='--', label=f'Mean min: {z_mins.mean():.3f}')
        axes[0, 1].set_xlabel('Z Position (m)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Z Position Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Gripper states
        axes[0, 2].hist(gripper_states, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].axvline(0.5, color='r', linestyle='--', label='Threshold: 0.5')
        axes[0, 2].set_xlabel('Gripper Position')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Gripper State Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Z descent per episode
        axes[1, 0].bar(range(num_episodes), z_descents*100, edgecolor='black')
        axes[1, 0].axhline(z_descents.mean()*100, color='r', linestyle='--', label=f'Mean: {z_descents.mean()*100:.2f} cm')
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

        # Plot 6: Sample trajectory
        if num_episodes > 0:
            # Show first episode trajectory
            start_idx = 0
            end_idx = int(episode_ends[0])
            ep_poses = robot_poses[start_idx:end_idx]
            ep_gripper = gripper_states[start_idx:end_idx]

            # Color by gripper state
            colors = ['red' if g < 0.5 else 'green' for g in ep_gripper]
            axes[1, 2].scatter(range(len(ep_poses)), ep_poses[:, 2], c=colors, s=10, alpha=0.6)
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Z Position (m)')
            axes[1, 2].set_title(f'Sample Trajectory (Episode 0)\nRed=Closed, Green=Open')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = Path(dataset_path).parent / 'dataset_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {output_path}")

        # Don't show in non-interactive environments
        try:
            plt.show(block=False)
        except:
            pass

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze pick-place dataset')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to zarr dataset')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset not found at {args.dataset_path}")
        return

    analyze_dataset(args.dataset_path, show_plots=not args.no_plots)


if __name__ == '__main__':
    main()
