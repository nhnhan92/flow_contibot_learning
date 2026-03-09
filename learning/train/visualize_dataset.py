#!/usr/bin/env python3
"""
Visualize collected dataset using Rerun

Usage:
    python visualize/visualize_dataset.py --dataset data/real_data/dataset.zarr
    python visualize/visualize_dataset.py --dataset data/real_data/dataset.zarr --episode 0
"""

import argparse
import numpy as np
import zarr
import rerun as rr
from pathlib import Path

# Joint names for UR5e robot
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def print_zarr_structure(zarr_root):
    """Print the structure of the zarr dataset"""
    print("\n" + "="*60)
    print("ZARR DATASET STRUCTURE")
    print("="*60)
    print(zarr_root.tree())
    print("="*60)


def visualize_dataset(dataset_path, episode_idx=None, speed=1.0, show_structure=False):
    """Visualize dataset with Rerun"""

    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    zarr_root = zarr.open(dataset_path, mode='r')

    if show_structure:
        print_zarr_structure(zarr_root)

    # Get episode boundaries
    episode_ends = zarr_root['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    print(f"Total episodes: {n_episodes}")

    # Select episodes to visualize
    if episode_idx is not None:
        if episode_idx >= n_episodes:
            print(f"Error: Episode {episode_idx} does not exist. Max: {n_episodes-1}")
            return
        episodes = [episode_idx]
    else:
        episodes = range(n_episodes)

    # Initialize Rerun
    rr.init("PickPlace Dataset Viewer", spawn=True)

    # Visualize each episode
    for ep_idx in episodes:
        # Get episode range
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
        end_idx = episode_ends[ep_idx]
        episode_length = end_idx - start_idx

        print(f"\nEpisode {ep_idx}: {episode_length} steps")

        # Load episode data
        timestamps = zarr_root['data/timestamp'][start_idx:end_idx]
        robot_poses = zarr_root['data/robot_eef_pose'][start_idx:end_idx]
        robot_joints = zarr_root['data/robot_joint'][start_idx:end_idx]
        gripper_positions = zarr_root['data/gripper_position'][start_idx:end_idx]
        actions = zarr_root['data/action'][start_idx:end_idx]

        # Detect all camera keys automatically
        camera_keys = [key for key in zarr_root['data'].keys() if 'camera' in key.lower()]
        camera_data = {}
        for cam_key in camera_keys:
            camera_data[cam_key] = zarr_root['data'][cam_key][start_idx:end_idx]

        has_camera = len(camera_data) > 0

        # Visualize each timestep
        for i in range(episode_length):
            # Set timeline (new API - separate calls for each timeline)
            timestamp = timestamps[i] - timestamps[0]  # Relative time
            rr.set_time("step", sequence=i)
            rr.set_time("timestamp", timestamp=timestamp)

            # Current pose
            pose = robot_poses[i]
            position = pose[:3]
            orientation = pose[3:]  # Roll, pitch, yaw

            # Log robot end-effector position
            rr.log(
                f"episode_{ep_idx}/robot/end_effector",
                rr.Points3D(
                    positions=[position],
                    radii=[0.02],
                    colors=[[0, 255, 0]],  # Green
                )
            )

            # Log robot trajectory (all points up to current)
            rr.log(
                f"episode_{ep_idx}/robot/trajectory",
                rr.LineStrips3D(
                    strips=[robot_poses[:i+1, :3]],
                    colors=[[0, 255, 0, 128]],  # Semi-transparent green
                )
            )

            # Log action target position
            action_pose = actions[i]
            action_position = action_pose[:3]
            rr.log(
                f"episode_{ep_idx}/robot/action_target",
                rr.Points3D(
                    positions=[action_position],
                    radii=[0.015],
                    colors=[[255, 0, 0]],  # Red
                )
            )

            # Log gripper state (using Scalars in new API)
            gripper_value = gripper_positions[i]
            rr.log(
                f"episode_{ep_idx}/robot/gripper",
                rr.Scalars(gripper_value)
            )

            # Log gripper visual indicator (size based on openness)
            gripper_size = 0.01 + gripper_value * 0.03  # Scale 0.01 to 0.04
            rr.log(
                f"episode_{ep_idx}/robot/gripper_visual",
                rr.Points3D(
                    positions=[position + np.array([0, 0, 0.05])],  # Offset above EEF
                    radii=[gripper_size],
                    colors=[[255, 255, 0]],  # Yellow
                )
            )

            # Log joint angles with proper names
            for j_idx, joint_angle in enumerate(robot_joints[i]):
                joint_name = JOINT_NAMES[j_idx] if j_idx < len(JOINT_NAMES) else f"joint_{j_idx}"
                rr.log(
                    f"episode_{ep_idx}/joints/{joint_name}",
                    rr.Scalars(joint_angle)
                )

            # Log pose components
            rr.log(f"episode_{ep_idx}/pose/x", rr.Scalars(position[0]))
            rr.log(f"episode_{ep_idx}/pose/y", rr.Scalars(position[1]))
            rr.log(f"episode_{ep_idx}/pose/z", rr.Scalars(position[2]))
            rr.log(f"episode_{ep_idx}/pose/rx", rr.Scalars(orientation[0]))
            rr.log(f"episode_{ep_idx}/pose/ry", rr.Scalars(orientation[1]))
            rr.log(f"episode_{ep_idx}/pose/rz", rr.Scalars(orientation[2]))

            # Log all camera images
            if has_camera:
                for cam_key, images in camera_data.items():
                    img = images[i]

                    # Handle different image formats
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        # Normalize float images to uint8
                        if img.max() <= 1.0:
                            img = (img * 255.0).astype(np.uint8)
                        else:
                            img = np.clip(img, 0, 255).astype(np.uint8)

                    # Handle depth images (if single channel)
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        if len(img.shape) == 3:
                            img = img[:, :, 0]
                        # Normalize depth for visualization
                        img_normalized = img / (img.max() + 1e-6) * 255.0
                        img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
                        rr.log(f"episode_{ep_idx}/cameras/{cam_key}_depth", rr.Image(img_normalized))
                    else:
                        # RGB image
                        rr.log(f"episode_{ep_idx}/cameras/{cam_key}", rr.Image(img))

            # Log coordinate frame and workspace bounds (only once per episode)
            if i == 0:
                # World origin frame
                rr.log(
                    "world/origin",
                    rr.Arrows3D(
                        origins=[[0, 0, 0]] * 3,
                        vectors=[
                            [0.1, 0, 0],    # X axis (red)
                            [0, 0.1, 0],    # Y axis (green)
                            [0, 0, 0.1],    # Z axis (blue)
                        ],
                        colors=[
                            [255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                        ],
                    )
                )

                # Compute and visualize workspace bounds for this episode
                positions = robot_poses[:, :3]
                min_pos = positions.min(axis=0)
                max_pos = positions.max(axis=0)
                center = (min_pos + max_pos) / 2
                size = max_pos - min_pos

                # Log workspace bounding box
                rr.log(
                    f"episode_{ep_idx}/workspace/bounds",
                    rr.Points3D(
                        positions=[min_pos, max_pos, center],
                        radii=[0.01, 0.01, 0.015],
                        colors=[[100, 100, 100], [100, 100, 100], [255, 165, 0]],
                    )
                )

            # Add delay for playback speed
            if speed > 0:
                import time
                time.sleep(0.1 / speed)

        print(f"  ✅ Episode {ep_idx} visualized")

    print(f"\n✅ Visualization complete!")
    print("Rerun viewer is running. Check the viewer window.")


def main():
    parser = argparse.ArgumentParser(description='Visualize dataset with Rerun')
    parser.add_argument('--dataset', type=str, default='data/camera_demos/dataset.zarr',
                        help='Path to zarr dataset')
    parser.add_argument('--episode', type=int, default=None,
                        help='Specific episode to visualize (default: all)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--show_structure', action='store_true',
                        help='Show zarr dataset structure')
    args = parser.parse_args()

    print("="*60)
    print("   DATASET VISUALIZATION WITH RERUN")
    print("="*60)

    visualize_dataset(args.dataset, args.episode, args.speed, args.show_structure)


if __name__ == '__main__':
    main()
