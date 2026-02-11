#!/usr/bin/env python3
"""
Visualize predicted trajectory in 3D

This helps debug why the robot isn't reaching the grasp position.

Usage:
    python deploy/visualize_predictions.py --checkpoint train/checkpoints/best_model.pt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import cv2
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from train.eval import DiffusionPolicyInference
from custom.realsense_camera import RealSenseCamera
from custom.ur5e_rtde import UR5eRobot
from custom.dynamixel_gripper import DynamixelGripper


def visualize_predictions(checkpoint_path, robot_ip="192.168.11.20"):
    """Visualize predicted trajectory"""

    print("="*60)
    print("   VISUALIZE PREDICTED TRAJECTORY")
    print("="*60)

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    policy = DiffusionPolicyInference(checkpoint_path)

    # Get config
    obs_horizon = policy.config['obs_horizon']
    pred_horizon = policy.config['pred_horizon']
    image_size = tuple(policy.config['image_size'])

    # Load normalization stats
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_mean = checkpoint['state_mean']
    state_std = checkpoint['state_std']
    action_mean = checkpoint['action_mean']
    action_std = checkpoint['action_std']

    # Initialize hardware
    print(f"\nConnecting to robot at {robot_ip}...")
    robot = UR5eRobot(robot_ip)

    print("Initializing camera...")
    camera = RealSenseCamera()

    print("Initializing gripper...")
    gripper = DynamixelGripper()

    # Get current state
    current_pose = robot.get_tcp_pose()
    print(f"\n📌 Current robot pose:")
    print(f"   Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}]")
    print(f"   Rotation: [{current_pose[3]:.4f}, {current_pose[4]:.4f}, {current_pose[5]:.4f}]")

    # Initialize observation buffers
    image_buffer = deque(maxlen=obs_horizon)
    state_buffer = deque(maxlen=obs_horizon)

    # Fill buffer
    print("\nInitializing observation buffer...")
    for _ in range(obs_horizon):
        robot_pose = robot.get_tcp_pose()
        gripper_pos = gripper.get_position()
        state = np.concatenate([robot_pose, [gripper_pos]]).astype(np.float32)

        color_image, _ = camera.get_frames()
        image = cv2.resize(color_image, image_size)

        # Preprocess
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = np.transpose(image, (2, 0, 1))

        state = (state - state_mean) / state_std

        image_buffer.append(image.copy())
        state_buffer.append(state.copy())

    # Get prediction
    print("\nPredicting trajectory...")
    obs_images = np.stack(list(image_buffer), axis=0)
    obs_states = np.stack(list(state_buffer), axis=0)

    obs_images = torch.from_numpy(obs_images).float().unsqueeze(0).cuda()
    obs_states = torch.from_numpy(obs_states).float().unsqueeze(0).cuda()

    with torch.no_grad():
        actions = policy.predict(obs_states, obs_images)
        actions = actions.cpu().numpy()[0]  # (pred_horizon, 7)

    # Denormalize
    actions_denorm = actions * action_std + action_mean

    # Extract positions and gripper
    positions = actions_denorm[:, :3]  # (pred_horizon, 3)
    grippers = actions_denorm[:, 6]    # (pred_horizon,)

    # Print predictions
    print(f"\n{'='*60}")
    print("PREDICTED TRAJECTORY")
    print(f"{'='*60}")
    print(f"\nCurrent position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}]")
    print(f"\nPredicted positions (first 8 actions):")
    for i in range(min(8, pred_horizon)):
        gripper_state = "CLOSE" if grippers[i] < 0.5 else "OPEN"
        print(f"  Action {i}: [{positions[i, 0]:.4f}, {positions[i, 1]:.4f}, {positions[i, 2]:.4f}] | Gripper: {grippers[i]:.3f} ({gripper_state})")

    # Calculate distances
    print(f"\n{'='*60}")
    print("MOVEMENT ANALYSIS")
    print(f"{'='*60}")

    # Distance from current position to first predicted position
    dist_to_first = np.linalg.norm(positions[0] - current_pose[:3])
    print(f"\nDistance to first predicted position: {dist_to_first*100:.2f} cm")

    # Total path length
    path_lengths = [np.linalg.norm(positions[i+1] - positions[i]) for i in range(pred_horizon-1)]
    total_path_length = sum(path_lengths)
    print(f"Total predicted path length: {total_path_length*100:.2f} cm")

    # Check if going down
    z_start = current_pose[2]
    z_end = positions[-1, 2]
    z_change = z_end - z_start
    print(f"\nZ movement: {z_change*100:.2f} cm ({'DOWN' if z_change < 0 else 'UP'})")

    # Check gripper closing
    if np.any(grippers < 0.5):
        first_close_idx = np.where(grippers < 0.5)[0][0]
        print(f"\n✅ Gripper closes at action {first_close_idx}")
        print(f"   Position when closing: [{positions[first_close_idx, 0]:.4f}, {positions[first_close_idx, 1]:.4f}, {positions[first_close_idx, 2]:.4f}]")
    else:
        print(f"\n⚠️  Gripper stays OPEN for all {pred_horizon} actions")

    # Visualize in 3D
    print(f"\n{'='*60}")
    print("Creating 3D visualization...")
    print(f"{'='*60}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot current position
    ax.scatter(current_pose[0], current_pose[1], current_pose[2],
              c='green', marker='o', s=200, label='Current Position')

    # Plot predicted trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
           'b-', linewidth=2, label='Predicted Trajectory')

    # Mark positions where gripper should close
    close_indices = np.where(grippers < 0.5)[0]
    if len(close_indices) > 0:
        ax.scatter(positions[close_indices, 0],
                  positions[close_indices, 1],
                  positions[close_indices, 2],
                  c='red', marker='x', s=100, label='Gripper Closes')

    # Mark action_horizon boundary (first 8 actions)
    ax.scatter(positions[7, 0], positions[7, 1], positions[7, 2],
              c='orange', marker='s', s=150, label='Action Horizon (8)')

    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Predicted Robot Trajectory')
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([positions[:, 0].max()-positions[:, 0].min(),
                         positions[:, 1].max()-positions[:, 1].min(),
                         positions[:, 2].max()-positions[:, 2].min()]).max() / 2.0

    mid_x = (positions[:, 0].max()+positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max()+positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max()+positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig('predicted_trajectory.png', dpi=150)
    print(f"\n✅ Visualization saved to: predicted_trajectory.png")
    print("   Open this file to see the 3D trajectory!")

    # Cleanup
    robot.disconnect()
    camera.stop()
    gripper.disconnect()

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print("\nLook at the 3D plot to diagnose:")
    print("1. Is the trajectory going towards the object?")
    print("2. Is the robot descending (Z going down)?")
    print("3. Does the gripper close at the right position?")
    print("4. Is the movement distance reasonable?")
    print("\nIf trajectory looks wrong:")
    print("  → Check camera view matches training")
    print("  → Check starting position matches training")
    print("  → May need to collect more training data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--robot_ip', type=str, default='192.168.11.20', help='Robot IP')
    args = parser.parse_args()

    visualize_predictions(args.checkpoint, args.robot_ip)
