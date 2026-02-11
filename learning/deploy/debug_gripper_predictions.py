#!/usr/bin/env python3
"""
Debug gripper predictions during deployment

This script runs the model and prints out predicted gripper values
to see if the model is actually predicting gripper closing.

Usage:
    python deploy/debug_gripper_predictions.py --checkpoint train/checkpoints/best_model.pt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import cv2
from collections import deque

from train.eval import DiffusionPolicyInference
from custom.realsense_camera import RealSenseCamera
from custom.ur5e_rtde import UR5eRobot
from custom.dynamixel_gripper import DynamixelGripper


def debug_predictions(checkpoint_path, robot_ip="192.168.11.20", num_predictions=20):
    """Run model and print gripper predictions"""

    print("="*60)
    print("   DEBUG GRIPPER PREDICTIONS")
    print("="*60)

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    policy = DiffusionPolicyInference(checkpoint_path)

    # Get config
    obs_horizon = policy.config['obs_horizon']
    pred_horizon = policy.config['pred_horizon']
    action_horizon = policy.config['action_horizon']
    image_size = tuple(policy.config['image_size'])

    print(f"\nConfig:")
    print(f"  obs_horizon: {obs_horizon}")
    print(f"  pred_horizon: {pred_horizon}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  image_size: {image_size}")

    # Load normalization stats from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_mean' in checkpoint:
        state_mean = checkpoint['state_mean']
        state_std = checkpoint['state_std']
        action_mean = checkpoint['action_mean']
        action_std = checkpoint['action_std']
        print("\n✅ Loaded normalization stats from checkpoint")
        print(f"  Action mean (gripper): {action_mean[6]:.4f}")
        print(f"  Action std (gripper): {action_std[6]:.4f}")
    else:
        print("\n❌ No normalization stats in checkpoint!")
        return

    # Initialize hardware
    print(f"\nConnecting to robot at {robot_ip}...")
    robot = UR5eRobot(robot_ip)

    print("Initializing camera...")
    camera = RealSenseCamera()

    print("Initializing gripper...")
    gripper = DynamixelGripper()

    # Get current gripper position
    current_gripper = gripper.get_position()
    print(f"\n📌 Current gripper position: {current_gripper:.4f}")

    # Initialize observation buffers
    image_buffer = deque(maxlen=obs_horizon)
    state_buffer = deque(maxlen=obs_horizon)

    # Fill buffer
    print("\nInitializing observation buffer...")
    for _ in range(obs_horizon):
        # Get observation
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

    print("✅ Buffer initialized")

    # Run predictions
    print(f"\n{'='*60}")
    print(f"Running {num_predictions} predictions...")
    print(f"{'='*60}\n")

    for i in range(num_predictions):
        # Get model input
        obs_images = np.stack(list(image_buffer), axis=0)
        obs_states = np.stack(list(state_buffer), axis=0)

        obs_images = torch.from_numpy(obs_images).float().unsqueeze(0).cuda()
        obs_states = torch.from_numpy(obs_states).float().unsqueeze(0).cuda()

        # Predict
        with torch.no_grad():
            actions = policy.predict(obs_states, obs_images)
            actions = actions.cpu().numpy()[0]  # (pred_horizon, 7)

        # Denormalize
        actions_denorm = actions * action_std + action_mean

        # Extract gripper predictions
        gripper_predictions = actions_denorm[:, 6]

        # Print analysis
        print(f"Prediction {i+1}:")
        print(f"  First action gripper: {gripper_predictions[0]:.4f} ({'CLOSE' if gripper_predictions[0] < 0.5 else 'OPEN'})")
        print(f"  All {pred_horizon} gripper values: [{', '.join([f'{g:.3f}' for g in gripper_predictions])}]")
        print(f"  Min: {gripper_predictions.min():.4f}, Max: {gripper_predictions.max():.4f}, Mean: {gripper_predictions.mean():.4f}")

        # Check if any action closes gripper
        close_actions = gripper_predictions < 0.5
        if np.any(close_actions):
            first_close_idx = np.where(close_actions)[0][0]
            print(f"  ✅ Gripper closes at action {first_close_idx}: {gripper_predictions[first_close_idx]:.4f}")
        else:
            print(f"  ⚠️  Gripper stays OPEN for all {pred_horizon} actions")

        print()

        # Update buffers with new observation (simulate)
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

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Threshold for closing: < 0.5")
    print(f"Current gripper position: {current_gripper:.4f}")
    print(f"\nIf gripper predictions are all > 0.5, the model thinks gripper should stay open.")
    print(f"If gripper predictions include values < 0.5, but gripper doesn't close,")
    print(f"then there's a hardware/control issue.")

    # Cleanup
    robot.disconnect()
    camera.stop()
    gripper.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--robot_ip', type=str, default='192.168.11.20', help='Robot IP')
    parser.add_argument('--num_predictions', type=int, default=20, help='Number of predictions to run')
    args = parser.parse_args()

    debug_predictions(args.checkpoint, args.robot_ip, args.num_predictions)
