#!/usr/bin/env python3
"""
Deploy trained Diffusion Policy on real UR5e robot

Usage:
    python deploy/deploy_real_robot.py --checkpoint train/checkpoints/best_model.pt
    python deploy/deploy_real_robot.py --checkpoint train/checkpoints/best_model.pt --num_episodes 5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import cv2
import torch
from collections import deque
from pathlib import Path
import logging
from datetime import datetime

# Import robot and camera interfaces
from custom.ur5e_rtde import UR5eRobot
from custom.realsense_camera import RealSenseCamera
from custom.dynamixel_gripper import DynamixelGripper

# Import policy
from train.eval import DiffusionPolicyInference


class RobotDeployment:
    """Deploy Diffusion Policy on real robot"""

    def __init__(
        self,
        checkpoint_path,
        robot_ip="192.168.1.102",
        camera_serial=None,
        image_size=None,  # Will be loaded from checkpoint config
        control_frequency=10.0,  # Hz
        action_horizon_override=None,  # Override action_horizon from config
        verbose=True,
        log_file=None,  # Path to log file
    ):
        self.checkpoint_path = checkpoint_path
        self.control_dt = 1.0 / control_frequency
        self.verbose = verbose

        # Setup logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"deploy_log_{timestamp}.txt"
        self.log_file = log_file
        self.log_handle = open(self.log_file, 'w')
        self.log(f"{'='*60}")
        self.log(f"DEPLOYMENT LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"{'='*60}\n")

        # Load policy
        print("="*60)
        print("   DIFFUSION POLICY DEPLOYMENT")
        print("="*60)
        print(f"\nLoading policy from: {checkpoint_path}")
        self.policy = DiffusionPolicyInference(checkpoint_path)

        self.obs_horizon = self.policy.config['obs_horizon']
        self.pred_horizon = self.policy.config['pred_horizon']
        self.action_horizon = action_horizon_override if action_horizon_override is not None else self.policy.config['action_horizon']

        # Load image_size from config (CRITICAL: must match training!)
        if image_size is None:
            self.image_size = tuple(self.policy.config['image_size'])
        else:
            self.image_size = image_size
        print(f"  image_size: {self.image_size}")

        print(f"Policy loaded successfully!")
        print(f"  obs_horizon: {self.obs_horizon}")
        print(f"  pred_horizon: {self.pred_horizon}")
        if action_horizon_override is not None:
            print(f"  action_horizon: {self.action_horizon} (overridden from config: {self.policy.config['action_horizon']})")
        else:
            print(f"  action_horizon: {self.action_horizon}")

        # Load normalization stats from checkpoint (CRITICAL: use same stats as training!)
        print("\nLoading normalization stats from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Try to load stats from checkpoint (new checkpoints with min-max normalization)
        if 'action_min' in checkpoint:
            self.state_min = checkpoint['state_min']
            self.state_max = checkpoint['state_max']
            self.state_range = checkpoint['state_range']
            self.action_min = checkpoint['action_min']
            self.action_max = checkpoint['action_max']
            self.action_range = checkpoint['action_range']
            print("  ✅ Loaded Min-Max normalization stats from checkpoint!")
        elif 'state_mean' in checkpoint:
            # Old checkpoints with z-score normalization (deprecated)
            print("  ⚠️  WARNING: Checkpoint uses old Z-score normalization!")
            print("  ⚠️  Please retrain with new Min-Max normalization for better performance!")
            self.state_mean = checkpoint['state_mean']
            self.state_std = checkpoint['state_std']
            self.action_mean = checkpoint['action_mean']
            self.action_std = checkpoint['action_std']
            # Set flags to use old normalization
            self.use_old_normalization = True
        else:
            # Fallback: compute from dataset (old checkpoints without stats)
            print("  ⚠️  Checkpoint missing stats, computing from dataset...")
            from train.dataset import PickPlaceDataset
            temp_dataset = PickPlaceDataset(
                dataset_path=self.policy.config['dataset_path'],
                obs_horizon=self.obs_horizon,
                pred_horizon=self.pred_horizon,
                action_horizon=self.action_horizon,
                image_size=tuple(self.policy.config['image_size']),
                exclude_episodes=self.policy.config.get('exclude_episodes', []),
            )
            self.state_min = temp_dataset.state_min
            self.state_max = temp_dataset.state_max
            self.state_range = temp_dataset.state_range
            self.action_min = temp_dataset.action_min
            self.action_max = temp_dataset.action_max
            self.action_range = temp_dataset.action_range
            del temp_dataset  # Free memory
            print("  ⚠️  Stats may differ from training!")

        print(f"  Action range: [{self.action_min[0]:.4f}, {self.action_max[0]:.4f}] (X)")
        print(f"                [{self.action_min[1]:.4f}, {self.action_max[1]:.4f}] (Y)")
        print(f"                [{self.action_min[2]:.4f}, {self.action_max[2]:.4f}] (Z)")

        # Initialize robot
        print(f"\nConnecting to robot at {robot_ip}...")
        self.robot = UR5eRobot(robot_ip)
        print("✅ Robot connected!")

        # Initialize camera
        print("\nInitializing camera...")
        self.camera = RealSenseCamera(serial_number=camera_serial)
        print("✅ Camera initialized!")

        # Initialize gripper
        print("\nInitializing gripper...")
        self.gripper = DynamixelGripper()
        print("✅ Gripper initialized!")

        # Observation buffers (store last obs_horizon observations)
        self.image_buffer = deque(maxlen=self.obs_horizon)
        self.state_buffer = deque(maxlen=self.obs_horizon)

        # Action queue for temporal smoothing
        self.action_queue = deque()

        print("\n" + "="*60)
        print("   DEPLOYMENT READY!")
        print("="*60)

    def log(self, message, print_to_console=True):
        """Log message to file and optionally console"""
        self.log_handle.write(message + '\n')
        self.log_handle.flush()
        if print_to_console:
            print(message)

    def get_observation(self):
        """Get current observation from robot and camera"""
        # Get robot state
        robot_pose = self.robot.get_tcp_pose()  # (6,) [x, y, z, rx, ry, rz]
        gripper_pos = self.gripper.get_position()  # scalar [0, 1]

        # Combine into state (7D)
        state = np.concatenate([robot_pose, [gripper_pos]]).astype(np.float32)

        # Get camera image
        color_image, depth_image = self.camera.get_frames()

        # Center crop then resize (matches training preprocessing)
        h, w = color_image.shape[:2]
        target_h, target_w = self.image_size

        # Calculate crop boundaries for center crop
        crop_h = min(h, int(target_h * 1.5))
        crop_w = min(w, int(target_w * 1.5))

        # Center crop
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        image_cropped = color_image[start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Resize to target size
        image = cv2.resize(image_cropped, (target_w, target_h))

        return image, state

    def preprocess_observation(self, image, state):
        """Preprocess observation for model input"""
        # Image: resize and normalize to [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # State: normalize using Min-Max to [-1, 1]
        # x_norm = 2 * (x - min) / range - 1
        state = 2.0 * (state - self.state_min) / self.state_range - 1.0

        return image, state

    def initialize_observation_buffer(self):
        """Fill observation buffer with current observations"""
        print("\nInitializing observation buffer...")

        image, state = self.get_observation()
        image, state = self.preprocess_observation(image, state)

        # Fill buffer with same observation
        for _ in range(self.obs_horizon):
            self.image_buffer.append(image.copy())
            self.state_buffer.append(state.copy())

        print("✅ Observation buffer initialized!")

    def get_model_input(self):
        """Get model input from observation buffers"""
        # Stack observations
        obs_images = np.stack(list(self.image_buffer), axis=0)  # (obs_horizon, 3, H, W)
        obs_states = np.stack(list(self.state_buffer), axis=0)  # (obs_horizon, 7)

        # Convert to torch tensors
        obs_images = torch.from_numpy(obs_images).float().unsqueeze(0)  # (1, obs_horizon, 3, H, W)
        obs_states = torch.from_numpy(obs_states).float().unsqueeze(0)  # (1, obs_horizon, 7)

        return obs_states, obs_images

    def predict_actions(self):
        """Predict actions using the policy"""
        obs_states, obs_images = self.get_model_input()

        # Predict (returns NORMALIZED actions from model)
        with torch.no_grad():
            actions = self.policy.predict(obs_states, obs_images)  # (1, pred_horizon, 7)

        # Convert to numpy and remove batch dimension
        actions = actions.cpu().numpy()
        if len(actions.shape) == 3:
            actions = actions[0]  # Remove batch dimension: (pred_horizon, 7)

        # Denormalize actions (Min-Max from [-1, 1] to original range)
        # x = (x_norm + 1) * 0.5 * range + min
        actions = (actions + 1.0) * 0.5 * self.action_range + self.action_min

        return actions

    def execute_action(self, action):
        """Execute a single action on the robot"""
        # Extract pose and gripper
        target_pose = action[:6]  # [x, y, z, rx, ry, rz]
        target_gripper = action[6]  # [0, 1]

        # Send commands
        # Use servoL for smooth absolute pose control (correct for Diffusion Policy!)
        self.robot.servo_tcp_pose(target_pose, dt=self.control_dt, lookahead_time=0.1, gain=300)
        self.gripper.set_position(target_gripper)

        if self.verbose:
            print(f"  → Pose: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}, "
                  f"{target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}]")
            print(f"  → Gripper: {target_gripper:.3f}")

    def run_episode(self, max_steps=250):
        """Run one episode of deployment"""
        print("\n" + "="*60)
        print("   STARTING EPISODE")
        print("="*60)

        # Wait for user confirmation
        input("\nPress Enter to start episode (Ctrl+C to abort)...")

        # Initialize observation buffer
        self.initialize_observation_buffer()

        # Get starting position
        start_pose = self.robot.get_tcp_pose()
        start_gripper = self.gripper.get_position()

        self.log(f"\n{'='*60}")
        self.log(f"EPISODE START")
        self.log(f"{'='*60}")
        self.log(f"Starting position:")
        self.log(f"  X: {start_pose[0]:.4f}, Y: {start_pose[1]:.4f}, Z: {start_pose[2]:.4f}")
        self.log(f"  Rotation: [{start_pose[3]:.4f}, {start_pose[4]:.4f}, {start_pose[5]:.4f}]")
        self.log(f"  Gripper: {start_gripper:.4f}")
        self.log(f"")

        # Clear action queue
        self.action_queue.clear()

        step = 0
        iter_idx = 0
        episode_start_time = time.time()

        # Track trajectory for analysis
        trajectory_positions = [start_pose[:3].copy()]
        trajectory_grippers = [start_gripper]
        prediction_count = 0

        print(f"\nRunning episode (max {max_steps} steps)...")
        print("Press Ctrl+C to stop early\n")

        try:
            while step < max_steps:
                # Calculate timing for this iteration
                # Following original diffusion_policy: predict every action_horizon steps
                t_cycle_end = episode_start_time + (iter_idx + self.action_horizon) * self.control_dt

                # Prediction timing
                inference_start = time.time()

                if self.verbose:
                    print(f"\n[Step {step}] Predicting actions...")

                # Get current observation with timestamp
                obs_timestamp = time.time()

                # Predict new action sequence
                actions = self.predict_actions()  # (pred_horizon, 7)

                inference_time = time.time() - inference_start
                if self.verbose:
                    print(f"  Inference time: {inference_time*1000:.1f}ms")

                # Log detailed prediction info for first few predictions
                if prediction_count < 3:
                    current_pos = self.robot.get_tcp_pose()
                    self.log(f"\n--- PREDICTION #{prediction_count + 1} at Step {step} ---")
                    self.log(f"Current position: [{current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}]")
                    self.log(f"\nFirst 8 predicted actions:")
                    for i in range(min(8, len(actions))):
                        pos = actions[i, :3]
                        grip = actions[i, 6]
                        grip_state = "CLOSE" if grip < 0.5 else "OPEN"
                        self.log(f"  Action {i}: Pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] | Grip={grip:.3f} ({grip_state})")

                    # Analyze movement
                    first_target = actions[0, :3]
                    dist_to_first = np.linalg.norm(first_target - current_pos[:3])
                    z_change = actions[-1, 2] - current_pos[2]

                    self.log(f"\nMovement analysis:")
                    self.log(f"  Distance to first action: {dist_to_first*100:.2f} cm")
                    self.log(f"  Z change over {self.pred_horizon} actions: {z_change*100:.2f} cm")

                    # Check if gripper closes
                    close_indices = np.where(actions[:, 6] < 0.5)[0]
                    if len(close_indices) > 0:
                        self.log(f"  ✅ Gripper closes at action {close_indices[0]}")
                    else:
                        self.log(f"  ⚠️  Gripper stays OPEN for all {self.pred_horizon} actions")
                    self.log("")

                prediction_count += 1

                # Calculate action timestamps (following original diffusion_policy approach)
                # Actions are scheduled relative to observation timestamp
                action_timestamps = obs_timestamp + np.arange(len(actions)) * self.control_dt

                # Filter stale actions (key innovation from original diffusion_policy!)
                action_exec_latency = 0.01  # 10ms execution latency
                curr_time = time.time()
                is_new = action_timestamps > (curr_time + action_exec_latency)

                if np.sum(is_new) == 0:
                    # All actions are stale! Execute only the last one
                    actions_to_execute = actions[[-1]]
                    n_executed = 1
                    if self.verbose:
                        print(f"  ⚠️  All actions stale! Executing last action only")
                else:
                    # Execute only fresh actions
                    actions_to_execute = actions[is_new]
                    n_executed = len(actions_to_execute)
                    n_stale = len(actions) - n_executed
                    if self.verbose and n_stale > 0:
                        print(f"  Filtered {n_stale} stale actions, executing {n_executed} fresh actions")

                if self.verbose:
                    print(f"  Predicted {self.pred_horizon} actions, executing {n_executed}")

                # Execute actions sequentially
                for i, action in enumerate(actions_to_execute):
                    if self.verbose and i == 0:
                        print(f"\n[Step {step}] Executing action...")

                    self.execute_action(action)

                    # Wait for control frequency
                    step_start_time = time.time()
                    time.sleep(self.control_dt)

                    # Get new observation and update buffers
                    image, state = self.get_observation()
                    image, state = self.preprocess_observation(image, state)

                    self.image_buffer.append(image)
                    self.state_buffer.append(state)

                    # Track trajectory
                    current_pose = self.robot.get_tcp_pose()
                    current_gripper = self.gripper.get_position()
                    trajectory_positions.append(current_pose[:3].copy())
                    trajectory_grippers.append(current_gripper)

                    step += 1

                    # Print step summary
                    if step % 10 == 0:
                        elapsed_time = time.time() - episode_start_time
                        print(f"\n{'='*60}")
                        print(f"Step {step}/{max_steps} | Time: {elapsed_time:.1f}s | Hz: {step/elapsed_time:.1f}")
                        print(f"{'='*60}")

                    # Log summary every 50 steps
                    if step % 50 == 0:
                        dist_from_start = np.linalg.norm(current_pose[:3] - start_pose[:3])
                        z_change = current_pose[2] - start_pose[2]
                        self.log(f"\n[Step {step}] Progress:", print_to_console=False)
                        self.log(f"  Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}]", print_to_console=False)
                        self.log(f"  Distance from start: {dist_from_start*100:.2f} cm", print_to_console=False)
                        self.log(f"  Z change: {z_change*100:.2f} cm", print_to_console=False)
                        self.log(f"  Gripper: {current_gripper:.3f}", print_to_console=False)

                # Update iteration counter (predict every action_horizon steps)
                iter_idx += self.action_horizon

        except KeyboardInterrupt:
            print("\n\n⚠️  Episode interrupted by user!")

        finally:
            # Stop robot
            print("\nStopping robot...")
            self.robot.stop()

            total_time = time.time() - episode_start_time

            # Calculate final statistics
            final_pose = self.robot.get_tcp_pose()
            final_gripper = self.gripper.get_position()

            trajectory_positions = np.array(trajectory_positions)
            trajectory_grippers = np.array(trajectory_grippers)

            total_distance = np.sum([np.linalg.norm(trajectory_positions[i+1] - trajectory_positions[i])
                                     for i in range(len(trajectory_positions)-1)])
            dist_from_start = np.linalg.norm(final_pose[:3] - start_pose[:3])
            z_change_total = final_pose[2] - start_pose[2]

            # Log final summary
            self.log(f"\n{'='*60}")
            self.log(f"EPISODE COMPLETE")
            self.log(f"{'='*60}")
            self.log(f"Total steps: {step}")
            self.log(f"Total time: {total_time:.1f}s")
            self.log(f"Average Hz: {step/total_time:.1f}")
            self.log(f"")
            self.log(f"Movement Summary:")
            self.log(f"  Starting pos: [{start_pose[0]:.4f}, {start_pose[1]:.4f}, {start_pose[2]:.4f}]")
            self.log(f"  Final pos:    [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}]")
            self.log(f"  Total path distance: {total_distance*100:.2f} cm")
            self.log(f"  Straight-line distance: {dist_from_start*100:.2f} cm")
            self.log(f"  Z change: {z_change_total*100:.2f} cm")
            self.log(f"")
            self.log(f"Gripper Summary:")
            self.log(f"  Starting: {start_gripper:.3f}")
            self.log(f"  Final: {final_gripper:.3f}")
            self.log(f"  Min value: {trajectory_grippers.min():.3f}")
            self.log(f"  Max value: {trajectory_grippers.max():.3f}")
            if trajectory_grippers.min() < 0.5:
                self.log(f"  ✅ Gripper closed during episode")
            else:
                self.log(f"  ⚠️  Gripper stayed OPEN (all values > 0.5)")
            self.log(f"")
            self.log(f"Number of predictions: {prediction_count}")
            self.log(f"")

            print(f"\n{'='*60}")
            print(f"   EPISODE COMPLETE")
            print(f"{'='*60}")
            print(f"Total steps: {step}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Average Hz: {step/total_time:.1f}")
            print(f"\n📋 Log saved to: {self.log_file}")

    def move_to_start_position(self, start_pose=None):
        """Move robot to starting position"""
        if start_pose is None:
            # Default start position (current robot position)
            start_pose = [0.0521, -0.3485, 0.4590, 3.1028, 0.0172, -0.1296]  # [x, y, z, rx, ry, rz]

        print(f"\nMoving to start position: {start_pose}")
        self.robot.move_tcp_pose(start_pose, velocity=0.1, acceleration=0.3, asynchronous=False)
        self.gripper.set_position(1.0)  # Open gripper
        print("✅ At start position!")

    def shutdown(self):
        """Clean shutdown"""
        print("\nShutting down...")
        self.robot.disconnect()
        self.camera.stop()
        self.gripper.disconnect()

        # Close log file
        if hasattr(self, 'log_handle') and self.log_handle:
            self.log_handle.close()

        print("✅ Shutdown complete!")


def main():
    parser = argparse.ArgumentParser(description='Deploy Diffusion Policy on real robot')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--robot_ip', type=str, default='192.168.11.20', help='Robot IP address')
    parser.add_argument('--camera_serial', type=str, default=None, help='Camera serial number')
    parser.add_argument('--num_episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=250, help='Max steps per episode')
    parser.add_argument('--frequency', type=float, default=10.0, help='Control frequency (Hz)')
    parser.add_argument('--action_horizon', type=int, default=6, help='Override action_horizon (default: 6, following original diffusion_policy)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file (default: deploy_log_TIMESTAMP.txt)')
    args = parser.parse_args()

    # Initialize deployment
    deployment = RobotDeployment(
        checkpoint_path=args.checkpoint,
        robot_ip=args.robot_ip,
        camera_serial=args.camera_serial,
        control_frequency=args.frequency,
        action_horizon_override=args.action_horizon,
        verbose=not args.quiet,
        log_file=args.log_file,
    )

    try:
        # Move to start position
        deployment.move_to_start_position()

        # Run episodes
        for episode in range(args.num_episodes):
            print(f"\n{'='*60}")
            print(f"   EPISODE {episode + 1}/{args.num_episodes}")
            print(f"{'='*60}")

            deployment.run_episode(max_steps=args.max_steps)

            if episode < args.num_episodes - 1:
                # Ask if user wants to continue
                response = input("\nContinue to next episode? [Y/n]: ")
                if response.lower() == 'n':
                    print("Stopping deployment.")
                    break

                # Return to start
                deployment.move_to_start_position()

    except Exception as e:
        print(f"\n❌ Error during deployment: {e}")
        import traceback
        traceback.print_exc()

    finally:
        deployment.shutdown()

    print("\n" + "="*60)
    print("   DEPLOYMENT FINISHED")
    print("="*60)


if __name__ == '__main__':
    main()
