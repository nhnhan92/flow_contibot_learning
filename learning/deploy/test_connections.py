#!/usr/bin/env python3
"""
Test all robot connections before deployment

Usage:
    python deploy/test_connections.py
    python deploy/test_connections.py --robot_ip 192.168.1.102
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np

from custom.ur5e_rtde import UR5eRobot
from custom.realsense_camera import RealSenseCamera
from custom.dynamixel_gripper import DynamixelGripper


def test_robot(robot_ip):
    """Test robot connection"""
    print("\n" + "="*60)
    print("Testing Robot Connection")
    print("="*60)

    try:
        print(f"Connecting to robot at {robot_ip}...")
        robot = UR5eRobot(robot_ip)

        print("✅ Robot connected!")

        # Get current pose
        pose = robot.get_tcp_pose()
        print(f"Current TCP pose: {pose}")

        # Get joint angles
        joints = robot.get_joint_angles()
        print(f"Current joints: {joints}")

        # Test small movement
        print("\nTesting small movement...")
        response = input("Move robot slightly? [y/N]: ")

        if response.lower() == 'y':
            current_pose = robot.get_tcp_pose()
            target_pose = current_pose.copy()
            target_pose[2] += 0.05  # Move up 5cm

            print(f"Moving from Z={current_pose[2]:.3f} to Z={target_pose[2]:.3f}")
            robot.move_tcp_pose(target_pose, velocity=0.1, acceleration=0.3, asynchronous=False)

            time.sleep(1)

            # Move back
            print("Moving back...")
            robot.move_tcp_pose(current_pose, velocity=0.1, acceleration=0.3, asynchronous=False)

            print("✅ Movement test complete!")

        robot.disconnect()
        print("✅ Robot test PASSED!\n")
        return True

    except Exception as e:
        print(f"❌ Robot test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_camera(serial_number=None):
    """Test camera connection"""
    print("\n" + "="*60)
    print("Testing Camera Connection")
    print("="*60)

    try:
        print("Initializing camera...")
        camera = RealSenseCamera(serial_number=serial_number)

        print("✅ Camera initialized!")

        # Get frames
        print("Capturing frames...")
        color, depth = camera.get_frames()

        print(f"Color image shape: {color.shape}")
        print(f"Depth image shape: {depth.shape}")

        # Test multiple captures
        print("\nTesting frame rate...")
        n_frames = 30
        start_time = time.time()

        for i in range(n_frames):
            color, depth = camera.get_frames()

        elapsed = time.time() - start_time
        fps = n_frames / elapsed

        print(f"Captured {n_frames} frames in {elapsed:.2f}s")
        print(f"Frame rate: {fps:.1f} FPS")

        camera.stop()
        print("✅ Camera test PASSED!\n")
        return True

    except Exception as e:
        print(f"❌ Camera test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gripper():
    """Test gripper connection"""
    print("\n" + "="*60)
    print("Testing Gripper Connection")
    print("="*60)

    try:
        print("Connecting to gripper...")
        gripper = DynamixelGripper()

        print("✅ Gripper connected!")

        # Get current position
        pos = gripper.get_position()
        print(f"Current position: {pos:.3f}")

        # Test movement
        print("\nTesting gripper movement...")
        response = input("Test gripper open/close? [y/N]: ")

        if response.lower() == 'y':
            print("Opening gripper...")
            gripper.set_position(1.0)
            time.sleep(1)

            pos = gripper.get_position()
            print(f"Position after open: {pos:.3f}")

            print("Closing gripper...")
            gripper.set_position(0.0)
            time.sleep(1)

            pos = gripper.get_position()
            print(f"Position after close: {pos:.3f}")

            print("Opening gripper...")
            gripper.set_position(1.0)
            time.sleep(1)

            print("✅ Movement test complete!")

        gripper.disconnect()
        print("✅ Gripper test PASSED!\n")
        return True

    except Exception as e:
        print(f"❌ Gripper test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_inference_speed(checkpoint_path):
    """Test model inference speed"""
    print("\n" + "="*60)
    print("Testing Model Inference Speed")
    print("="*60)

    try:
        from train.eval import DiffusionPolicyInference
        import torch

        print(f"Loading model from: {checkpoint_path}")
        policy = DiffusionPolicyInference(checkpoint_path)

        print("✅ Model loaded!")

        # Create dummy input
        obs_horizon = policy.config['obs_horizon']
        obs_images = torch.randn(1, obs_horizon, 3, 96, 96).cuda()
        obs_states = torch.randn(1, obs_horizon, 7).cuda()

        print(f"\nRunning inference benchmark...")
        n_runs = 50

        # Warmup
        for _ in range(5):
            _ = policy.predict(obs_states, obs_images)

        # Benchmark
        start_time = time.time()
        for i in range(n_runs):
            actions = policy.predict(obs_states, obs_images)
        elapsed = time.time() - start_time

        avg_time = elapsed / n_runs
        fps = 1.0 / avg_time

        print(f"\nResults:")
        print(f"  Average inference time: {avg_time*1000:.1f} ms")
        print(f"  Max FPS: {fps:.1f}")
        print(f"  Predicted actions shape: {actions.shape}")

        if avg_time < 0.1:  # < 100ms = 10 Hz possible
            print("\n✅ Inference speed is good for 10 Hz control!")
        else:
            print(f"\n⚠️  Inference too slow for 10 Hz (need < 100ms, got {avg_time*1000:.1f}ms)")

        print("✅ Inference test PASSED!\n")
        return True

    except Exception as e:
        print(f"❌ Inference test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test robot connections')
    parser.add_argument('--robot_ip', type=str, default='192.168.11.20', help='Robot IP')
    parser.add_argument('--camera_serial', type=str, default=None, help='Camera serial')
    parser.add_argument('--checkpoint', type=str, default='train/checkpoints/best_model.pt',
                        help='Model checkpoint for inference test')
    parser.add_argument('--skip_robot', action='store_true', help='Skip robot test')
    parser.add_argument('--skip_camera', action='store_true', help='Skip camera test')
    parser.add_argument('--skip_gripper', action='store_true', help='Skip gripper test')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference test')
    args = parser.parse_args()

    print("="*60)
    print("   ROBOT DEPLOYMENT CONNECTION TEST")
    print("="*60)

    results = {}

    # Test robot
    if not args.skip_robot:
        results['robot'] = test_robot(args.robot_ip)
    else:
        print("\n⏭️  Skipping robot test")
        results['robot'] = None

    # Test camera
    if not args.skip_camera:
        results['camera'] = test_camera(args.camera_serial)
    else:
        print("\n⏭️  Skipping camera test")
        results['camera'] = None

    # Test gripper
    if not args.skip_gripper:
        results['gripper'] = test_gripper()
    else:
        print("\n⏭️  Skipping gripper test")
        results['gripper'] = None

    # Test inference
    if not args.skip_inference and os.path.exists(args.checkpoint):
        results['inference'] = test_inference_speed(args.checkpoint)
    elif not args.skip_inference:
        print(f"\n⏭️  Skipping inference test (checkpoint not found: {args.checkpoint})")
        results['inference'] = None
    else:
        print("\n⏭️  Skipping inference test")
        results['inference'] = None

    # Summary
    print("="*60)
    print("   TEST SUMMARY")
    print("="*60)

    all_passed = True
    for component, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
            all_passed = False

        print(f"{component.upper():.<20} {status}")

    print("="*60)

    if all_passed and any(results.values()):
        print("\n🎉 All tests passed! Ready for deployment!")
        print("\nNext step:")
        print("  python deploy/deploy_real_robot.py --checkpoint train/checkpoints/best_model.pt")
    elif not all_passed:
        print("\n⚠️  Some tests failed. Please fix issues before deployment.")
    else:
        print("\n⚠️  No tests were run.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
