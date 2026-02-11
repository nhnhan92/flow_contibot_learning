#!/usr/bin/env python3
"""
Move robot to start position

Usage:
    python deploy/move_to_start.py
    python deploy/move_to_start.py --robot_ip 192.168.11.20
    python deploy/move_to_start.py --custom_pose 0.1 -0.3 0.4 3.14 0 0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from custom.ur5e_rtde import UR5eRobot
from custom.dynamixel_gripper import DynamixelGripper


# Default start position (from deployment script)
DEFAULT_START_POSE = [0.0521, -0.3485, 0.4590, 3.1028, 0.0172, -0.1296]


def move_to_start(robot_ip, start_pose, velocity=0.1, acceleration=0.3, open_gripper=True):
    """Move robot to start position"""

    print("="*60)
    print("   MOVE TO START POSITION")
    print("="*60)

    # Connect to robot
    print(f"\nConnecting to robot at {robot_ip}...")
    robot = UR5eRobot(robot_ip)

    # Get current pose
    current_pose = robot.get_tcp_pose()
    print(f"\nCurrent position:")
    print(f"  Position: [{current_pose[0]:.4f}, {current_pose[1]:.4f}, {current_pose[2]:.4f}] m")
    print(f"  Rotation: [{current_pose[3]:.4f}, {current_pose[4]:.4f}, {current_pose[5]:.4f}] rad")

    # Show target
    print(f"\nTarget position:")
    print(f"  Position: [{start_pose[0]:.4f}, {start_pose[1]:.4f}, {start_pose[2]:.4f}] m")
    print(f"  Rotation: [{start_pose[3]:.4f}, {start_pose[4]:.4f}, {start_pose[5]:.4f}] rad")

    # Ask for confirmation
    response = input("\nMove to start position? [Y/n]: ")
    if response.lower() == 'n':
        print("Aborted.")
        robot.disconnect()
        return

    # Initialize gripper if requested
    gripper = None
    if open_gripper:
        try:
            print("\nInitializing gripper...")
            gripper = DynamixelGripper()
            print("Opening gripper...")
            gripper.set_position(1.0)  # Open
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  Gripper initialization failed: {e}")
            print("Continuing without gripper...")

    # Move robot
    print(f"\nMoving robot...")
    print(f"  Velocity: {velocity} m/s")
    print(f"  Acceleration: {acceleration} m/s²")

    robot.move_tcp_pose(start_pose, velocity=velocity, acceleration=acceleration, asynchronous=False)

    print("✅ Robot moved to start position!")

    # Verify position
    time.sleep(0.5)
    final_pose = robot.get_tcp_pose()
    print(f"\nFinal position:")
    print(f"  Position: [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}] m")
    print(f"  Rotation: [{final_pose[3]:.4f}, {final_pose[4]:.4f}, {final_pose[5]:.4f}] rad")

    # Calculate error
    error = [(final_pose[i] - start_pose[i]) for i in range(6)]
    pos_error = sum([e**2 for e in error[:3]])**0.5
    print(f"\nPosition error: {pos_error*1000:.2f} mm")

    # Cleanup
    robot.disconnect()
    if gripper:
        gripper.disconnect()

    print("\n" + "="*60)
    print("   DONE!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Move robot to start position')
    parser.add_argument('--robot_ip', type=str, default='192.168.11.20',
                        help='Robot IP address (default: 192.168.11.20)')
    parser.add_argument('--custom_pose', type=float, nargs=6, default=None,
                        help='Custom start pose: x y z rx ry rz')
    parser.add_argument('--velocity', type=float, default=0.1,
                        help='Movement velocity in m/s (default: 0.1)')
    parser.add_argument('--acceleration', type=float, default=0.3,
                        help='Movement acceleration in m/s² (default: 0.3)')
    parser.add_argument('--no_gripper', action='store_true',
                        help='Do not open gripper')
    args = parser.parse_args()

    # Use custom pose or default
    if args.custom_pose:
        start_pose = args.custom_pose
        print(f"Using custom pose: {start_pose}")
    else:
        start_pose = DEFAULT_START_POSE
        print(f"Using default start pose")

    # Move robot
    try:
        move_to_start(
            robot_ip=args.robot_ip,
            start_pose=start_pose,
            velocity=args.velocity,
            acceleration=args.acceleration,
            open_gripper=not args.no_gripper,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
