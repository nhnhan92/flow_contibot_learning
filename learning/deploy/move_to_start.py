#!/usr/bin/env python3
"""
Move UR5e to start position and reset Flowbot

Useful to:
  - Reset the robot to a known start before a deployment episode
  - Test that both UR5e and Flowbot are responsive
  - Quickly reset after a failed episode

Usage:
    python deploy/move_to_start.py --robot_ip 192.168.1.100
    python deploy/move_to_start.py --robot_ip 192.168.1.100 --flowbot_port /dev/ttyACM0
    python deploy/move_to_start.py --robot_ip 192.168.1.100 --pose 0.206 -0.467 0.443 3.14 -0.14 0.0
"""

import os
import sys
import time
import argparse
import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEPLOY_DIR)
FLOWBOT_DIR = os.path.join(PROJECT_DIR, 'flowbot')
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, FLOWBOT_DIR)

# Default start pose (from collect_demos_with_camera.py)
DEFAULT_START_POSE = [0.20636, -0.46706, 0.44268, 3.14, -0.14, 0.0]


def move_to_start(
    robot_ip: str,
    target_pose: list,
    speed: float = 0.3,
    accel: float = 0.3,
    flowbot_port: str = None,
    flowbot_baud: int = 115200,
):
    """
    Move UR5e to target_pose using moveL, then reset Flowbot.

    Args:
        robot_ip     : UR5e IP address
        target_pose  : [x, y, z, rx, ry, rz] in metres / radians
        speed        : moveL speed (m/s)
        accel        : moveL acceleration (m/s^2)
        flowbot_port : Serial port for Flowbot. If None, skip flowbot.
        flowbot_baud : Serial baud rate
    """
    # ── UR5e ─────────────────────────────────────────────────────────────────
    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        print("❌ rtde_control not installed. Run: pip install ur-rtde")
        return False

    print("="*50)
    print("MOVING TO START POSITION")
    print("="*50)

    try:
        print(f"\nConnecting to UR5e at {robot_ip} ...")
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        current_pose = rtde_r.getActualTCPPose()
        print(f"  Current TCP : {[f'{v:.4f}' for v in current_pose]}")
        print(f"  Target TCP  : {[f'{v:.4f}' for v in target_pose]}")

        print(f"\nExecuting moveL (speed={speed} m/s, accel={accel} m/s^2) ...")
        rtde_c.moveL(target_pose, speed, accel)

        final_pose = rtde_r.getActualTCPPose()
        print(f"  Final TCP   : {[f'{v:.4f}' for v in final_pose]}")

        pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target_pose[:3]))
        print(f"  Position error: {pos_error*1000:.2f} mm")

        rtde_c.disconnect()
        rtde_r.disconnect()
        print("  ✅ Robot at start position!")

    except Exception as e:
        print(f"  ❌ Robot move failed: {e}")
        return False

    # ── Flowbot ───────────────────────────────────────────────────────────────
    if flowbot_port is not None:
        print(f"\nResetting Flowbot on {flowbot_port} ...")
        try:
            from flowbot import Flowbot
            fb = Flowbot(port=flowbot_port, baud=flowbot_baud)
            time.sleep(2.0)
            fb.reset()
            time.sleep(0.5)
            print(f"  ✅ Flowbot reset!  Last PWM: {fb.last_pwm}")
        except Exception as e:
            print(f"  ❌ Flowbot reset failed: {e}")
            return False

    print("\n✅ Ready for deployment!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Move UR5e to start and reset Flowbot')
    parser.add_argument('--robot_ip',     type=str,   required=True,
                        help='UR5e IP address (e.g. 192.168.1.100)')
    parser.add_argument('--pose',         type=float, nargs=6,
                        default=DEFAULT_START_POSE,
                        metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        help='Target TCP pose [x y z rx ry rz] (default: collection start)')
    parser.add_argument('--speed',        type=float, default=0.3, help='moveL speed (m/s)')
    parser.add_argument('--accel',        type=float, default=0.3, help='moveL accel (m/s^2)')
    parser.add_argument('--flowbot_port', type=str,   default=None,
                        help='Flowbot serial port (e.g. /dev/ttyACM0). '
                             'If omitted, only the robot is moved.')
    parser.add_argument('--flowbot_baud', type=int,   default=115200,
                        help='Flowbot baud rate')
    args = parser.parse_args()

    print(f"Target pose: {args.pose}")

    ok = move_to_start(
        robot_ip=args.robot_ip,
        target_pose=args.pose,
        speed=args.speed,
        accel=args.accel,
        flowbot_port=args.flowbot_port,
        flowbot_baud=args.flowbot_baud,
    )
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
