#!/usr/bin/env python3
"""
Move UR5e back to the start position and reset Flowbot.

Uses the same hardware wrappers as deploy_flowbot_w_policy.py.

Usage:
    python deploy/returning.py --robot_ip 192.168.1.100
    python deploy/returning.py --robot_ip 192.168.1.100 --flowbot_port /dev/ttyACM0
    python deploy/returning.py --robot_ip 192.168.1.100 --pose 0.206 -0.467 0.443 3.14 -0.14 0.0
    python deploy/returning.py --robot_ip 192.168.1.100 --speed 0.1 --accel 0.1
"""

import os
import sys
import time
import argparse
import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(DEPLOY_DIR)
sys.path.insert(0, LEARNING_DIR)

from hardware.ur5e_rtde import UR5eRobot
from hardware.flowbot import flowbot

# ── Constants — must match deploy_flowbot_w_policy.py ─────────────────────────
PWM_MIN = 1
PWM_MAX = 26
CONTROL_FREQ = 10.0
FLOWBOT_FREQ = 10

DEFAULT_START_POSE = [0.20636, -0.46706, 0.44268, 3.14, -0.14, 0.0]


def return_to_start(
    robot_ip: str,
    target_pose: list,
    speed: float = 0.3,
    accel: float = 0.3,
    flowbot_port: str = None,
    flowbot_baud: int = 115200,
):
    """
    Move UR5e to target_pose using moveL, then reset Flowbot PWM.

    Args:
        robot_ip     : UR5e IP address
        target_pose  : [x, y, z, rx, ry, rz] in metres / radians
        speed        : moveL speed (m/s)
        accel        : moveL acceleration (m/s^2)
        flowbot_port : Serial port for Flowbot. If None, skip flowbot.
        flowbot_baud : Serial baud rate
    """
    print("=" * 50)
    print("RETURNING TO START POSITION")
    print("=" * 50)

    # ── UR5e ──────────────────────────────────────────────────────────────────
    ur5 = None
    try:
        print(f"\nConnecting to UR5e at {robot_ip} ...")
        ur5 = UR5eRobot(robot_ip=robot_ip, frequency=CONTROL_FREQ)
        print("  UR5e connected")

        current_pose = ur5.get_tcp_pose()
        print(f"  Current TCP : {[f'{v:.4f}' for v in current_pose]}")
        print(f"  Target TCP  : {[f'{v:.4f}' for v in target_pose]}")

        print(f"\nExecuting moveL (speed={speed} m/s, accel={accel} m/s^2) ...")
        ur5.move_tcp_pose(target_pose, velocity=speed, acceleration=accel)

        final_pose = ur5.get_tcp_pose()
        print(f"  Final TCP   : {[f'{v:.4f}' for v in final_pose]}")

        pos_error = np.linalg.norm(np.array(final_pose[:3]) - np.array(target_pose[:3]))
        print(f"  Position error: {pos_error * 1000:.2f} mm")
        print("  ✅ Robot at start position!")

    except Exception as e:
        print(f"  ❌ Robot move failed: {e}")
        return False
    finally:
        if ur5 is not None:
            try:
                ur5.disconnect()
            except Exception:
                pass

    # ── Flowbot ───────────────────────────────────────────────────────────────
    if flowbot_port is not None:
        print(f"\nResetting Flowbot on {flowbot_port} ...")
        fb = None
        try:
            fb = flowbot(
                serial_port=flowbot_port,
                baud=flowbot_baud,
                pwm_min=PWM_MIN,
                pwm_max=PWM_MAX,
                enable_plot=False,
                frequency=FLOWBOT_FREQ,
                max_pos_speed=30,
            )
            fb.start()
            time.sleep(2.0)     # Arduino reset delay
            fb.reset()
            time.sleep(0.5)
            print(f"  ✅ Flowbot reset!  Last PWM: {fb.last_pwm}")
        except Exception as e:
            print(f"  ❌ Flowbot reset failed: {e}")
            return False

    print("\n✅ Ready for next episode!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Return UR5e to start position and reset Flowbot'
    )
    parser.add_argument('--robot_ip',     type=str,   required=True,
                        help='UR5e IP address (e.g. 192.168.1.100)')
    parser.add_argument('--pose',         type=float, nargs=6,
                        default=DEFAULT_START_POSE,
                        metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                        help='Target TCP pose [x y z rx ry rz] '
                             '(default: collection start pose)')
    parser.add_argument('--speed',        type=float, default=0.3,
                        help='moveL speed (m/s, default 0.3)')
    parser.add_argument('--accel',        type=float, default=0.3,
                        help='moveL acceleration (m/s^2, default 0.3)')
    parser.add_argument('--flowbot_port', type=str,   default=None,
                        help='Flowbot serial port (e.g. /dev/ttyACM0). '
                             'If omitted, only the robot is moved.')
    parser.add_argument('--flowbot_baud', type=int,   default=115200,
                        help='Flowbot baud rate (default 115200)')
    args = parser.parse_args()

    print(f"Target pose: {args.pose}")

    ok = return_to_start(
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
