#!/usr/bin/env python3
"""
Check if the Franka robot is connected and responsive via FrankaRobot interface.

Usage:
    python check_franka_connection.py [--ip 172.16.0.2]
"""

import argparse
import sys
import traceback

import numpy as np


def check_franky_import():
    print("[1/4] Checking franky installation ...")
    try:
        import franky
        print(f"      franky found: {franky.__file__}")
        return True
    except ImportError as e:
        print(f"      FAIL: {e}")
        print("      Install with: pip install franky-control")
        return False


def check_connection(robot_ip: str):
    print(f"[2/4] Connecting to Franka at {robot_ip} ...")
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "hardware"))
        from franka_robot import FrankaRobot
        robot = FrankaRobot(robot_ip=robot_ip)
        print("      Connection OK")
        return robot
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return None


def check_state(robot):
    print("[3/4] Reading robot state ...")
    try:
        joints = robot.get_joint_angles()
        assert joints.shape == (7,), f"Expected (7,) joint angles, got {joints.shape}"
        print(f"      Joint angles (rad): {np.round(joints, 4)}")

        tcp = robot.get_tcp_pose()
        assert tcp.shape == (6,), f"Expected (6,) TCP pose, got {tcp.shape}"
        print(f"      TCP pose [x,y,z,rx,ry,rz]: {np.round(tcp, 4)}")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_error_recovery(robot):
    print("[4/4] Testing error recovery call ...")
    try:
        robot.recover()
        print("      recover_from_errors() OK")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Franka robot connection.")
    parser.add_argument("--ip", default="172.16.0.2", help="Franka FCI IP address")
    args = parser.parse_args()

    print("=" * 50)
    print(" Franka Connection Check")
    print("=" * 50)

    results = {}

    results["franky"] = check_franky_import()
    if not results["franky"]:
        print("\nAborting: franky not available.")
        sys.exit(1)

    robot = check_connection(args.ip)
    results["connection"] = robot is not None
    if robot is None:
        print("\nAborting: could not connect to robot.")
        sys.exit(1)

    results["state"] = check_state(robot)
    results["recovery"] = check_error_recovery(robot)

    robot.disconnect()

    print("\n" + "=" * 50)
    all_ok = all(results.values())
    for check, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {check:<12} : {status}")
    print("=" * 50)
    print("Overall:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
