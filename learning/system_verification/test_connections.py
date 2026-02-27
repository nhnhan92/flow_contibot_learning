#!/usr/bin/env python3
"""
Test all robot connections before deployment

Usage:
    python system_verification/test_connections.py \
        --robot_ip 192.168.1.100 \
        --camera_id 0 \
        --flowbot_port /dev/ttyACM0

Tests:
    1. UR5e RTDE connection
    2. Camera capture
    3. Flowbot (Arduino serial) connection
"""

import os
import sys
import time
import argparse
import numpy as np

SYSVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SYSVER_DIR)
sys.path.insert(0, PROJECT_DIR)

FLOWBOT_DIR = os.path.join(PROJECT_DIR, 'flowbot')
sys.path.insert(0, FLOWBOT_DIR)


def test_robot(robot_ip: str) -> bool:
    """Test UR5e RTDE connection and read current TCP pose."""
    print("\n" + "="*50)
    print("TEST 1: UR5e Robot Connection")
    print("="*50)

    try:
        import rtde_control
        import rtde_receive
    except ImportError:
        print("❌ rtde_control not installed. Run: pip install ur-rtde")
        return False

    try:
        print(f"  Connecting to UR5e at {robot_ip} ...")
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)

        tcp_pose = rtde_r.getActualTCPPose()
        joint_pos = rtde_r.getActualQ()
        robot_mode = rtde_r.getRobotMode()

        print(f"  ✅ Connected!")
        print(f"     Robot mode : {robot_mode}  (7 = running)")
        print(f"     TCP pose   : {[f'{v:.4f}' for v in tcp_pose]}")
        print(f"     Joints (°) : {[f'{np.degrees(j):.1f}' for j in joint_pos]}")

        rtde_c.disconnect()
        rtde_r.disconnect()
        return True

    except Exception as e:
        print(f"  ❌ Robot connection failed: {e}")
        return False


def test_camera(camera_id: int) -> bool:
    """Test camera capture and display frame info."""
    print("\n" + "="*50)
    print("TEST 2: Camera")
    print("="*50)

    try:
        import cv2
    except ImportError:
        print("❌ opencv-python not installed.")
        return False

    try:
        print(f"  Opening camera id={camera_id} ...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"  ❌ Cannot open camera id={camera_id}")
            return False

        # Warmup
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print("  ❌ Cannot read frame from camera")
            cap.release()
            return False

        h, w, c = frame.shape
        print(f"  ✅ Camera OK!")
        print(f"     Resolution : {w} × {h}  ({c} channels)")

        # Check brightness (very dark → wrong camera or lens cap)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        print(f"     Brightness : {mean_brightness:.1f} / 255  ", end="")
        if mean_brightness < 10:
            print("⚠️  (very dark — check lens cap)")
        elif mean_brightness < 30:
            print("⚠️  (dim — check lighting)")
        else:
            print("✅")

        cap.release()
        return True

    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        return False


def test_flowbot(port: str, baud: int = 115200) -> bool:
    """
    Test Flowbot (Arduino) serial connection.
    Sends a zero-PWM command, then a small test pulse, then resets.
    """
    print("\n" + "="*50)
    print("TEST 3: Flowbot (Arduino Serial)")
    print("="*50)

    try:
        from flowbot import Flowbot
    except ImportError:
        print("  ❌ flowbot module not found.")
        print(f"     Expected at: {FLOWBOT_DIR}/flowbot.py")
        return False

    try:
        print(f"  Connecting to Flowbot on {port} @ {baud} baud ...")
        fb = Flowbot(port=port, baud=baud)
        time.sleep(2.0)   # Arduino bootloader delay

        print("  ✅ Connected!")

        # Read initial PWM state
        print(f"     Initial PWM: {fb.last_pwm}")

        # Send a zero command (safe)
        fb.serial_sending(np.array([0, 0, 0], dtype=int))
        time.sleep(0.3)
        print(f"     After zero command — PWM: {fb.last_pwm}")

        # Small test pulse on valve 1 only
        print("  Sending test pulse (PWM1=5, PWM2=0, PWM3=0) for 1s ...")
        fb.serial_sending(np.array([5, 0, 0], dtype=int))
        time.sleep(1.0)

        # Reset all valves
        print("  Resetting Flowbot ...")
        fb.reset()
        time.sleep(0.3)
        print(f"     After reset — PWM: {fb.last_pwm}")
        print("  ✅ Flowbot OK!")

        return True

    except Exception as e:
        print(f"  ❌ Flowbot test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test robot connections before deployment')
    parser.add_argument('--robot_ip',     type=str, default='192.168.1.100',
                        help='UR5e IP address')
    parser.add_argument('--camera_id',    type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--flowbot_port', type=str, default='/dev/ttyACM0',
                        help='Arduino serial port for Flowbot')
    parser.add_argument('--flowbot_baud', type=int, default=115200,
                        help='Flowbot serial baud rate')
    parser.add_argument('--skip_robot',   action='store_true', help='Skip robot test')
    parser.add_argument('--skip_camera',  action='store_true', help='Skip camera test')
    parser.add_argument('--skip_flowbot', action='store_true', help='Skip flowbot test')
    args = parser.parse_args()

    print("="*50)
    print("CONNECTION TESTS — UR5e + Flowbot")
    print("="*50)

    results = {}

    if not args.skip_robot:
        results['UR5e Robot'] = test_robot(args.robot_ip)
    else:
        print("\n⏭  Skipping robot test")

    if not args.skip_camera:
        results['Camera'] = test_camera(args.camera_id)
    else:
        print("\n⏭  Skipping camera test")

    if not args.skip_flowbot:
        results['Flowbot'] = test_flowbot(args.flowbot_port, args.flowbot_baud)
    else:
        print("\n⏭  Skipping flowbot test")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Safe to run deploy_real_robot.py")
    else:
        print("\n⚠️  Fix the failing tests before deploying.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
