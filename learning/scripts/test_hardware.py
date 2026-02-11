#!/usr/bin/env python3
"""
Test all hardware components before data collection
"""

import sys
import os

# Add paths
PICKPLACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFUSION_POLICY_DIR = os.path.join(os.path.dirname(PICKPLACE_DIR), 'diffusion_policy')
sys.path.insert(0, PICKPLACE_DIR)
sys.path.insert(0, DIFFUSION_POLICY_DIR)

import time
import numpy as np
import click


def test_gripper(port, dxl_id):
    """Test Dynamixel gripper"""
    print("\n" + "="*50)
    print("Testing DYNAMIXEL GRIPPER")
    print("="*50)

    try:
        from custom.dynamixel_gripper import DynamixelGripper

        gripper = DynamixelGripper(port=port, dxl_id=dxl_id)

        print("Opening...")
        gripper.open()
        time.sleep(1)
        print(f"Position: {gripper.get_position():.2f}")

        print("Closing...")
        gripper.close()
        time.sleep(1)
        print(f"Position: {gripper.get_position():.2f}")

        gripper.open()
        gripper.disconnect()
        print("GRIPPER TEST: PASSED")
        return True

    except Exception as e:
        print(f"GRIPPER TEST: FAILED - {e}")
        return False


def test_robot(robot_ip):
    """Test UR5e connection"""
    print("\n" + "="*50)
    print("Testing UR5e ROBOT")
    print("="*50)

    try:
        from rtde_receive import RTDEReceiveInterface

        rtde_r = RTDEReceiveInterface(robot_ip)
        pose = rtde_r.getActualTCPPose()
        joints = rtde_r.getActualQ()

        print(f"TCP Pose: {[f'{x:.3f}' for x in pose]}")
        print(f"Joints (deg): {[f'{np.degrees(x):.1f}' for x in joints]}")
        print("ROBOT TEST: PASSED")
        return True

    except Exception as e:
        print(f"ROBOT TEST: FAILED - {e}")
        return False


def test_camera():
    """Test RealSense camera"""
    print("\n" + "="*50)
    print("Testing REALSENSE CAMERA")
    print("="*50)

    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            print("No RealSense device found!")
            print("CAMERA TEST: FAILED")
            return False

        for i, dev in enumerate(devices):
            print(f"Device {i}: {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")

        print("CAMERA TEST: PASSED")
        return True

    except Exception as e:
        print(f"CAMERA TEST: FAILED - {e}")
        return False


def test_spacemouse():
    """Test SpaceMouse"""
    print("\n" + "="*50)
    print("Testing SPACEMOUSE (custom with correct axis mapping)")
    print("="*50)

    try:
        from custom.spacemouse import SpaceMouse

        print("Move SpaceMouse to test (5 seconds)...")
        print("  X = forward/back, Y = left/right, Z = up/down\n")

        with SpaceMouse(deadzone=0.1, max_value=350) as sm:
            for i in range(50):
                state = sm.get_motion_state_transformed()
                btn_l = sm.is_button_pressed(0)
                btn_r = sm.is_button_pressed(1)

                if i % 10 == 0:
                    print(f"  XYZ:[{state[0]:+5.2f},{state[1]:+5.2f},{state[2]:+5.2f}] "
                          f"Rot:[{state[3]:+5.2f},{state[4]:+5.2f},{state[5]:+5.2f}] "
                          f"Btn: L={int(btn_l)} R={int(btn_r)}")
                time.sleep(0.1)

        print("SPACEMOUSE TEST: PASSED")
        return True

    except Exception as e:
        print(f"SPACEMOUSE TEST: FAILED - {e}")
        return False


@click.command()
@click.option('--robot_ip', default='192.168.1.100', help='UR5e IP')
@click.option('--gripper_port', default='/dev/ttyUSB0', help='Gripper port')
@click.option('--gripper_id', default=1, help='Dynamixel ID')
@click.option('--skip_robot', is_flag=True, help='Skip robot test')
@click.option('--skip_gripper', is_flag=True, help='Skip gripper test')
def main(robot_ip, gripper_port, gripper_id, skip_robot, skip_gripper):
    print("\n" + "="*50)
    print("       HARDWARE TEST SUITE")
    print("="*50)

    results = {}

    # Test each component
    if not skip_gripper:
        results['gripper'] = test_gripper(gripper_port, gripper_id)
    else:
        results['gripper'] = None
        print("\nGRIPPER TEST: SKIPPED")

    if not skip_robot:
        results['robot'] = test_robot(robot_ip)
    else:
        results['robot'] = None
        print("\nROBOT TEST: SKIPPED")

    results['camera'] = test_camera()
    results['spacemouse'] = test_spacemouse()

    # Summary
    print("\n" + "="*50)
    print("       TEST SUMMARY")
    print("="*50)

    all_passed = True
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
            all_passed = False
        print(f"  {name.upper():15} : {status}")

    print("="*50)
    if all_passed:
        print("All tests passed! Ready for data collection.")
    else:
        print("Some tests failed. Please fix before proceeding.")


if __name__ == '__main__':
    main()
