#!/usr/bin/env python3
"""
Test UR5e Robot Connection via RTDE

Usage:
    cd ~/Desktop/my_pickplace
    python scripts/test_robot.py --robot_ip 150.65.146.87
"""

import sys
import os
import time
import click
import numpy as np

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKPLACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PICKPLACE_DIR)


def test_rtde_receive(robot_ip: str) -> bool:
    """Test RTDE Receive interface - read robot state"""
    print("\n" + "="*50)
    print("Testing RTDE RECEIVE (Read robot state)")
    print("="*50)

    try:
        from rtde_receive import RTDEReceiveInterface

        print(f"Connecting to {robot_ip}...")
        rtde_r = RTDEReceiveInterface(robot_ip)

        # Get TCP pose
        tcp_pose = rtde_r.getActualTCPPose()
        print(f"\nTCP Pose (x, y, z, rx, ry, rz):")
        print(f"  Position: [{tcp_pose[0]:.4f}, {tcp_pose[1]:.4f}, {tcp_pose[2]:.4f}] m")
        print(f"  Rotation: [{tcp_pose[3]:.4f}, {tcp_pose[4]:.4f}, {tcp_pose[5]:.4f}] rad")

        # Get joint positions
        joints = rtde_r.getActualQ()
        joints_deg = [np.degrees(j) for j in joints]
        print(f"\nJoint Positions:")
        print(f"  Radians: [{', '.join([f'{j:.4f}' for j in joints])}]")
        print(f"  Degrees: [{', '.join([f'{j:.1f}' for j in joints_deg])}]")

        # Get TCP speed
        tcp_speed = rtde_r.getActualTCPSpeed()
        print(f"\nTCP Speed: [{', '.join([f'{s:.4f}' for s in tcp_speed])}]")

        # Get robot mode
        robot_mode = rtde_r.getRobotMode()
        mode_names = {
            -1: "ROBOT_MODE_NO_CONTROLLER",
            0: "ROBOT_MODE_DISCONNECTED",
            1: "ROBOT_MODE_CONFIRM_SAFETY",
            2: "ROBOT_MODE_BOOTING",
            3: "ROBOT_MODE_POWER_OFF",
            4: "ROBOT_MODE_POWER_ON",
            5: "ROBOT_MODE_IDLE",
            6: "ROBOT_MODE_BACKDRIVE",
            7: "ROBOT_MODE_RUNNING",
        }
        print(f"\nRobot Mode: {robot_mode} ({mode_names.get(robot_mode, 'UNKNOWN')})")

        # Get safety status
        safety_mode = rtde_r.getSafetyMode()
        safety_names = {
            1: "NORMAL",
            2: "REDUCED",
            3: "PROTECTIVE_STOP",
            4: "RECOVERY",
            5: "SAFEGUARD_STOP",
            6: "SYSTEM_EMERGENCY_STOP",
            7: "ROBOT_EMERGENCY_STOP",
            8: "VIOLATION",
            9: "FAULT",
        }
        print(f"Safety Mode: {safety_mode} ({safety_names.get(safety_mode, 'UNKNOWN')})")

        print("\nRTDE RECEIVE: ✅ PASSED")
        return True

    except Exception as e:
        print(f"\nRTDE RECEIVE: ❌ FAILED - {e}")
        return False


def test_rtde_control(robot_ip: str) -> bool:
    """Test RTDE Control interface - check if we can send commands"""
    print("\n" + "="*50)
    print("Testing RTDE CONTROL (Send commands)")
    print("="*50)

    try:
        from rtde_control import RTDEControlInterface

        print(f"Connecting to {robot_ip}...")
        rtde_c = RTDEControlInterface(robot_ip)

        # Check if connected
        if rtde_c.isConnected():
            print("Connected to RTDE Control interface")
        else:
            print("Failed to connect to RTDE Control")
            return False

        # Check controller version
        print(f"\nController ready for commands")

        # Note: We don't actually move the robot here for safety
        print("\n⚠️  Not sending any motion commands (safety)")
        print("    Use demo_pickplace.py for actual control")

        # Disconnect
        rtde_c.disconnect()

        print("\nRTDE CONTROL: ✅ PASSED")
        return True

    except Exception as e:
        print(f"\nRTDE CONTROL: ❌ FAILED - {e}")
        print("\nPossible issues:")
        print("  1. Robot is in Local mode (switch to Remote)")
        print("  2. Robot is not powered on")
        print("  3. Emergency stop is pressed")
        print("  4. Another program is controlling the robot")
        return False


def continuous_monitor(robot_ip: str, duration: int = 10):
    """Continuously monitor robot state"""
    print("\n" + "="*50)
    print(f"Monitoring robot for {duration} seconds...")
    print("="*50)
    print("Press Ctrl+C to stop\n")

    try:
        from rtde_receive import RTDEReceiveInterface

        rtde_r = RTDEReceiveInterface(robot_ip)
        start_time = time.time()

        while time.time() - start_time < duration:
            tcp = rtde_r.getActualTCPPose()
            speed = rtde_r.getActualTCPSpeed()
            speed_magnitude = np.sqrt(sum([s**2 for s in speed[:3]]))

            print(f"TCP: [{tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f}] m | "
                  f"Speed: {speed_magnitude:.4f} m/s", end='\r')

            time.sleep(0.1)

        print("\n\nMonitoring complete!")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")


@click.command()
@click.option('--robot_ip', required=True, help='UR5e IP address')
@click.option('--monitor', is_flag=True, help='Continuously monitor robot state')
@click.option('--duration', default=10, help='Monitor duration in seconds')
def main(robot_ip, monitor, duration):
    print("="*50)
    print("         UR5e ROBOT TEST")
    print("="*50)
    print(f"\nRobot IP: {robot_ip}")

    # Test RTDE Receive (read-only, always safe)
    receive_ok = test_rtde_receive(robot_ip)

    # Test RTDE Control (need robot in Remote mode)
    control_ok = test_rtde_control(robot_ip)

    # Summary
    print("\n" + "="*50)
    print("         TEST SUMMARY")
    print("="*50)
    print(f"  RTDE Receive (read state) : {'✅ PASSED' if receive_ok else '❌ FAILED'}")
    print(f"  RTDE Control (send cmds)  : {'✅ PASSED' if control_ok else '❌ FAILED'}")
    print("="*50)

    if receive_ok and control_ok:
        print("\n✅ Robot ready for teleoperation!")

        if monitor:
            continuous_monitor(robot_ip, duration)
    else:
        print("\n⚠️  Some tests failed. Check:")
        print("  1. Robot is powered on")
        print("  2. Robot is in Remote Control mode")
        print("  3. No protective stop / emergency stop")
        print("  4. RTDE is enabled in robot settings")


if __name__ == '__main__':
    main()
