#!/usr/bin/env python3
"""
Test Teleoperation: Control UR5e with SpaceMouse

Điều khiển robot UR5e bằng SpaceMouse để test trước khi thu thập dữ liệu.
Không cần camera hay gripper.

Usage:
    cd ~/Desktop/my_pickplace
    python scripts/test_teleop.py --robot_ip 150.65.146.87

Controls:
    SpaceMouse:
        - Push forward/back  → Robot X (forward/backward)
        - Push left/right    → Robot Y (left/right)
        - Lift up/down       → Robot Z (up/down)
        - Tilt/Rotate        → Robot rotation (rx, ry, rz)
        - Left button        → (reserved)
        - Right button       → Emergency stop

    Keyboard:
        - 'q' or ESC         → Quit
        - 'r'                → Reset to home position
        - 's'                → Print current status
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
import scipy.spatial.transform as st

from custom.spacemouse import SpaceMouse
from custom.ur5e_rtde import UR5eRobot


def print_status(tcp_pose, target_pose, speed_scale):
    """Print current robot status"""
    print("\n" + "-"*50)
    print("Current Status:")
    print(f"  TCP Position: [{tcp_pose[0]:+.4f}, {tcp_pose[1]:+.4f}, {tcp_pose[2]:+.4f}] m")
    print(f"  TCP Rotation: [{tcp_pose[3]:+.4f}, {tcp_pose[4]:+.4f}, {tcp_pose[5]:+.4f}] rad")
    print(f"  Target Pos:   [{target_pose[0]:+.4f}, {target_pose[1]:+.4f}, {target_pose[2]:+.4f}] m")
    print(f"  Speed Scale:  {speed_scale:.2f}")
    print("-"*50)


@click.command()
@click.option('--robot_ip',default = '192.168.11.20', required=False, help='UR5e IP address')
@click.option('--frequency', default=10, help='Control frequency (Hz)')
@click.option('--max_pos_speed', default=0.05, help='Max linear speed (m/s)')
@click.option('--max_rot_speed', default=0.1, help='Max angular speed (rad/s)')
@click.option('--speed_scale', default=0.5, help='Speed scaling factor (0.1-1.0)')
def main(robot_ip, frequency, max_pos_speed, max_rot_speed, speed_scale):
    print("="*60)
    print("       UR5e TELEOPERATION TEST (SpaceMouse)")
    print("="*60)
    ur5 = UR5eRobot(robot_ip=robot_ip,frequency=frequency)

    # Workspace limits (meters) - safety bounds
    WORKSPACE = {
        'x_min': -0.6, 'x_max': 0.6,
        'y_min': -0.6, 'y_max': 0.6,
        'z_min': 0.02, 'z_max': 0.6,  # Min 2cm above table
    }

    print(f"\nRobot IP: {robot_ip}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Max linear speed: {max_pos_speed} m/s")
    print(f"Max angular speed: {max_rot_speed} rad/s")
    print(f"Speed scale: {speed_scale}")
    print(f"\nWorkspace limits:")
    print(f"  X: [{WORKSPACE['x_min']}, {WORKSPACE['x_max']}] m")
    print(f"  Y: [{WORKSPACE['y_min']}, {WORKSPACE['y_max']}] m")
    print(f"  Z: [{WORKSPACE['z_min']}, {WORKSPACE['z_max']}] m")

    # Check robot mode
    robot_mode = ur5.get_robot_mode()
    if robot_mode != 7:  # ROBOT_MODE_RUNNING
        print(f"\nWarning: Robot mode is {robot_mode}, expected 7 (RUNNING)")
        print("Make sure robot is in Remote Control mode and running!")

    # Connect to SpaceMouse
    print("\nConnecting to SpaceMouse...")
    try:
        sm = SpaceMouse(deadzone=0.15, max_value=350)
        print("SpaceMouse connected!")
    except Exception as e:
        print(f"Failed to connect to SpaceMouse: {e}")
        ur5.disconnect()
        return

    # Get initial pose
    tcp_pose = ur5.get_tcp_pose()
    target_pose = tcp_pose.copy()
 
    target_pose[3:] = [3.14, 0.0 ,0.0]

    print("\n" + "="*60)
    print("Controls:")
    print("  SpaceMouse  → Move robot (XYZ + rotation)")
    print("  Right btn   → Emergency stop (hold to pause)")
    print("  'q' or ESC  → Quit")
    print("  'r'         → Reset to current position")
    print("  's'         → Print status")
    print("  '+'/'-'     → Increase/decrease speed")
    print("="*60)
    print("\nStarting teleoperation... (Press 'q' to quit)")
    print_status(tcp_pose, target_pose, speed_scale)

    # Control loop parameters
    dt = 1.0 / frequency
    running = True
    paused = False

    # For keyboard input (non-blocking)
    import select
    import termios
    import tty

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        iter_count = 0
        last_print_time = time.time()

        while running:
            loop_start = time.time()

            # Check keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'q' or key == '\x1b':  # q or ESC
                    print("\n\nQuitting...")
                    running = False
                    continue
                elif key == 'r':
                    # Reset target to current pose
                    tcp_pose = ur5.get_tcp_pose()
                    target_pose = tcp_pose.copy()
                    print("\nReset to current position")
                elif key == 's':
                    tcp_pose = ur5.get_tcp_pose()
                    print_status(tcp_pose, target_pose, speed_scale)
                elif key == '+' or key == '=':
                    speed_scale = min(1.0, speed_scale + 0.1)
                    print(f"\nSpeed scale: {speed_scale:.1f}")
                elif key == '-':
                    speed_scale = max(0.1, speed_scale - 0.1)
                    print(f"\nSpeed scale: {speed_scale:.1f}")

            # Get SpaceMouse state
            sm_state = sm.get_motion_state_transformed()
            btn_right = sm.is_button_pressed(1)

            # Emergency pause if right button pressed
            if btn_right:
                if not paused:
                    print("\n⚠️  PAUSED (release right button to continue)")
                    paused = True
                    ur5.stop()
                time.sleep(0.1)
                continue
            elif paused:
                print("Resuming...")
                paused = False
                # Reset target to current pose when resuming
                tcp_pose = ur5.get_tcp_pose()
                target_pose = tcp_pose.copy()

            # Calculate velocity from SpaceMouse
            # sm_state = [x, y, z, rx, ry, rz] normalized ~[-1, 1]
            vel_linear = np.array([
                sm_state[0] * max_pos_speed * speed_scale,  # X
                sm_state[1] * max_pos_speed * speed_scale,  # Y
                sm_state[2] * max_pos_speed * speed_scale,  # Z
            ])

            vel_angular = np.array([
                sm_state[3] * max_rot_speed * speed_scale,  # rx
                sm_state[4] * max_rot_speed * speed_scale,  # ry
                sm_state[5] * max_rot_speed * speed_scale,  # rz
            ])

            # Update target pose
            target_pose[:3] += vel_linear * dt
            target_pose[3:] += vel_angular * dt

            if np.any(vel_angular != 0):
                drot = st.Rotation.from_euler('xyz', vel_angular)
                current_rot = st.Rotation.from_rotvec(target_pose[3:])
                target_pose[3:] = (drot * current_rot).as_rotvec()

            # Apply workspace limits
            target_pose[0] = np.clip(target_pose[0], WORKSPACE['x_min'], WORKSPACE['x_max'])
            target_pose[1] = np.clip(target_pose[1], WORKSPACE['y_min'], WORKSPACE['y_max'])
            target_pose[2] = np.clip(target_pose[2], WORKSPACE['z_min'], WORKSPACE['z_max'])

            print(f"target_pose = {target_pose}")
            # Send command to robot using servoL
            # servoL(pose, velocity, acceleration, time, lookahead_time, gain)
            try:
                ur5.servo_tcp_pose(target_pose=target_pose,velocity=0.5,
                                  acceleration=0.5,dt=dt,lookahead_time=0.1,gain=300)
            except Exception as e:
                print(f"\nControl error: {e}")
                # Try to recover
                tcp_pose = ur5.get_tcp_pose()
                target_pose = tcp_pose.copy()

            # Print status periodically
            iter_count += 1
            if time.time() - last_print_time > 1.0:
                tcp_pose = ur5.get_tcp_pose()
                sm_mag = np.sqrt(sm_state[0]**2 + sm_state[1]**2 + sm_state[2]**2)

                print(f"TCP:[{tcp_pose[0]:+.3f},{tcp_pose[1]:+.3f},{tcp_pose[2]:+.3f}] | "
                      f"SM:[{sm_state[0]:+.2f},{sm_state[1]:+.2f},{sm_state[2]:+.2f}] | "
                      f"Speed:{speed_scale:.1f}  ", end='\r')

                last_print_time = time.time()

            # Maintain loop frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Stop robot and cleanup
        print("\nStopping robot...")
        ur5.disconnect()
        sm.close()

        print("Teleoperation ended.")


if __name__ == '__main__':
    main()
