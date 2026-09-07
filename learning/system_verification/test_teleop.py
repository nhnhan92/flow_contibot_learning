#!/usr/bin/env python3
"""
Test Teleoperation: Control UR5e or Franka with SpaceMouse

Điều khiển robot (UR5e hoặc Franka) bằng SpaceMouse để test trước khi thu
thập dữ liệu. Không cần camera hay gripper.

Usage:
    cd ~/Desktop/flow_contibot_learning/learning
    python system_verification/test_teleop.py --arm ur5 --robot_ip 150.65.146.87
    python system_verification/test_teleop.py --arm franka --robot_ip 172.16.0.2 \\
        --max_pos_speed 0.02 --max_rot_speed 0.02 --speed_scale 0.2

    # Franka only: compare velocity vs position control for smoothness under
    # a sustained SpaceMouse push (see FrankaRobot.set_ee_velocity() vs
    # set_tcp_pose() in hardware/franka_robot.py).
    python system_verification/test_teleop.py --arm franka --control_mode velocity ...
    python system_verification/test_teleop.py --arm franka --control_mode position ...

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
SYSVER_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(SYSVER_DIR)
sys.path.insert(0, LEARNING_DIR)
import scipy.spatial.transform as st

from hardware.spacemouse import SpaceMouse
from hardware.ur5e_rtde import UR5eRobot
from hardware.franka_robot import FrankaRobot

INIT_POSE_UR5E = np.array([0.550, 0.045, 0.45, 3.14, 0.0, -0.05])
INIT_POSE_FRANKA = np.array([0.45, 0.15, 0.5, 3.14, 0.0, -0.05])

def print_status(tcp_pose, target_pose, speed_scale):
    """Print current robot status"""
    print("\n" + "-"*50)
    print("Current Status:")
    print(f"  TCP Position: [{tcp_pose[0]:+.4f}, {tcp_pose[1]:+.4f}, {tcp_pose[2]:+.4f}] m")
    print(f"  TCP Rotation: [{tcp_pose[3]:+.4f}, {tcp_pose[4]:+.4f}, {tcp_pose[5]:+.4f}] rad")
    print(f"  Target Pos:   [{target_pose[0]:+.4f}, {target_pose[1]:+.4f}, {target_pose[2]:+.4f}] m")
    print(f"  Speed Scale:  {speed_scale:.2f}")
    print("-"*50)


def _servo_toward(arm, is_franka, target_pose, dt, velocity, acceleration,
                   gain=300, lookahead_time=0.1):
    """
    Command `arm` one tick toward an absolute 6D target_pose.

    UR5eRobot: servo_tcp_pose(target_pose, ...) -- RTDE servoL tracks the
    absolute target directly.

    FrankaRobot (franky): has no absolute-position streaming primitive,
    only set_ee_velocity(). So the absolute target gets translated into an
    error-based feed-forward linear velocity, (target - current)/dt,
    clipped to `velocity` m/s. Angular velocity is left at zero, matching
    how rotation-via-spacemouse is already disabled on the UR5e path here.
    """
    if is_franka:
        current_pose = arm.get_tcp_pose()
        lin_vel = (np.asarray(target_pose[:3], dtype=float) - current_pose[:3]) / dt
        speed = float(np.linalg.norm(lin_vel))
        if speed > velocity and speed > 1e-9:
            lin_vel = lin_vel / speed * velocity
        arm.set_ee_velocity(lin_vel, angular_velocity=np.zeros(3),
                             max_vel=velocity, max_ang_vel=acceleration)
    else:
        arm.servo_tcp_pose(
            target_pose=target_pose, velocity=velocity, acceleration=acceleration,
            dt=dt, lookahead_time=lookahead_time, gain=gain,
        )


def move_2_init_pos(arm, start_pose, goal_pose, dt, duration=5.0,
                    velocity=0.1, acceleration=0.1, gain=200, lookahead_time=0.15,
                    is_franka=False, settle_time=0.3):
    """Move arm from start_pose to goal_pose.

    UR5eRobot: interpolates (position lerp + rotation slerp) over `duration`
    seconds, streaming each waypoint via servo_tcp_pose (RTDE servoL) at
    ~1/dt Hz."""
    start_pose = np.asarray(start_pose, dtype=float).copy()
    goal_pose  = np.asarray(goal_pose,  dtype=float).copy()

    if is_franka:
        arm.stop()
        if settle_time > 0:
            time.sleep(settle_time)
        arm.move_tcp_pose(goal_pose, velocity=velocity, acceleration=acceleration)
        return

    r0    = st.Rotation.from_rotvec(start_pose[3:])
    r1    = st.Rotation.from_rotvec(goal_pose[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))

    n = max(2, int(duration / dt))
    for i in range(n):
        a    = (i + 1) / n
        pose = start_pose.copy()
        pose[:3] = (1 - a) * start_pose[:3] + a * goal_pose[:3]
        pose[3:] = slerp([a])[0].as_rotvec()
        _servo_toward(arm, is_franka, pose, dt, velocity, acceleration,
                      gain=gain, lookahead_time=lookahead_time)
        time.sleep(dt)


@click.command()
@click.option('--arm', default='ur5', type=click.Choice(['ur5', 'franka'], case_sensitive=False),
              help='Which robotic arm to use: "ur5" (default) or "franka".')
@click.option('--robot_ip',default = None, required=False,
              help='Arm IP. Default: 150.65.146.87 (UR5e) or 172.16.0.2 (Franka).')
@click.option('--frequency', default=10, help='Control frequency (Hz)')
@click.option('--max_pos_speed', default=0.05, help='Max linear speed (m/s)')
@click.option('--max_rot_speed', default=0.1, help='Max angular speed (rad/s)')
@click.option('--speed_scale', default=1, help='Speed scaling factor (0.1-1.0)')
@click.option('--control_mode', default='velocity', type=click.Choice(['velocity', 'position'], case_sensitive=False),
              help='Franka only: drive via set_ee_velocity() ("velocity", default) or '
                   'set_tcp_pose() ("position"). UR5e always streams absolute position '
                   '(servo_tcp_pose) regardless of this flag.')
def main(arm, robot_ip, frequency, max_pos_speed, max_rot_speed, speed_scale, control_mode):
    is_franka = arm.lower() == 'franka'
    control_mode = control_mode.lower()
    _default_ip = {'ur5': '150.65.146.87', 'franka': '172.16.0.2'}
    robot_ip = robot_ip or _default_ip[arm.lower()]

    print("="*60)
    print(f"       {arm.upper()} TELEOPERATION TEST (SpaceMouse)")
    print("="*60)
    if is_franka:
        if control_mode == 'velocity':
            print("\nControl mode: VELOCITY -- linear velocity computed from the SpaceMouse")
            print("each tick is sent directly to set_ee_velocity(). Start with a low")
            print("--max_pos_speed/--max_rot_speed/--speed_scale.\n")
        else:
            print("\nControl mode: POSITION -- target_pose is integrated from the SpaceMouse")
            print("each tick and streamed to set_tcp_pose() (franky/Ruckig plans a fresh")
            print("trajectory toward it every call). Start with a low")
            print("--max_pos_speed/--max_rot_speed/--speed_scale.\n")
        robot = FrankaRobot(robot_ip=robot_ip, use_gripper=False)
    else:
        robot = UR5eRobot(robot_ip=robot_ip, frequency=frequency)
    # keep `robot` as the variable name so the rest of the loop is unchanged,
    # same convention as demo_collect.py

    # Workspace limits (meters) - safety bounds
    WORKSPACE = {
        'x_min': -0.6, 'x_max': 0.6,
        'y_min': -0.6, 'y_max': 0.6,
        'z_min': 0.05, 'z_max': 0.6,  # Min 2cm above table
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

    # Check robot mode (UR5e/RTDE-specific -- Franka has no equivalent
    # "remote control mode" concept to check here)
    if not is_franka:
        robot_mode = robot.get_robot_mode()
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
        robot.disconnect()
        return

    # Control loop parameters
    dt = 1.0 / frequency
    running = True
    paused = False

    # Get initial pose
    tcp_pose = robot.get_tcp_pose()
    print(f"\nCurrent TCP pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    init_pose = INIT_POSE_FRANKA if is_franka else INIT_POSE_UR5E
    target_pose = init_pose.copy()

    move_2_init_pos(robot, tcp_pose, init_pose, dt=dt, velocity=0.05, duration=5.0, gain=150, is_franka=is_franka)
    tcp_pose = robot.get_tcp_pose()
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")


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
                    tcp_pose = robot.get_tcp_pose()
                    target_pose = tcp_pose.copy()
                    print("\nReset to current position")
                elif key == 's':
                    tcp_pose = robot.get_tcp_pose()
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
                    robot.stop()
                time.sleep(0.1)
                continue
            elif paused:
                print("Resuming...")
                paused = False
                # Reset target to current pose when resuming
                tcp_pose = robot.get_tcp_pose()
                target_pose = tcp_pose.copy()

            # Calculate velocity from SpaceMouse
            # sm_state = [x, y, z, rx, ry, rz] normalized ~[-1, 1]
            vel_linear = np.array([
                sm_state[0] * max_pos_speed * speed_scale,  # X
                sm_state[1] * max_pos_speed * speed_scale,  # Y
                sm_state[2] * max_pos_speed * speed_scale  ,  # Z
            ])
            print(f"vel_linear: {vel_linear}")
            vel_angular = np.array([
                sm_state[3] * max_rot_speed * speed_scale,  # rx
                sm_state[4] * max_rot_speed * speed_scale,  # ry
                sm_state[5] * max_rot_speed * speed_scale,  # rz
            ])
            vel_angular = np.array([0, 0, 0])  # Disable rotation for now

            # Update target pose
            target_pose[:3] += vel_linear * dt
            target_pose[3:] += vel_angular * dt

            if np.any(vel_angular != 0):
                drot = st.Rotation.from_euler('xyz', vel_angular)
                current_rot = st.Rotation.from_rotvec(target_pose[3:])
                target_pose[3:] = (drot * current_rot).as_rotvec()

            target_pose[0] = np.clip(target_pose[0], WORKSPACE['x_min'], WORKSPACE['x_max'])
            target_pose[1] = np.clip(target_pose[1], WORKSPACE['y_min'], WORKSPACE['y_max'])
            target_pose[2] = np.clip(target_pose[2], WORKSPACE['z_min'], WORKSPACE['z_max'])

            # print(f"target_pose = {target_pose}")
            # Send command to robot.
            try:
                if is_franka and control_mode == 'position':
                    robot.set_tcp_pose(target_pose, velocity=max_pos_speed * speed_scale,
                                       acceleration=max_pos_speed * speed_scale)
                elif is_franka:
                    current_pose = robot.get_tcp_pose()
                    lin_vel = vel_linear.copy()
                    bounds = (
                        (0, WORKSPACE['x_min'], WORKSPACE['x_max']),
                        (1, WORKSPACE['y_min'], WORKSPACE['y_max']),
                        (2, WORKSPACE['z_min'], WORKSPACE['z_max']),
                    )
                    for axis, lo, hi in bounds:
                        if current_pose[axis] >= hi and lin_vel[axis] > 0:
                            lin_vel[axis] = 0.0
                        elif current_pose[axis] <= lo and lin_vel[axis] < 0:
                            lin_vel[axis] = 0.0

                    robot.set_ee_velocity(lin_vel, angular_velocity=vel_angular,
                                         max_vel=max_pos_speed * speed_scale,
                                         max_ang_vel=max_rot_speed * speed_scale)
                    target_pose = current_pose.copy()  # keep in sync for status/'r'/'s'
                else:
                    # servoL(pose, velocity, acceleration, time, lookahead_time, gain)
                    robot.servo_tcp_pose(target_pose=target_pose,velocity=0.5,
                                      acceleration=0.5,dt=dt,lookahead_time=0.1,gain=300)
            except Exception as e:
                print(f"\nControl error: {e}")
                # Try to recover
                tcp_pose = robot.get_tcp_pose()
                target_pose = tcp_pose.copy()

            # Print status periodically
            iter_count += 1
            if time.time() - last_print_time > 1.0:
                tcp_pose = robot.get_tcp_pose()
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
        robot.disconnect()
        sm.close()

        print("Teleoperation ended.")


if __name__ == '__main__':
    main()
