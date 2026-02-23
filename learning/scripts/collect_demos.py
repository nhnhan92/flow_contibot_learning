#!/usr/bin/env python3
"""
Simple Data Collection for Pick-Place Task
WITHOUT RealEnv - Direct RTDE control

Usage:
    python scripts/collect_demos.py -o data/demos --robot_ip 192.168.11.20

Controls:
    SpaceMouse:
        - Move: Robot XYZ position
        - Button LEFT: Toggle gripper
        - Button RIGHT: Hold for rotation mode

    Keyboard:
        - 'C': Start recording
        - 'S': Stop recording
        - 'Q': Quit
        - Backspace: Delete last episode
"""

import sys
import os
import time
import click
import numpy as np
import zarr
import scipy.spatial.transform as st
from pathlib import Path
import platform
# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKPLACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PICKPLACE_DIR)

from custom.spacemouse import SpaceMouse
# from custom.realsense_camera import RealSenseCamera
from custom.spacemouse import _build_spacemouse
from custom.flowbot import flowbot
from custom.ur5e_rtde import UR5eRobot
# Keyboard input
import select
import termios
import tty


class DataBuffer:
    """Simple buffer for collecting episode data"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.timestamps = []
        self.robot_states = []  # TCP poses
        self.joint_states = []
        self.actions = []  # Commanded poses
        self.pwm_signals = []

    def add(self, timestamp, robot_state, joint_state, action,pwm_signals):
        self.timestamps.append(timestamp)
        self.robot_states.append(robot_state.copy())
        self.joint_states.append(joint_state.copy())
        self.actions.append(action.copy())
        self.pwm_signals.append(pwm_signals.copy())

    def __len__(self):
        return len(self.timestamps)

    def to_dict(self):
        """Convert to dictionary for zarr storage"""
        return {
            'timestamp': np.array(self.timestamps),
            'robot_eef_pose': np.array(self.robot_states),  # End-effector pose
            'robot_joint': np.array(self.joint_states),
            'action': np.array(self.actions),
            'pwm_signals': np.array(self.pwm_signals)
        }


def create_zarr_dataset(output_dir):
    """Create zarr dataset structure"""
    zarr_path = Path(output_dir) / 'dataset.zarr'

    # Create root
    root = zarr.open(str(zarr_path), mode='a')

    # Create groups if they don't exist
    if 'data' not in root:
        root.create_group('data')
    if 'meta' not in root:
        meta = root.create_group('meta')
        meta.create_dataset('episode_ends', shape=(0,), dtype=np.int64, chunks=(100,))

    return root


def save_episode(zarr_root, episode_data):
    """Save one episode to zarr"""
    data_group = zarr_root['data']
    meta_group = zarr_root['meta']

    episode_ends = meta_group['episode_ends']
    n_eps = episode_ends.shape[0]   # <-- instead of len(episode_ends)

    if n_eps == 0:
        current_len = 0
    else:
        current_len = int(episode_ends[n_eps - 1])

    episode_len = episode_data['timestamp'].shape[0]
    new_len = current_len + episode_len

    # Append data
    for key, value in episode_data.items():
        if key not in data_group:
            data_group.create_dataset(
                key,
                shape=(new_len,) + value.shape[1:],
                dtype=value.dtype,
                chunks=(100,) + value.shape[1:]
            )
        else:
            dataset = data_group[key]
            # dataset.resize(new_len, *value.shape[1:])
            dataset.resize((new_len,) + value.shape[1:])



        data_group[key][current_len:new_len] = value

    # Update episode_ends
    episode_ends.resize(n_eps + 1)     # <-- use n_eps, not len()
    episode_ends[n_eps] = new_len      # last element

    return n_eps  # Episode ID (0-based)

def move_2_init_pos(ur5, start_pose, goal_pose, dt, duration=5.0,
                      velocity=0.1, acceleration=0.1, gain=200, lookahead_time=0.15):
    start_pose = np.asarray(start_pose, dtype=float).copy()
    goal_pose  = np.asarray(goal_pose, dtype=float).copy()

    # interpolate rotation with slerp for stability
    r0 = st.Rotation.from_rotvec(start_pose[3:])
    r1 = st.Rotation.from_rotvec(goal_pose[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))

    n = max(2, int(duration / dt))
    for i in range(n):
        a = (i + 1) / n

        pose = start_pose.copy()
        pose[:3] = (1 - a) * start_pose[:3] + a * goal_pose[:3]
        pose[3:] = slerp([a])[0].as_rotvec()

        ur5.servo_tcp_pose(
            target_pose=pose,
            velocity=velocity,
            acceleration=acceleration,
            dt=dt,
            lookahead_time=lookahead_time,
            gain=gain
        )
        time.sleep(dt)


@click.command()
@click.option('--robot_ip', '-ri', default = '192.168.11.20',required=False, help='UR5e IP')
@click.option('--arduino_port', default="/dev/ttyACM0")
@click.option('--frequency', '-f', default=10.0, type=float, help='Control Hz')
@click.option('--max_pos_speed', default=0.1, type=float, help='Max linear m/s')
@click.option('--max_rot_speed', default=0.1, type=float, help='Max angular rad/s')
@click.option('--deadzone', default=0.1, type=float, help='Spacemouse threshold')
def main(robot_ip, deadzone,arduino_port, frequency, max_pos_speed, max_rot_speed):

    print("="*60)
    print("   SIMPLE PICK-PLACE DATA COLLECTION")
    print("="*60)

    # Create output directory
    # parent_dir = Path(__file__).parent.parent
    output_dir = Path("/home/protac/Desktop/flow_contibot_learning/data/demo_data/")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize zarr dataset
    print(f"\nOutput: {output_dir}")
    zarr_root = create_zarr_dataset(output_dir)

    # Connect to robot
    print(f"\nConnecting to robot at {robot_ip}...")
    ur5 = UR5eRobot(robot_ip=robot_ip,frequency=frequency)


    # Initialize Flowbot
    print(f"\nInitializing Flotbot ...")
    os_name = platform.system().lower()
    if "linux" in os_name:
        serial_port = arduino_port
    elif "windows" in os_name:
        serial_port = "COM9"
    CONTROL_HZ = 30.0          # integrate pc at this rate

    ### Flowbot
    fb = flowbot(serial_port = serial_port,
                 pwm_min= 5,
                 pwm_max= 26,
                 enable_plot = True,
                frequency = CONTROL_HZ,
                max_pos_speed = 30)
    fb.start()

    # Connect SpaceMouse
    print("\nConnecting SpaceMouse...")
    sm = _build_spacemouse(os_name=os_name)
    sm.start()
    print("✅ SpaceMouse connected!")

    print("\n" + "="*60)
    print("Controls:")
    print("  SpaceMouse      → Move robot")
    print("  Left button     → Toggle Soft Manipulator")
    print("  Right button    → Hold for controlling UR5")
    print("  'C'             → Start recording")
    print("  'S'             → Stop recording")
    print("  'Q'             → Quit")
    
    print("="*60)
    # Control loop
    dt = 1.0 / frequency
    is_recording = False
    episode_buffer = DataBuffer()
    episode_count = 0
    iter_count = 0
    prev_button_0 = False

    # Get initial pose
    tcp_pose = ur5.get_tcp_pose()
    target_pose = tcp_pose.copy()
    target_pose = [0.10267188, -0.4243451 ,  0.2850566,3.14, 0.0 ,0.0]
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    move_2_init_pos(ur5, tcp_pose, target_pose, dt=dt, duration=3.0,gain=150)
    print("\nReady! Press 'C' to start recording.\n")

    

    # Setup terminal for keyboard input
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            loop_start = time.time()

            # Check keyboard
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)

                if key in ['q', 'Q', '\x1b']:  # Q or ESC
                    print("\n\nQuitting...")
                    break

                elif key in ['c', 'C']:
                    if not is_recording:
                        episode_buffer.reset()
                        is_recording = True
                        print("\n>>> RECORDING STARTED <<<\n")

                elif key in ['s', 'S']:
                    if is_recording and len(episode_buffer) > 0:
                        # Save episode
                        ep_data = episode_buffer.to_dict()
                        ep_id = save_episode(zarr_root, ep_data)
                        episode_count += 1
                        is_recording = False
                        print(f"\n>>> Episode {ep_id} SAVED ({len(episode_buffer)} steps) <<<")
                        print(f"Total episodes: {episode_count}\n")
                    elif is_recording:
                        print("\n⚠️  No data recorded yet!\n")
                        is_recording = False
                    else:
                        print("\n⚠️  Not recording!\n")

                elif key == '\x7f':  # Backspace
                    if episode_count > 0:
                        print("\n⚠️  Delete last episode? (manual in zarr file)\n")

            # Get SpaceMouse state
            
            button_status = sm.get_button_status()
            if button_status[1] and not button_status[0]: ### right button is held
                xyz_fb = sm.get_latest_xyz()
                xyz_fb[2] = -xyz_fb[2] 
                xyz_fb = np.where(np.abs(xyz_fb) < deadzone, 0.0, xyz_fb)
                fb.step(xyz_fb)

                fb.update_plot()   # just update handles + one pause

            elif button_status[0] and not button_status[1]: ### left button is held
                xyz_ur5 = sm.get_latest_xyz()

                # Calculate velocities
                vel_linear = xyz_ur5[:3] * max_pos_speed * dt
                vel_angular = xyz_ur5[3:] * max_rot_speed * dt
                vel_angular[:] = 0
                # Update target pose
                target_pose[:3] += vel_linear
                if np.any(vel_angular != 0):
                    drot = st.Rotation.from_euler('xyz', vel_angular)
                    current_rot = st.Rotation.from_rotvec(target_pose[3:])
                    target_pose[3:] = (drot * current_rot).as_rotvec()
                # Execute command
                try:
                    ur5.servo_tcp_pose(target_pose=target_pose,velocity=0.1,
                                    acceleration=0.1,dt=dt,lookahead_time=0.1,gain=300)
                except Exception as e:
                    print(f"\nControl error: {e}")
                    # Try to recover
                    tcp_pose = ur5.get_tcp_pose()
                    target_pose = tcp_pose.copy()
            elif button_status[0] and button_status[1]:
                print("======== RELEASING =========")
                fb.release()
            # Button 1 (Right): rotation mode
            # if not sm.is_button_pressed(1):
            #     vel_angular[:] = 0
            # else:
            #     vel_linear[:] = 0

            # Collect data if recording
            if is_recording:
                current_tcp = ur5.get_tcp_pose()
                current_joints = ur5.get_joint_angles()
                episode_buffer.add(
                    timestamp=time.time(),
                    robot_state=current_tcp,
                    joint_state=current_joints,
                    action=target_pose,
                    pwm_signals=fb.last_pwm,
                )

            # Status print
            iter_count += 1
            if iter_count % (frequency * 2) == 0:  # Every 2 seconds
                status = "REC" if is_recording else "---"
                n_steps = len(episode_buffer) if is_recording else 0
                print(f"[{status}] iter={iter_count:4d} eps={episode_count} "
                      f"steps={n_steps:3d}")

            # Maintain frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        # Restore terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Cleanup
        print("\nCleaning up...")
        ur5.disconnect()
        fb.stop()
        sm.stop()

        print(f"\n✅ Done! Collected {episode_count} episodes")
        print(f"Data saved to: {output_dir / 'dataset.zarr'}\n")


if __name__ == '__main__':
    main()
