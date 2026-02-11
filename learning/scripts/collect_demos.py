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

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKPLACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PICKPLACE_DIR)

from custom.dynamixel_gripper import DynamixelGripper
from custom.spacemouse import SpaceMouse
from custom.realsense_camera import RealSenseCamera

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
        self.gripper_states = []
        self.actions = []  # Commanded poses

    def add(self, timestamp, robot_state, joint_state, gripper_state, action):
        self.timestamps.append(timestamp)
        self.robot_states.append(robot_state.copy())
        self.joint_states.append(joint_state.copy())
        self.gripper_states.append(gripper_state)
        self.actions.append(action.copy())

    def __len__(self):
        return len(self.timestamps)

    def to_dict(self):
        """Convert to dictionary for zarr storage"""
        return {
            'timestamp': np.array(self.timestamps),
            'robot_eef_pose': np.array(self.robot_states),  # End-effector pose
            'robot_joint': np.array(self.joint_states),
            'gripper_position': np.array(self.gripper_states),
            'action': np.array(self.actions),
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

    # Get current total length
    episode_ends = meta_group['episode_ends']
    if len(episode_ends) == 0:
        current_len = 0
    else:
        current_len = int(episode_ends[-1])

    episode_len = len(episode_data['timestamp'])
    new_len = current_len + episode_len

    # Append data
    for key, value in episode_data.items():
        if key not in data_group:
            # Create dataset
            data_group.create_dataset(
                key,
                shape=(new_len,) + value.shape[1:],
                dtype=value.dtype,
                chunks=(100,) + value.shape[1:]
            )
        else:
            # Resize and append
            dataset = data_group[key]
            dataset.resize(new_len, *value.shape[1:])

        # Write data
        data_group[key][current_len:new_len] = value

    # Update episode_ends
    episode_ends.resize(len(episode_ends) + 1)
    episode_ends[-1] = new_len

    return len(episode_ends) - 1  # Episode ID


@click.command()
@click.option('-o', '--output', required=True, help='Output directory')
@click.option('--robot_ip', '-ri', required=True, help='UR5e IP')
@click.option('--gripper_port', default='/dev/ttyUSB0')
@click.option('--gripper_id', default=7, type=int)
@click.option('--frequency', '-f', default=10.0, type=float, help='Control Hz')
@click.option('--max_pos_speed', default=0.15, type=float, help='Max linear m/s')
@click.option('--max_rot_speed', default=0.3, type=float, help='Max angular rad/s')
def main(output, robot_ip, gripper_port, gripper_id, frequency, max_pos_speed, max_rot_speed):

    print("="*60)
    print("   SIMPLE PICK-PLACE DATA COLLECTION")
    print("="*60)

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize zarr dataset
    print(f"\nOutput: {output_dir}")
    zarr_root = create_zarr_dataset(output_dir)

    # Connect to robot
    print(f"\nConnecting to robot at {robot_ip}...")
    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        rtde_c = RTDEControlInterface(robot_ip)
        rtde_r = RTDEReceiveInterface(robot_ip)
        print("✅ Robot connected!")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Initialize gripper
    print(f"\nInitializing gripper (ID {gripper_id})...")
    gripper = DynamixelGripper(
        port=gripper_port,
        dxl_id=gripper_id,
        position_open=10,
        position_close=900
    )
    gripper.open()
    gripper_is_open = True
    print("✅ Gripper ready!")

    # Connect SpaceMouse
    print("\nConnecting SpaceMouse...")
    sm = SpaceMouse(deadzone=0.15, max_value=350)
    print("✅ SpaceMouse connected!")

    print("\n" + "="*60)
    print("Controls:")
    print("  SpaceMouse      → Move robot")
    print("  Left button     → Toggle gripper")
    print("  Right button    → Hold for rotation")
    print("  'C'             → Start recording")
    print("  'S'             → Stop recording")
    print("  'Q'             → Quit")
    
    print("="*60)

    # Get initial pose
    tcp_pose = np.array(rtde_r.getActualTCPPose())
    target_pose = tcp_pose.copy()
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    print("\nReady! Press 'C' to start recording.\n")

    # Control loop
    dt = 1.0 / frequency
    is_recording = False
    episode_buffer = DataBuffer()
    episode_count = 0
    iter_count = 0
    prev_button_0 = False

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
            sm_state = sm.get_motion_state_transformed()

            # Calculate velocities
            vel_linear = sm_state[:3] * max_pos_speed * dt
            vel_angular = sm_state[3:] * max_rot_speed * dt

            # Button 1 (Right): rotation mode
            if not sm.is_button_pressed(1):
                vel_angular[:] = 0
            else:
                vel_linear[:] = 0

            # Button 0 (Left): toggle gripper
            button_0 = sm.is_button_pressed(0)
            if button_0 and not prev_button_0:
                gripper_is_open = not gripper_is_open
                if gripper_is_open:
                    gripper.open()
                    print("  Gripper: OPEN")
                else:
                    gripper.close()
                    print("  Gripper: CLOSE")
            prev_button_0 = button_0

            # Update target pose
            target_pose[:3] += vel_linear
            if np.any(vel_angular != 0):
                drot = st.Rotation.from_euler('xyz', vel_angular)
                current_rot = st.Rotation.from_rotvec(target_pose[3:])
                target_pose[3:] = (drot * current_rot).as_rotvec()

            # Execute command
            rtde_c.servoL(
                target_pose.tolist(),
                0.5, 0.5,  # Not used
                dt,
                0.1,  # Lookahead
                300   # Gain
            )

            # Collect data if recording
            if is_recording:
                current_tcp = np.array(rtde_r.getActualTCPPose())
                current_joints = np.array(rtde_r.getActualQ())
                gripper_pos = gripper.get_position()

                episode_buffer.add(
                    timestamp=time.time(),
                    robot_state=current_tcp,
                    joint_state=current_joints,
                    gripper_state=gripper_pos,
                    action=target_pose
                )

            # Status print
            iter_count += 1
            if iter_count % (frequency * 2) == 0:  # Every 2 seconds
                status = "REC" if is_recording else "---"
                gripper_pct = gripper.get_position() * 100
                n_steps = len(episode_buffer) if is_recording else 0
                print(f"[{status}] iter={iter_count:4d} eps={episode_count} "
                      f"steps={n_steps:3d} grip={gripper_pct:5.1f}%")

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
        rtde_c.servoStop()
        rtde_c.stopScript()
        rtde_c.disconnect()
        gripper.open()
        time.sleep(0.2)
        gripper.disconnect()
        sm.close()

        print(f"\n✅ Done! Collected {episode_count} episodes")
        print(f"Data saved to: {output_dir / 'dataset.zarr'}\n")


if __name__ == '__main__':
    main()
