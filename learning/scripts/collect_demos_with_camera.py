#!/usr/bin/env python3
"""
Data Collection with Camera - Single Process (No Multiprocessing)

Synchronized data collection:
- Robot states
- Gripper states
- Camera RGB frames
- All with SAME timestamp

Usage:
     python scripts/collect_demos_with_camera.py     -o data/real_data     --robot_ip 192.168.11.20     --camera_width 640     --camera_height 480     --camera_fps 30


Controls:
    SpaceMouse:
        - Move: Robot XYZ
        - Left button: Toggle gripper
        - Right button: Rotation mode
    Keyboard:
        - 'C': Start recording
        - 'S': Stop recording
        - 'Q': Quit
"""

import sys
import os
import time
import click
import numpy as np
import zarr
import scipy.spatial.transform as st
from pathlib import Path
import cv2
# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKPLACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PICKPLACE_DIR)

from custom.ur5e_rtde import UR5eRobot
from custom.spacemouse import _build_spacemouse
from custom.flowbot import flowbot
from custom.realsense_camera import RealSenseCamera
# Keyboard
import select
import termios
import tty
import platform
# Camera
try:
    import pyrealsense2 as rs
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️  Warning: pyrealsense2 not installed. Running without camera.")


class DataBuffer:
    """Buffer for collecting episode data with camera"""

    def __init__(self, with_camera=False):
        self.with_camera = with_camera
        self.reset()

    def reset(self):
        self.timestamps = []
        self.robot_states = []
        self.joint_states = []
        self.actions = []
        self.pwm_signals = []
        if self.with_camera:
            self.camera_frames = []  # RGB images

    def add(self, timestamp, robot_state, joint_state, pwm_signals, action, camera_frame=None):
        self.timestamps.append(timestamp)
        self.robot_states.append(robot_state.copy())
        self.joint_states.append(joint_state.copy())
        self.actions.append(action.copy())
        self.pwm_signals.append(pwm_signals.copy())
        if self.with_camera:
            if camera_frame is not None:
                self.camera_frames.append(camera_frame.copy())
            else:
                raise ValueError("Camera frame required when with_camera=True")

    def __len__(self):
        return len(self.timestamps)

    def to_dict(self):
        """Convert to dictionary for zarr"""
        data = {
            'timestamp': np.array(self.timestamps),
            'robot_eef_pose': np.array(self.robot_states),
            'robot_joint': np.array(self.joint_states),
            'pwm_signals': np.array(self.pwm_signals),
            'action': np.array(self.actions),
        }

        if self.with_camera:
            # Stack frames: (T, H, W, 3)
            data['camera_0'] = np.array(self.camera_frames)

        return data


def create_zarr_dataset(output_dir, with_camera=False, image_shape=None):
    """Create zarr dataset"""
    zarr_path = Path(output_dir) / 'dataset.zarr'
    root = zarr.open(str(zarr_path), mode='a')

    if 'data' not in root:
        root.create_group('data')
    if 'meta' not in root:
        meta = root.create_group('meta')
        meta.create_dataset('episode_ends', shape=(0,), dtype=np.int64, chunks=(100,))

    # Store metadata
    if with_camera and 'camera_info' not in root:
        camera_info = root.create_group('camera_info')
        if image_shape:
            camera_info.attrs['image_shape'] = image_shape
            camera_info.attrs['format'] = 'RGB'

    return root


def save_episode(zarr_root, episode_data):
    """Save episode to zarr"""
    data_group = zarr_root['data']
    meta_group = zarr_root['meta']

    episode_ends = meta_group['episode_ends']
    current_len = 0 if len(episode_ends) == 0 else int(episode_ends[-1])

    episode_len = len(episode_data['timestamp'])
    new_len = current_len + episode_len

    # Save each data key
    for key, value in episode_data.items():
        if key not in data_group:
            # Create dataset
            if key == 'camera_0':
                # Images: use compression
                data_group.create_dataset(
                    key,
                    shape=(new_len,) + value.shape[1:],
                    dtype=value.dtype,
                    chunks=(1,) + value.shape[1:],  # Chunk per image
                    compressor=zarr.Blosc(cname='lz4', clevel=3)
                )
            else:
                # Regular data
                data_group.create_dataset(
                    key,
                    shape=(new_len,) + value.shape[1:],
                    dtype=value.dtype,
                    chunks=(100,) + value.shape[1:]
                )
        else:
            # Resize
            dataset = data_group[key]
            # dataset.resize(new_len, *value.shape[1:])
            dataset.resize((new_len,) + value.shape[1:])

        # Write data
        data_group[key][current_len:new_len] = value

    # Update episode_ends
    episode_ends.resize(len(episode_ends) + 1)
    episode_ends[-1] = new_len

    return len(episode_ends) - 1

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
@click.option('--output', '-o', required=True, default = 'data_demo', help='output folder name')
@click.option('--robot_ip', '-ri', required=True, default = '192.168.11.20', help='UR5e IP')
@click.option('--arduino_port', default="/dev/ttyACM0")
@click.option('--camera_serial', default=827112072398, help='RealSense serial (auto-detect if None)')
@click.option('--no_camera', is_flag=True, help='Run without camera')
@click.option('--camera_width', default=640, type=int, help='Camera width')
@click.option('--camera_height', default=480, type=int, help='Camera height')
@click.option('--camera_fps', default=30, type=int, help='Camera FPS')
@click.option('--frequency', '-f', default=10.0, type=float, help='Control Hz')
@click.option('--flowbot_freqency', '-fb_freq', default=30.0, type=float, help='Control Hz for flowbot')
@click.option('--max_pos_speed', default=0.15, type=float)
@click.option('--max_rot_speed', default=0.3, type=float)
@click.option('--deadzone', default=0.1, type=float, help='Spacemouse threshold')
def main(output, robot_ip, camera_serial, no_camera, camera_width, camera_height,
         camera_fps, arduino_port,flowbot_freqency, frequency, max_pos_speed, max_rot_speed,deadzone):

    print("="*60)
    print("   PICK-PLACE DATA COLLECTION WITH CAMERA")
    print("="*60)

    # Create output
    parent_dir = Path("/home/nhnhan/Desktop/flow_contibot_learning/data/")
    output_dir = Path(parent_dir + output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Initialize camera
    camera = None
    with_camera = not no_camera and CAMERA_AVAILABLE

    if with_camera:
        try:
            print("\nInitializing camera...")
            camera = RealSenseCamera(
                serial_number=camera_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                enable_depth=False,
            )
            image_shape = (camera_height, camera_width, 3)
        except Exception as e:
            print(f"⚠️  Camera failed: {e}")
            print("   Continuing without camera...")
            with_camera = False
            camera = None
    else:
        print("\nSkipping camera (--no_camera or not available)")

    # Initialize zarr
    zarr_root = create_zarr_dataset(
        output_dir,
        with_camera=with_camera,
        image_shape=image_shape if with_camera else None
    )

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
    ### Flowbot
    fb = flowbot(serial_port = serial_port,
                 pwm_min= 5,
                 pwm_max= 26,
                 enable_plot = True,
                frequency = flowbot_freqency,
                max_pos_speed = 30)
    fb.start()

    # Connect SpaceMouse
    print("\nConnecting SpaceMouse...")
    sm = _build_spacemouse(os_name=os_name)
    sm.start()
    print("✅ SpaceMouse connected!")

    print("\n" + "="*60)
    print("Controls:")
    print("  SpaceMouse  → Move robot")
    print("  Left btn    → Toggle gripper")
    print("  Right btn   → Rotation mode")
    print("  'C'         → Start recording")
    print("  'S'         → Stop recording")
    print("  'Q'         → Quit")
    print("="*60)

    # Control loop
    dt = 1.0 / frequency
    is_recording = False
    episode_buffer = DataBuffer()
    episode_count = 0
    iter_count = 0

    # Get initial pose
    tcp_pose = ur5.get_tcp_pose()
    init_pose = [0.10267188, -0.4243451 ,  0.2850566,3.14, 0.0 ,0.0]
    target_pose = init_pose.copy()
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    move_2_init_pos(ur5, tcp_pose, init_pose, dt=dt, duration=3.0,gain=150)
    print("\nReady! Press 'C' to start recording.\n")

    # Terminal setup
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            loop_start = time.time()

            # Keyboard
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)

                if key in ['q', 'Q', '\x1b']:
                    print("\n\nQuitting...")
                    break

                elif key in ['c', 'C']:
                    if not is_recording:
                        episode_buffer.reset()
                        is_recording = True
                        print("\n>>> RECORDING STARTED <<<\n")

                elif key in ['s', 'S']:
                    if is_recording and len(episode_buffer) > 0:
                        ep_data = episode_buffer.to_dict()
                        ep_id = save_episode(zarr_root, ep_data)
                        episode_count += 1
                        is_recording = False
                        print(f"\n>>> Episode {ep_id} SAVED ({len(episode_buffer)} steps)")
                        if with_camera:
                            print(f"    Camera: {ep_data['camera_0'].shape}")
                        print(f"    Total episodes: {episode_count}")

                        # Auto-return to start pose
                        print(f"\n🔄 Moving robot back to start pose...")
                        try:
                            tcp_pose = ur5.get_tcp_pose()
                            move_2_init_pos(ur5, tcp_pose, init_pose, dt=dt, duration=3.0,gain=150)
                            print(f"✅ Robot returned to start pose!\n")
                        except Exception as e:
                            print(f"⚠️  Failed to return to start: {e}\n")
                    elif is_recording:
                        print("\n⚠️  No data recorded!\n")
                        is_recording = False

            # Get camera frame FIRST (synchronized timestamp)
            camera_frame = None
            if with_camera and camera:
                try:
                    camera_frame = camera.get_frame()
                except Exception as e:
                    print(f"\n⚠️  Camera error: {e}\n")

            # SpaceMouse
            button_status = sm.get_button_status()
            if button_status[1] and not button_status[0]: ### right button is held
                xyz_fb = sm.get_latest_xyz()
                xyz_fb[2] = -xyz_fb[2] 
                xyz_fb = np.where(np.abs(xyz_fb) < deadzone, 0.0, xyz_fb)
                fb.step(xyz_fb)
            elif button_status[0] and not button_status[1]: ### left button is held
                xyz_ur5 = sm.get_latest_xyz()

                vel_linear = xyz_ur5[:3] * max_pos_speed * dt
                vel_angular = xyz_ur5[3:] * max_rot_speed * dt
                vel_angular[:] = 0
                # if not sm.is_button_pressed(1):
                #     vel_angular[:] = 0
                # else:
                #     vel_linear[:] = 0

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

            # Collect data - ALL with SAME timestamp
            if is_recording:
                if with_camera and camera_frame is None:
                    print("\n⚠️  Warning: No camera frame!\n")
                    continue

                current_tcp = ur5.get_tcp_pose()
                current_joints = ur5.get_joint_angles()
                episode_buffer.add(
                    timestamp=time.time(),
                    robot_state=current_tcp,
                    joint_state=current_joints,
                    action=target_pose,
                    pwm_signals=fb.last_pwm,
                    camera_frame=camera_frame
                )

            # Status
            iter_count += 1
            if iter_count % (frequency * 2) == 0:
                status = "REC" if is_recording else "---"
                n_steps = len(episode_buffer) if is_recording else 0
                cam_str = "CAM" if (with_camera and camera_frame is not None) else "---"
                print(f"[{status}][{cam_str}] iter={iter_count:4d} eps={episode_count} steps={n_steps:3d}")

            # Maintain frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # Cleanup
        print("\nCleaning up...")
        ur5.disconnect()
        fb.stop()
        time.sleep(0.2)
        sm.stop()
        
        if camera:
            camera.stop()

        print(f"\n✅ Done! Collected {episode_count} episodes")
        print(f"Data: {output_dir / 'dataset.zarr'}\n")


if __name__ == '__main__':
    main()
