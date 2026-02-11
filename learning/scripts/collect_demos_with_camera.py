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

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKPLACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PICKPLACE_DIR)

from custom.dynamixel_gripper import DynamixelGripper
from custom.spacemouse import SpaceMouse

# Keyboard
import select
import termios
import tty

# Camera
try:
    import pyrealsense2 as rs
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    print("⚠️  Warning: pyrealsense2 not installed. Running without camera.")


class RealsenseCamera:
    """Simple RealSense camera wrapper - single process"""

    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        if not CAMERA_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed")

        self.width = width
        self.height = height
        self.fps = fps

        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable specific camera if serial provided
        if serial_number:
            self.config.enable_device(serial_number)

        # Enable RGB stream only (no depth for now - faster)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

        # Start pipeline
        print(f"  Starting RealSense {width}x{height} @ {fps}fps...")
        try:
            self.pipeline.start(self.config)

            # Warm up - skip first few frames
            for _ in range(10):
                self.pipeline.wait_for_frames()

            print("  ✅ Camera ready!")
        except Exception as e:
            raise RuntimeError(f"Failed to start camera: {e}")

    def get_frame(self):
        """Get RGB frame as numpy array (H, W, 3) uint8"""
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()

        if not color_frame:
            return None

        # Convert to numpy (H, W, 3) RGB
        frame = np.asanyarray(color_frame.get_data())
        return frame

    def stop(self):
        """Stop camera"""
        self.pipeline.stop()


class DataBuffer:
    """Buffer for collecting episode data with camera"""

    def __init__(self, with_camera=False):
        self.with_camera = with_camera
        self.reset()

    def reset(self):
        self.timestamps = []
        self.robot_states = []
        self.joint_states = []
        self.gripper_states = []
        self.actions = []
        if self.with_camera:
            self.camera_frames = []  # RGB images

    def add(self, timestamp, robot_state, joint_state, gripper_state, action, camera_frame=None):
        self.timestamps.append(timestamp)
        self.robot_states.append(robot_state.copy())
        self.joint_states.append(joint_state.copy())
        self.gripper_states.append(gripper_state)
        self.actions.append(action.copy())

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
            'gripper_position': np.array(self.gripper_states),
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
            dataset.resize(new_len, *value.shape[1:])

        # Write data
        data_group[key][current_len:new_len] = value

    # Update episode_ends
    episode_ends.resize(len(episode_ends) + 1)
    episode_ends[-1] = new_len

    return len(episode_ends) - 1


@click.command()
@click.option('-o', '--output', required=True, help='Output directory')
@click.option('--robot_ip', '-ri', required=True, help='UR5e IP')
@click.option('--camera_serial', default=None, help='RealSense serial (auto-detect if None)')
@click.option('--no_camera', is_flag=True, help='Run without camera')
@click.option('--camera_width', default=640, type=int, help='Camera width')
@click.option('--camera_height', default=480, type=int, help='Camera height')
@click.option('--camera_fps', default=30, type=int, help='Camera FPS')
@click.option('--gripper_port', default='/dev/ttyUSB0')
@click.option('--gripper_id', default=7, type=int)
@click.option('--frequency', '-f', default=10.0, type=float, help='Control Hz')
@click.option('--max_pos_speed', default=0.15, type=float)
@click.option('--max_rot_speed', default=0.3, type=float)
def main(output, robot_ip, camera_serial, no_camera, camera_width, camera_height,
         camera_fps, gripper_port, gripper_id, frequency, max_pos_speed, max_rot_speed):

    print("="*60)
    print("   PICK-PLACE DATA COLLECTION WITH CAMERA")
    print("="*60)

    # Create output
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Initialize camera
    camera = None
    with_camera = not no_camera and CAMERA_AVAILABLE

    if with_camera:
        try:
            print("\nInitializing camera...")
            camera = RealsenseCamera(
                serial_number=camera_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps
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

    # SpaceMouse
    print("\nConnecting SpaceMouse...")
    sm = SpaceMouse(deadzone=0.15, max_value=350)
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

    # Get initial pose (will be used as start pose for auto-return)
    tcp_pose = np.array(rtde_r.getActualTCPPose())
    start_pose = tcp_pose.copy()  # Save start pose for auto-return
    target_pose = tcp_pose.copy()
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    print(f"Camera: {'Enabled' if with_camera else 'Disabled'}")
    print("\nReady! Press 'C' to start.\n")

    # Control loop
    dt = 1.0 / frequency
    is_recording = False
    episode_buffer = DataBuffer(with_camera=with_camera)
    episode_count = 0
    iter_count = 0
    prev_button_0 = False

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
                            # Open gripper first
                            gripper.open()
                            gripper_is_open = True
                            time.sleep(0.2)

                            # Stop servo mode before using moveL
                            rtde_c.servoStop()
                            time.sleep(0.1)

                            # Move to start pose using moveL (smooth motion)
                            # Use positional args: moveL(pose, speed, acceleration, asynchronous)
                            rtde_c.moveL(start_pose.tolist(), 0.2, 0.5, False)

                            # Update target_pose to match start_pose
                            target_pose[:] = start_pose

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
            sm_state = sm.get_motion_state_transformed()
            vel_linear = sm_state[:3] * max_pos_speed * dt
            vel_angular = sm_state[3:] * max_rot_speed * dt

            if not sm.is_button_pressed(1):
                vel_angular[:] = 0
            else:
                vel_linear[:] = 0

            # Gripper
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

            # Update target
            target_pose[:3] += vel_linear
            if np.any(vel_angular != 0):
                drot = st.Rotation.from_euler('xyz', vel_angular)
                current_rot = st.Rotation.from_rotvec(target_pose[3:])
                target_pose[3:] = (drot * current_rot).as_rotvec()

            # Execute
            rtde_c.servoL(target_pose.tolist(), 0.5, 0.5, dt, 0.1, 300)

            # Collect data - ALL with SAME timestamp
            if is_recording:
                if with_camera and camera_frame is None:
                    print("\n⚠️  Warning: No camera frame!\n")
                    continue

                current_time = time.time()  # Single timestamp for all
                current_tcp = np.array(rtde_r.getActualTCPPose())
                current_joints = np.array(rtde_r.getActualQ())
                gripper_pos = gripper.get_position()

                episode_buffer.add(
                    timestamp=current_time,
                    robot_state=current_tcp,
                    joint_state=current_joints,
                    gripper_state=gripper_pos,
                    action=target_pose,
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
        rtde_c.servoStop()
        rtde_c.stopScript()
        rtde_c.disconnect()
        gripper.open()
        time.sleep(0.2)
        gripper.disconnect()
        sm.close()

        if camera:
            camera.stop()

        print(f"\n✅ Done! Collected {episode_count} episodes")
        print(f"Data: {output_dir / 'dataset.zarr'}\n")


if __name__ == '__main__':
    main()
