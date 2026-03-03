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
from zarr.codecs.numcodecs import Blosc
from hardware.ur5e_rtde import UR5eRobot
from hardware.spacemouse import _build_spacemouse
from hardware.flowbot import flowbot
from hardware.realsense_camera import RealSenseCamera
# Keyboard
import select
import termios
import tty
import platform

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
        self.operation_modes = []
        if self.with_camera:
            self.camera_frames = []  # RGB images

    def add(self, timestamp, robot_state, joint_state, pwm_signals, action,
            operation_mode=None, camera_frame=None):
        self.timestamps.append(timestamp)
        self.robot_states.append(robot_state.copy())
        self.joint_states.append(joint_state.copy())
        self.actions.append(action.copy())
        self.pwm_signals.append(pwm_signals.copy())
        if operation_mode is not None:
            self.operation_modes.append(np.array(operation_mode, dtype=np.uint8))
        else:
            self.operation_modes.append(np.array([0, 0], dtype=np.uint8))
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
            'operation_mode': np.array(self.operation_modes, dtype=np.uint8),  # (T, 2)
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
        meta.create_array('episode_ends', shape=(0,), dtype=np.int64, chunks=(100,))

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
    n_eps = episode_ends.shape[0]
    if n_eps == 0:
        current_len = 0
    else:
        current_len = int(episode_ends[n_eps - 1])
    episode_len = episode_data['timestamp'].shape[0]
    new_len = current_len + episode_len

    # Save each data key
    for key, value in episode_data.items():
        if key not in data_group:
            # Create dataset
            if key == 'camera_0':
                # Images: use compression
                data_group.create_array(
                    key,
                    shape=(new_len,) + value.shape[1:],
                    dtype=value.dtype,
                    chunks=(1,) + value.shape[1:],  # Chunk per image
                    compressors=Blosc(cname='lz4', clevel=3),
                )
            else:
                # Regular data
                data_group.create_array(
                    key,
                    shape=(new_len,) + value.shape[1:],
                    dtype=value.dtype,
                    chunks=(100,) + value.shape[1:],
                )
        else:
            # Resize
            dataset = data_group[key]
            # dataset.resize(new_len, *value.shape[1:])
            dataset.resize((new_len,) + value.shape[1:])

        # Write data
        data_group[key][current_len:new_len] = value

    # Update episode_ends
    episode_ends.resize(n_eps + 1)
    episode_ends[-1] = new_len

    return n_eps

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
@click.option('--output', '-o', required=True, default = None, help='output folder name')
@click.option('--robot_ip', '-ri', required=True, default = '150.65.146.87', help='UR5e IP')
@click.option('--arduino_port', default="/dev/ttyACM0")
@click.option('--camera_serial', help='RealSense serial (auto-detect if None)')
@click.option('--no_camera', is_flag=True, help='Run without camera')
@click.option('--camera_width', default=640, type=int, help='Camera width')
@click.option('--camera_height', default=480, type=int, help='Camera height')
@click.option('--camera_fps', default=30, type=int, help='Camera FPS')
@click.option('--frequency', '-f', default=10.0, type=float, help='Control Hz')
@click.option('--flowbot_freqency', '-fb_freq', default=10.0, type=float, help='Control Hz for flowbot')
@click.option('--max_pos_speed', default=0.07, type=float)
@click.option('--max_rot_speed', default=0.05, type=float)
@click.option('--deadzone', default=0.2, type=float, help='Spacemouse threshold')
@click.option('--release_frames', default=10, type=int,
              help='Frames to record after release (both-button press). '
                   'At 10 Hz the default of 10 gives 1 s of released state.')
def main(output, robot_ip, camera_serial, no_camera, camera_width, camera_height,
         camera_fps, arduino_port, flowbot_freqency, frequency, max_pos_speed,
         max_rot_speed, deadzone, release_frames):

    print("="*60)
    print("   PICK-PLACE DATA COLLECTION WITH CAMERA")
    print("="*60)

    # Create output
    if output is None:
        parent_dir = Path(__file__).parent.parent
        output_dir = Path(parent_dir / "data" / "demo_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput: {output_dir}")
    else:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput: {output_dir}")
    # Initialize camera
    camera = None
    with_camera = not no_camera

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
                 pwm_min= 0,
                 pwm_max= 26,
                 enable_plot = True,
                frequency = flowbot_freqency,
                max_pos_speed = 30)
    fb.start()

    # Connect SpaceMouse
    print("\nConnecting SpaceMouse...")
    sm = _build_spacemouse(os_name=os_name, deadzone=deadzone)
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
    episode_buffer = DataBuffer(with_camera=with_camera)
    episode_count = 0
    iter_count = 0

    # Get initial pose
    tcp_pose = ur5.get_tcp_pose()
    init_pose = [0.20636, -0.46706,  0.44268,3.14, -0.14 ,0.0]
    target_pose = init_pose.copy()
    
    move_2_init_pos(ur5, tcp_pose, init_pose, dt=dt, duration=5.0,gain=150)
    tcp_pose = ur5.get_tcp_pose()
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    print("\nReady! Press 'C' to start recording.\n")

    # Terminal setup
    old_settings = termios.tcgetattr(sys.stdin)
    cam_obs = cv2.VideoCapture(0)
    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            loop_start = time.time()
            # Snapshot PWM BEFORE any new command this iteration.
            # The image (read after sleep at step 4) reflects this value, not the new command.
            prev_pwm = fb.last_pwm.copy()

            # ── 1. Keyboard ───────────────────────────────────────────────────
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)

                if key in ['q', 'Q', '\x1b']:
                    print("\n\nQuitting...")
                    break
                elif key in ['r', 'R']:
                    print("\nResetting robot to initial pose...")
                    try:
                        fb.reset()  # Reset flowbot
                        fb.update_plot()
                        tcp_pose = ur5.get_tcp_pose()
                        move_2_init_pos(ur5, tcp_pose, init_pose, dt=dt, duration=3.0, gain=150)
                        print(f"✅ Robot reset to initial pose!\n")
                        target_pose = init_pose.copy()

                        
                    except Exception as e:
                        print(f"⚠️  Failed to reset robot: {e}\n")
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
                            move_2_init_pos(ur5, tcp_pose, init_pose, dt=dt, duration=3.0, gain=150)
                            print(f"✅ Robot returned to start pose!\n")
                            target_pose = init_pose.copy()

                            fb.reset()  # Reset flowbot
                            fb.update_plot()
                        except Exception as e:
                            print(f"⚠️  Failed to return to start: {e}\n")
                    elif is_recording:
                        print("\n⚠️  No data recorded!\n")
                        is_recording = False

            # ── 2. SpaceMouse → send servo command BEFORE sleep ───────────────
            button_status = sm.get_button_status()
            # Determine operation mode from button state
            if button_status[0] and button_status[1]:              # both: release
                op_mode = np.array([1, 1], dtype=np.uint8)
            elif button_status[0] and not button_status[1]:        # left only: UR5
                op_mode = np.array([1, 0], dtype=np.uint8)
            elif button_status[1] and not button_status[0]:        # right only: flowbot
                op_mode = np.array([0, 1], dtype=np.uint8)
            else:
                op_mode = np.array([0, 0], dtype=np.uint8)         # idle

            if button_status[1] and not button_status[0]:          # right btn: flowbot
                xyz_fb = sm.get_latest_xyz()
                xyz_fb[2] = -xyz_fb[2]
                copied_xyz = xyz_fb.copy()
                xyz_fb[1] = -copied_xyz[0]  # for better visualization during teleop
                xyz_fb[0] = -copied_xyz[1]
                xyz_fb = np.where(np.abs(xyz_fb) < deadzone, 0.0, xyz_fb)
                fb.step(xyz_fb)
                fb.update_plot()

            elif button_status[0] and not button_status[1]:        # left btn: UR5e
                xyz_ur5 = sm.get_latest_xyz()
                vel_linear  = xyz_ur5[:3] * max_pos_speed * dt
                vel_angular = xyz_ur5[3:] * max_rot_speed * dt
                vel_angular[:] = 0

                target_pose[:3] += vel_linear
                if np.any(vel_angular != 0):
                    drot = st.Rotation.from_euler('xyz', vel_angular)
                    current_rot = st.Rotation.from_rotvec(target_pose[3:])
                    target_pose[3:] = (drot * current_rot).as_rotvec()
                    

                try:
                    ur5.servo_tcp_pose(target_pose=target_pose, velocity=0.1,
                                       acceleration=0.1, dt=dt, lookahead_time=0.1, gain=300)
                    print(f"Target pose updated: [{', '.join([f'{x:.3f}' for x in target_pose[:3]])}]")
                except Exception as e:
                    print(f"\nControl error: {e}")
                    tcp_pose = ur5.get_tcp_pose()
                    target_pose = tcp_pose.copy()

            elif button_status[0] and button_status[1]:            # both btns: release
                print("======== RELEASING =========")
                fb.reset()        # sets last_pwm = [0,0,0] and sends "0 0 0"
                fb.update_plot()
                fb.release()      # sends 'r' hardware command

                # ── Record release burst so the model learns the end state ────
                # Capture `release_frames` steps at PWM=0 while holding robot
                # position. Without this, the operator pressing 's' immediately
                # after release would save 0 release frames in the episode.
                if is_recording:
                    print(f"  Recording {release_frames} release frames ...")
                    for _ in range(release_frames):
                        # Hold robot at current target during release
                        try:
                            ur5.servo_tcp_pose(target_pose=target_pose, velocity=0.1,
                                               acceleration=0.1, dt=dt,
                                               lookahead_time=0.1, gain=300)
                        except Exception:
                            pass

                        # Sleep → observe (same pattern as main loop step 3→4)
                        time.sleep(dt)

                        rel_frame = None
                        if with_camera and camera:
                            try:
                                rel_frame, _ = camera.get_frames()
                            except Exception:
                                pass

                        if with_camera and rel_frame is None:
                            continue

                        rel_tcp    = ur5.get_tcp_pose()
                        rel_joints = ur5.get_joint_angles()
                        episode_buffer.add(
                            timestamp=time.time(),
                            robot_state=rel_tcp,
                            joint_state=rel_joints,
                            action=target_pose,      # robot not moving
                            pwm_signals=fb.last_pwm, # = [0,0,0] after reset
                            operation_mode=np.array([1, 1], dtype=np.uint8),
                            camera_frame=rel_frame,
                        )
                    print(f"  Release recorded ({release_frames} steps, PWM={fb.last_pwm.tolist()})")

            # ── 3. Sleep BEFORE reading observations ──────────────────────────

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # ── 4. Read ALL observations after robot has settled ──────────────
            camera_frame = None
            if with_camera and camera:
                try:
                    camera_frame, _ = camera.get_frames()
                    # ret, frame = cam_obs.read()
                    # cv2.imshow("Camera", frame)
                    # cv2.waitKey(1)
                except Exception as e:
                    print(f"\n⚠️  Camera error: {e}\n")

            # ── 5. Save to buffer (camera, TCP, PWM all at same settled pose) ─
            # Skip idle frames (no button pressed) — they represent operator hesitation,
            # not intentional actions, and would teach the model to stall mid-task.
            if is_recording and np.any(op_mode):
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
                    pwm_signals=prev_pwm,   # command from previous step (matches current image/tcp)
                    operation_mode=op_mode,
                    camera_frame=camera_frame
                )

            # ── 6. Status ─────────────────────────────────────────────────────
            iter_count += 1
            if iter_count % (frequency * 2) == 0:
                status  = "REC" if is_recording else "---"
                n_steps = len(episode_buffer) if is_recording else 0
                cam_str = "CAM" if (with_camera and camera_frame is not None) else "---"
                print(f"[{status}][{cam_str}] iter={iter_count:4d} eps={episode_count} steps={n_steps:3d}")

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
