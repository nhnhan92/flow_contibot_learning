#!/usr/bin/env python3
"""
Data Collection with Camera - Single Process (No Multiprocessing)

Synchronized data collection:
- Robot states
- Gripper states
- Camera RGB frames
- All with SAME timestamp

Usage:
     python demo_collect.py -o data/real_data --arm ur5 --robot_ip 150.65.146.87
     python demo_collect.py -o data/real_data --arm franka --robot_ip 172.16.0.2

Franka: driven via FrankaRobot (franky -- see learning/hardware/franka_robot.py).
    FrankaController (pylibfranka, learning/hardware/franka_control.py) was
    evaluated as an alternative backend and is kept in the tree for
    reference, but is no longer wired up here.

    Recorded "action" differs per arm -- a checkpoint trained on one arm is
    not valid for the other:
        UR5e:   (6,) the arm's absolute target TCP pose.
        Franka: (7,) joint velocities (rad/s), even though the live control
                input stays Cartesian (spacemouse -> set_ee_velocity()).
                franky/libfranka resolve that Cartesian command into joint
                velocities internally every control cycle; get_joint_velocities()
                reads back what was actually executed. Training/deployment
                then work purely in joint space (set_joint_velocity()),
                never inverting the Jacobian again at deploy time -- see
                hardware/franka_robot.py's get_joint_velocities() docstring.

Controls:
    SpaceMouse:
        - Move: Robot XYZ
        - Left button: Move robot (UR5e/Franka)
        - Right button: Move Flowbot
        - Both buttons: Release (Flowbot depressurizes; robot holds position)
    Keyboard:
        - 'C': Start recording
        - 'S': Stop recording (also saves the episode and returns to init_pose)
        - 'R': Reset robot + Flowbot to init_pose without saving
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
from zarr.codecs.numcodecs import Blosc
from hardware.ur5e_rtde import UR5eRobot
from hardware.franka_robot import FrankaRobot
from hardware.spacemouse import _build_spacemouse
from hardware.flowbot import flowbot
from hardware.realsense_camera import RealSenseCamera
# Keyboard
import select
import termios
import tty
import platform

class DataBuffer:
    """Buffer for collecting episode data with camera(s).

    Global and wrist cameras are independently optional -- either, both, or
    neither can be active for a given run (see --no_camera_global/
    --no_camera_wrist). camera_0 = global camera (matches the existing
    single-camera dataset schema every downstream training/analysis script
    hardcodes -- see learning/train/dataset.py etc.). camera_1 = wrist
    camera, a new, additive stream: it's recorded alongside camera_0 but the
    training pipeline doesn't consume it yet -- that's a separate follow-up
    if/when the model is updated to take both views.
    """

    def __init__(self, with_camera_global=False, with_camera_wrist=False):
        self.with_camera_global = with_camera_global
        self.with_camera_wrist = with_camera_wrist
        self.reset()

    def reset(self):
        self.timestamps = []
        self.robot_states = []
        self.joint_states = []
        self.actions = []
        self.pwm_signals = []
        self.operation_modes = []
        if self.with_camera_global:
            self.camera_frames = []        # RGB images, global camera (camera_0)
        if self.with_camera_wrist:
            self.camera_frames_wrist = []  # RGB images, wrist camera (camera_1)

    def add(self, timestamp, robot_state, joint_state, pwm_signals, action,
            operation_mode=None, camera_frame=None, camera_frame_wrist=None):
        self.timestamps.append(timestamp)
        self.robot_states.append(robot_state.copy())
        self.joint_states.append(joint_state.copy())
        self.actions.append(action.copy())
        self.pwm_signals.append(pwm_signals.copy())
        if operation_mode is not None:
            self.operation_modes.append(np.array(operation_mode, dtype=np.uint8))
        else:
            self.operation_modes.append(np.array([0, 0], dtype=np.uint8))
        if self.with_camera_global:
            if camera_frame is None:
                raise ValueError("Global camera frame required when with_camera_global=True")
            self.camera_frames.append(camera_frame.copy())
        if self.with_camera_wrist:
            if camera_frame_wrist is None:
                raise ValueError("Wrist camera frame required when with_camera_wrist=True")
            self.camera_frames_wrist.append(camera_frame_wrist.copy())

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

        if self.with_camera_global:
            # Stack frames: (T, H, W, 3)
            data['camera_0'] = np.array(self.camera_frames)
        if self.with_camera_wrist:
            data['camera_1'] = np.array(self.camera_frames_wrist)

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
            if key in ('camera_0', 'camera_1'):
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
    Control input stays Cartesian -- but see Returns below, the recorded
    action is not.

    Returns
    -------
    np.ndarray -- the command actually sent this tick, meant to be
    recorded as the dataset "action" so training/deployment stay
    consistent with what the robot really executes (not what was merely
    intended/interpolated):
        UR5e:   (6,) the absolute target_pose itself (servo_tcp_pose tracks
                an absolute target, so that *is* the executed command).
        Franka: (7,) joint velocities (rad/s) -- get_joint_velocities()
                read back immediately after commanding the Cartesian
                velocity above, i.e. what franky/libfranka's internal
                Cartesian-to-joint resolution actually executed this tick,
                not the Cartesian command itself. Deploy time then drives
                Franka purely in joint space via set_joint_velocity(),
                never inverting the Jacobian again -- see
                hardware/franka_robot.py's get_joint_velocities() docstring
                for why. Franka-only: UR5e keeps a Cartesian action.
    """
    if is_franka:
        current_pose = arm.get_tcp_pose()
        lin_vel = (np.asarray(target_pose[:3], dtype=float) - current_pose[:3]) / dt
        speed = float(np.linalg.norm(lin_vel))
        if speed > velocity and speed > 1e-9:
            lin_vel = lin_vel / speed * velocity
        ang_vel = np.zeros(3)
        arm.set_ee_velocity(lin_vel, angular_velocity=ang_vel,
                             max_vel=velocity, max_ang_vel=acceleration)
        return arm.get_joint_velocities()
    else:
        arm.servo_tcp_pose(
            target_pose=target_pose, velocity=velocity, acceleration=acceleration,
            dt=dt, lookahead_time=lookahead_time, gain=gain,
        )
        return np.asarray(target_pose, dtype=float).copy()


def move_2_init_pos(arm, start_pose, goal_pose, dt, duration=5.0,
                    velocity=0.05, acceleration=0.1, gain=200, lookahead_time=0.15,
                    is_franka=False):
    """Move arm from start_pose to goal_pose, interpolating (position lerp +
    rotation slerp) over `duration` seconds and streaming each waypoint --
    UR5eRobot via servo_tcp_pose (RTDE servoL) at ~1/dt Hz, FrankaRobot
    (franky) via _servo_toward's error-based feed-forward set_ee_velocity(),
    same as live spacemouse driving.

    Franka used to take a shortcut here: one direct move_tcp_pose() call
    (a single blocking CartesianMotion covering the whole distance). That's
    fine for a short, nearby move, but after teleoperating the arm away
    from goal_pose to some arbitrary reached position, the straight-line
    Cartesian path back can pass close to a kinematic singularity, where
    joint velocities spike (J^-1 * cartesian_velocity blows up) no matter
    how conservatively translation/rotation/elbow dynamics are scaled --
    tripping libfranka's cartesian_motion_generator_*_discontinuity /
    joint_velocity_discontinuity reflexes. Interpolating through many small
    waypoints instead reuses the exact mechanism that already drives the
    arm robustly during live teleoperation (see _servo_toward /
    demo_collect.py's spacemouse drive branch), rather than trusting a
    single big automatic move to navigate whatever path it computes."""
    start_pose = np.asarray(start_pose, dtype=float).copy()
    goal_pose  = np.asarray(goal_pose,  dtype=float).copy()

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
@click.option('--output', '-o', required=True, default=None, help='Output folder name')
@click.option('--arm', default='franka', type=click.Choice(['ur5', 'franka'], case_sensitive=False),
              help='Which robotic arm to use: "ur5" (default) or "franka".')
@click.option('--robot_ip', '-ri', default=None,
              help='Arm IP. Default: 150.65.146.87 (UR5) or 172.16.0.2 (Franka).')
@click.option('--arduino_port', default="/dev/ttyACM0")
@click.option('--camera_serial_global', default='051222061185',
              help='RealSense serial for the global (scene) camera.')
@click.option('--camera_serial_wrist', default='827112072398',
              help='RealSense serial for the wrist camera.')
@click.option('--no_camera_wrist', is_flag=True, help='Run without wrist_camera')
@click.option('--no_camera_global', is_flag=True, help='Run without global_camera')
@click.option('--camera_width', default=640, type=int, help='Camera width (both cameras)')
@click.option('--camera_height', default=480, type=int, help='Camera height (both cameras)')
@click.option('--camera_fps', default=30, type=int,
              help='Camera FPS (both cameras). Must be a rate the sensor natively supports '
                   '(RealSense color streams typically only offer 6/15/30/60) -- pipeline.start() '
                   'fails with "Couldn\'t resolve requests" for any other value. This does not need '
                   'to match --frequency: get_frames() is only called once per control tick regardless '
                   'of the sensor\'s configured rate, so the effective capture rate already follows '
                   '--frequency. Use system_verification/test_camera.py to check what a given camera supports.')
@click.option('--frequency', '-f', default=10.0, type=float, help='Control Hz')
@click.option('--flowbot_freqency', '-fb_freq', default=30.0, type=float, help='Control Hz for flowbot')
@click.option('--flowbot_speed_factor', '-fspeed', default =1.5, type=float)
@click.option('--max_pos_speed', default=0.05, type=float)
@click.option('--max_rot_speed', default=0.05, type=float)
@click.option('--deadzone', default=0.1, type=float, help='Spacemouse threshold')
@click.option('--release_frames', default=10, type=int,
              help='Frames to record after release (both-button press). '
                   'At 10 Hz the default of 10 gives 1 s of released state.')
def main(output, arm, robot_ip, camera_serial_global, camera_serial_wrist, no_camera_wrist,no_camera_global,
         camera_width, camera_height, camera_fps, arduino_port, flowbot_freqency,
         flowbot_speed_factor, frequency, max_pos_speed, max_rot_speed, deadzone, release_frames):

    print("="*60)
    print("   PICK-PLACE DATA COLLECTION WITH CAMERA")
    print("="*60)
    parent_dir = Path(__file__).parent.parent
    # Create output
    if output is None:
        
        output_dir = Path(parent_dir / "data" / "demo_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput: {output_dir}")
    else:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput: {output_dir}")
    # Initialize cameras (global + wrist). Each is independently optional
    # (--no_camera_global / --no_camera_wrist) -- either, both, or neither
    # can be active. Connected/failed independently too: the wrist camera
    # failing to start doesn't take the global camera down, and vice versa.
    def _connect_camera(role, serial):
        print(f"\nInitializing {role} camera...")
        try:
            return RealSenseCamera(
                serial_number=serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                enable_depth=False,
            )
        except Exception as e:
            print(f"⚠️  {role.capitalize()} camera failed: {e}")
            if "resolve" in str(e).lower():
                print("   \"Couldn't resolve requests\" means the requested "
                      "--camera_width/--camera_height/--camera_fps combo isn't one this "
                      "camera natively supports (RealSense color streams are usually only "
                      "6/15/30/60 fps at a handful of resolutions) -- check with "
                      "system_verification/test_camera.py.")
            print(f"   Continuing without the {role} camera...")
            return None

    with_camera_global = not no_camera_global
    with_camera_wrist = not no_camera_wrist

    camera_global = _connect_camera("global", camera_serial_global) if with_camera_global else None
    with_camera_global = camera_global is not None

    camera_wrist = _connect_camera("wrist", camera_serial_wrist) if with_camera_wrist else None
    with_camera_wrist = camera_wrist is not None

    with_camera = with_camera_global or with_camera_wrist
    image_shape = (camera_height, camera_width, 3) if with_camera else None
    if not with_camera:
        print("\nSkipping camera (--no_camera_global and --no_camera_wrist, or neither connected)")

    # Initialize zarr
    zarr_root = create_zarr_dataset(
        output_dir,
        with_camera=with_camera,
        image_shape=image_shape
    )

    # Connect to arm
    _default_ip = {"ur5": "150.65.146.87", "franka": "172.16.0.2"}
    ip = robot_ip or _default_ip[arm.lower()]
    is_franka = arm.lower() == "franka"
    print(f"\nConnecting to {arm.upper()} at {ip} ...")
    if is_franka:
        robot = FrankaRobot(robot_ip=ip, frequency=frequency, use_gripper=False)
    else:
        robot = UR5eRobot(robot_ip=ip, frequency=frequency)
    # keep `ur5` as alias so the rest of the code works unchanged
    ur5 = robot

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
                max_pos_speed = 40,
                draw_hull = True)
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
    episode_buffer = DataBuffer(with_camera_global=with_camera_global, with_camera_wrist=with_camera_wrist)
    episode_count = 0
    iter_count = 0

    # Get initial pose
    tcp_pose = ur5.get_tcp_pose()
    init_pose = np.array([0.550, 0.045, 0.45, 3.14, 0.0, -0.05])
    target_pose = init_pose.copy()

    last_action = init_pose.copy() if not is_franka else np.zeros(7)

    move_2_init_pos(robot, tcp_pose, init_pose, dt=dt, velocity=0.03, duration=2.0, gain=150, is_franka=is_franka)
    tcp_pose = robot.get_tcp_pose()
    print(f"\nInitial pose: [{', '.join([f'{x:.3f}' for x in tcp_pose])}]")
    print("\nReady! Press 'C' to start recording.\n")

    # Terminal setup
    old_settings = termios.tcgetattr(sys.stdin)
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
                        move_2_init_pos(ur5, tcp_pose, init_pose, dt=dt, velocity=0.1, duration=2.0, gain=150, is_franka=is_franka)
                        print(f"✅ Robot reset to initial pose!\n")
                        target_pose = init_pose.copy()
                        last_action = init_pose.copy() if not is_franka else np.zeros(7)

                        
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
                        if with_camera_global:
                            print(f"    Camera global: {ep_data['camera_0'].shape}")
                        if with_camera_wrist:
                            print(f"    Camera wrist:  {ep_data['camera_1'].shape}")
                        print(f"    Total episodes: {episode_count}")

                        # Auto-return to start pose
                        print(f"\n🔄 Moving robot back to start pose...")
                        time.sleep(1.0)
                        try:
                            tcp_pose = ur5.get_tcp_pose()
                            move_2_init_pos(ur5, tcp_pose, init_pose, velocity=0.1, dt=dt, duration=2.0, gain=150, is_franka=is_franka)
                            print(f"✅ Robot returned to start pose!\n")
                            target_pose = init_pose.copy()
                            last_action = init_pose.copy() if not is_franka else np.zeros(7)

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

            # Arm button (left) not held this tick (idle or flowbot-only) --
            # for Franka, fully close the Cartesian-velocity background
            # thread (robot.stop(), not just zeroing the velocity) rather
            # than relying on set_ee_velocity's ~0.5s staleness watchdog.
            # Two reasons this needs to be a full stop():
            #  1) Left unhandled, the arm coasts at the last commanded
            #     speed for up to 0.5s after button release -- a real,
            #     physically significant overshoot.
            #  2) That background thread has to service libfranka's
            #     real-time FCI loop on a strict cadence. It shares the GIL
            #     with the flowbot branch below, which does synchronous IK
            #     + serial I/O + a matplotlib redraw (slow, tens of ms).
            #     Leaving the servo thread merely "parked at zero velocity"
            #     (still running, still needing GIL time every cycle) while
            #     flowbot's slow work holds the GIL is exactly what was
            #     tripping "communication_constraints_violation" reflexes
            #     and serial write timeouts every time flowbot control
            #     started. Fully stopping tears the background thread down
            #     (blocking briefly while it ramps to zero and exits), so it
            #     isn't competing for the GIL at all while flowbot runs.
            #     stop() is a cheap no-op if the session's already closed.
            # Also resync target_pose to the arm's actual measured pose:
            # target_pose is an ideal accumulator that can run ahead of the
            # real (velocity-capped) position while driving, and if left
            # stale it causes a sudden lurch the next time the button is
            # pressed. UR5e doesn't need any of this -- servo_tcp_pose has
            # no persistent background thread and already holds its last
            # target when not called.
            if is_franka and not button_status[0]:
                try:
                    robot.stop()
                except Exception:
                    pass
                target_pose = robot.get_tcp_pose()

            if button_status[1] and not button_status[0]:          # right btn: flowbot
                cmd_sm = sm.get_latest_xyz()
                xyz_fb = cmd_sm * flowbot_speed_factor
                xyz_fb[2] = -xyz_fb[2]
                xyz_fb[1] = -xyz_fb[1]
                # copied_xyz = xyz_fb.copy()
                # xyz_fb[1] = -copied_xyz[0]  # for better visualization during teleop
                # xyz_fb[0] = -copied_xyz[1]
                xyz_fb = np.where(np.abs(xyz_fb) < deadzone, 0.0, xyz_fb)
                fb.step(xyz_fb)
                fb.update_plot()

            elif button_status[0] and not button_status[1]:        # left btn: UR5e/Franka
                cmd_arm = sm.get_latest_xyz()
                cpied_cmd = cmd_arm.copy()
                cmd_arm[0] = -cmd_arm[1]  # X
                cmd_arm[1] = cpied_cmd[0] # Y
                if is_franka:
                    # Command velocity directly from stick deflection instead
                    # of routing through a target_pose/position-error
                    # round-trip (_servo_toward): that indirection exists for
                    # UR5e's absolute-position tracker, but for Franka's
                    # native velocity primitive it creates a P-loop that
                    # fights the backend's own accel-limited ramp --
                    # target_pose keeps advancing every tick faster than the
                    # accel cap lets the arm follow, the commanded velocity
                    # saturates and overshoots, error goes negative, and the
                    # next tick commands a reversal -- a limit cycle that
                    # looks like jerky/non-smooth motion even at a constant
                    # max stick push. Feeding velocity straight in removes
                    # the feedback loop entirely: held at max deflection, the
                    # commanded velocity is constant and only the ramp (not
                    # a position error) shapes the acceleration.
                    lin_vel = cmd_arm[:3] * max_pos_speed
                    # Each axis is independently bounded to max_pos_speed
                    # above, but the combined vector's norm can still reach
                    # up to max_pos_speed*sqrt(3) on a diagonal push --
                    # set_ee_velocity clips that norm to max_vel internally
                    # before sending, so replicate the same clip here first.
                    # Otherwise last_action (the raw, pre-clip vector) would
                    # disagree with what the robot actually executed on any
                    # near-max diagonal push, breaking the train/deploy
                    # action-consistency this recording scheme relies on.
                    lin_speed = float(np.linalg.norm(lin_vel))
                    if lin_speed > max_pos_speed and lin_speed > 1e-9:
                        lin_vel = lin_vel / lin_speed * max_pos_speed
                    try:
                        robot.set_ee_velocity(lin_vel, angular_velocity=np.zeros(3),
                                               max_vel=max_pos_speed, max_ang_vel=max_rot_speed)
                        # Recorded action is joint velocity (7,), not the
                        # Cartesian command above -- see get_joint_velocities()'s
                        # docstring for why. Control input stays Cartesian;
                        # only what gets *recorded* changes.
                        last_action = robot.get_joint_velocities()
                        target_pose = robot.get_tcp_pose()  # keep in sync for release/idle-hold
                        # print(f"Commanded velocity: [{', '.join([f'{x:.3f}' for x in lin_vel])}]")
                        print(f"Current TCP pos: [{np.array_str(target_pose, precision=3, suppress_small=True)}]")
                    except Exception as e:
                        print(f"\nControl error: {e}")
                        target_pose = robot.get_tcp_pose()
                else:
                    vel_linear  = cmd_arm[:3] * max_pos_speed * dt
                    vel_angular = cmd_arm[3:] * max_rot_speed * dt
                    vel_angular[:] = 0

                    target_pose[:3] += vel_linear
                    if np.any(vel_angular != 0):
                        drot = st.Rotation.from_euler('xyz', vel_angular)
                        current_rot = st.Rotation.from_rotvec(target_pose[3:])
                        target_pose[3:] = (drot * current_rot).as_rotvec()

                    try:
                        last_action = _servo_toward(ur5, is_franka, target_pose, dt, velocity=0.1,
                                                     acceleration=0.1, lookahead_time=0.1, gain=300)
                        print(f"Target pose updated: [{', '.join([f'{x:.3f}' for x in target_pose[:3]])}]")
                    except Exception as e:
                        print(f"\nControl error: {e}")
                        tcp_pose = ur5.get_tcp_pose()
                        target_pose = tcp_pose.copy()

            elif button_status[0] and button_status[1]:            # both btns: release
                print("======== RELEASING =========")
                fb.release()      # sends 'r' hardware command
                time.sleep(0.5)
                fb.reset()        # sets last_pwm = [0,0,0] and sends "0 0 0"
                fb.update_plot()
                
 
                if is_recording:
                    print(f"  Recording {release_frames} release frames ...")
                    for _ in range(release_frames):
                        # Hold robot at current target during release
                        try:
                            last_action = _servo_toward(ur5, is_franka, target_pose, dt, velocity=0.1,
                                                         acceleration=0.1, lookahead_time=0.1, gain=300)
                        except Exception:
                            pass

                        # Sleep → observe (same pattern as main loop step 3→4)
                        time.sleep(dt)

                        rel_frame = None
                        rel_frame_wrist = None
                        if with_camera_global:
                            try:
                                rel_frame, _ = camera_global.get_frames()
                            except Exception:
                                pass
                        if with_camera_wrist:
                            try:
                                rel_frame_wrist, _ = camera_wrist.get_frames()
                            except Exception:
                                pass

                        if (with_camera_global and rel_frame is None) or \
                           (with_camera_wrist and rel_frame_wrist is None):
                            continue

                        rel_tcp    = ur5.get_tcp_pose()
                        rel_joints = ur5.get_joint_angles()
                        episode_buffer.add(
                            timestamp=time.time(),
                            robot_state=rel_tcp,
                            joint_state=rel_joints,
                            action=last_action,       # robot not moving (~0 velocity for Franka)
                            pwm_signals=fb.last_pwm, # = [0,0,0] after reset
                            operation_mode=np.array([1, 1], dtype=np.uint8),
                            camera_frame=rel_frame,
                            camera_frame_wrist=rel_frame_wrist,
                        )
                    print(f"  Release recorded ({release_frames} steps, PWM={fb.last_pwm.tolist()})")

            # ── 3. Sleep BEFORE reading observations ──────────────────────────

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # ── 4. Read ALL observations after robot has settled ──────────────
            camera_frame = None
            camera_frame_wrist = None
            if with_camera_global:
                try:
                    camera_frame, _ = camera_global.get_frames()
                except Exception as e:
                    print(f"\n⚠️  Global camera error: {e}\n")
            if with_camera_wrist:
                try:
                    camera_frame_wrist, _ = camera_wrist.get_frames()
                except Exception as e:
                    print(f"\n⚠️  Wrist camera error: {e}\n")

            # ── 5. Save to buffer (camera, TCP, PWM all at same settled pose) ─
            # Skip idle frames (no button pressed) — they represent operator hesitation,
            # not intentional actions, and would teach the model to stall mid-task.
            if is_recording and np.any(op_mode):
                if (with_camera_global and camera_frame is None) or \
                   (with_camera_wrist and camera_frame_wrist is None):
                    print("\n⚠️  Warning: Missing a camera frame!\n")
                    continue

                current_tcp = ur5.get_tcp_pose()
                current_joints = ur5.get_joint_angles()
                episode_buffer.add(
                    timestamp=time.time(),
                    robot_state=current_tcp,
                    joint_state=current_joints,
                    action=last_action,
                    pwm_signals=prev_pwm,   # command from previous step (matches current image/tcp)
                    operation_mode=op_mode,
                    camera_frame=camera_frame,
                    camera_frame_wrist=camera_frame_wrist,
                )

            # ── 6. Status ─────────────────────────────────────────────────────
            iter_count += 1
            if iter_count % (frequency * 2) == 0:
                status  = "REC" if is_recording else "---"
                n_steps = len(episode_buffer) if is_recording else 0
                cam_str = (
                    ("G" if with_camera_global and camera_frame is not None else "-")
                    + ("W" if with_camera_wrist and camera_frame_wrist is not None else "-")
                )
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
        
        if camera_global:
            camera_global.stop()
        if camera_wrist:
            camera_wrist.stop()

        print(f"\n✅ Done! Collected {episode_count} episodes")
        print(f"Data: {output_dir / 'dataset.zarr'}\n")


if __name__ == '__main__':
    main()
