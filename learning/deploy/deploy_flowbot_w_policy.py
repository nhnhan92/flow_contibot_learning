#!/usr/bin/env python3
"""
Deploy trained Diffusion Policy on real UR5e/Franka + Flowbot soft manipulator

Usage:
    python deploy/deploy_real_robot.py \
        --checkpoint train/checkpoints/best_model.pt \
        --arm ur5 \
        --robot_ip 192.168.1.100 \
        --flowbot_port /dev/ttyACM0

    python deploy/deploy_real_robot.py \
        --checkpoint train/checkpoints/franka_best_model.pt \
        --arm franka \
        --flowbot_port /dev/ttyACM0

Hardware:
    - UR5e (RTDE servoL) or Franka (franky -- set_joint_velocity() or
      set_tcp_pose(), see franka_action_space below)
    - Flowbot soft pneumatic manipulator (3 valves via Arduino serial)
    - Intel RealSense camera(s): determined automatically from the
      checkpoint's camera_mode config ('global', 'wrist', or 'both'),
      matching what it was trained with.

State  (tcp_dims+5 D): robot TCP pose[:tcp_dims] + flowbot pwm (3D) + operation_mode (2D)
                        Cartesian, both arms.
Action, depends on arm and (Franka only) the checkpoint's franka_action_space
config (see demo_collect.py's _servo_toward / hardware/franka_robot.py's
get_joint_velocities() docstrings for why joint_velocity differs from its
control input):
    UR5e, or Franka franka_action_space='position' (tcp_dims+5 D):
                           target TCP[:tcp_dims] + pwm (3D) + op_mode (2D)
    Franka franka_action_space='joint_velocity' (7+5=12 D, default):
                           7D joint velocities (rad/s) + pwm (3D) + op_mode (2D)
    A checkpoint is only valid for the arm (and, for Franka, the action
    space) it was trained on -- --arm must match the collection arm.
"""

import os
import sys
import time
import datetime
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import deque

# Add parent directory to path
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(DEPLOY_DIR)
PROJECT_ROOT = os.path.dirname(LEARNING_DIR)
sys.path.insert(0, LEARNING_DIR)
from hardware.ur5e_rtde import UR5eRobot
from hardware.franka_robot import FrankaRobot
from hardware.flowbot import flowbot
from hardware.realsense_camera import RealSenseCamera
from train.eval import DiffusionPolicyInference
from hardware.image_utils import crop_and_resize

# ── Constants ─────────────────────────────────────────────────────────────────
PWM_MIN = 0   # 0 = fully deflated (release); model must be able to command this
PWM_MAX = 26

_DEFAULT_ROBOT_IP = {"ur5": "150.65.146.87", "franka": "172.16.0.2"}

# UR5e start pose -- matches init_pose in demo_collect.py, i.e. where UR5e
# demonstrations actually started from. (Was stale until 2026-09 -- an old
# value left over from a since-renamed predecessor script,
# collect_demos_with_camera.py, that didn't match demo_collect.py's current
# init_pose in either position or rotation. That mismatch fed both
# move_to_start() -- UR5e episodes were starting from the wrong physical
# pose -- and self.tcp_fixed_rotation below -- see the Franka rotation bug
# this was found alongside.)
DEFAULT_START_POSE = [0.45, 0.15, 0.5, 3.14, 0.0, -0.05]

# Franka start pose -- matches init_pose in demo_collect.py, i.e. where
# Franka demonstrations actually started from. (Currently identical to
# DEFAULT_START_POSE above -- demo_collect.py's init_pose isn't arm-specific
# -- but kept separate in case that ever changes.)
FRANKA_START_POSE = [0.45, 0.15, 0.5, 3.14, 0.0, -0.05]

# Fixed TCP rotation used when executing XYZ-only (tcp_dims=3) position
# actions from the policy -- UR5e always, Franka when franka_action_space=
# 'position'. Rotation is not predicted by the model in that case (action_dim=8)
# so we hold it constant. Unused for Franka joint_velocity mode, which has no
# "target rotation" concept at all (its action is joint velocities, not a pose).
# Arm-specific -- see RobotDeployment.__init__'s self.tcp_fixed_rotation:
# UR5e's and Franka's start orientations differ (ry, rz), so a single shared
# constant here would silently command the wrong arm's rotation.

# Control frequency (Hz)
CONTROL_FREQ =10.0
DT = 1.0 / CONTROL_FREQ
DT_FLOWBOT = 0.3     # Step time (s) when flowbot is actively actuating
FLOWBOT_FREQ = 10.0  # Flowbot command frequency — must match CONTROL_FREQ

# servo_l speed/acceleration (lower = smoother) -- UR5e only, literal m/s / m/s^2
SERVO_SPEED = 0.05     # m/s
SERVO_ACCEL = 0.05     # m/s^2

# Franka position mode (set_tcp_pose) defaults -- relative_dynamics_factor
# fractions (0-1) of Franka's own hardware limits, NOT literal m/s (see
# FrankaRobot.move_tcp_pose's docstring). Separate from UR5e's SERVO_SPEED/
# SERVO_ACCEL above -- overridable per-instance via RobotDeployment's
# franka_position_velocity/franka_position_acceleration (CLI:
# --franka_position_speed/--franka_position_accel). Lower = slower and
# gentler; also lowers jerk, since _dyn_factor ties jerk to the acceleration
# factor -- so this is also the first thing to try if set_tcp_pose keeps
# tripping the "Motion finished commanded, but the robot is still moving!"
# discontinuity reflex.
FRANKA_POSITION_VELOCITY = 0.05
FRANKA_POSITION_ACCEL = 0.05

MAX_TCP_DELTA = 0.02   # m per step -- position control (UR5e, or Franka franka_action_space='position')
MAX_TCP_ROT_DELTA = 0.05   # rad per step, same scope as MAX_TCP_DELTA -- see
                            # the "Fixed TCP rotation" note above: this bounds
                            # accidental large rotation commands (e.g. a wrong
                            # or mismatched fixed rotation) the way MAX_TCP_DELTA
                            # already bounds accidental large position commands.

# Franka set_joint_velocity cap -- runtime safety limit on the per-joint
# speed a policy-predicted action is allowed to command, independent of
# whatever speed it saw in training data. Applied elementwise (each of the
# 7 joints clipped independently), not as a Euclidean-norm cap -- a joint
# velocity limit is inherently per-DOF.
FRANKA_MAX_JOINT_VEL = 0.3   # rad/s

# Default RealSense serials, matching demo_collect.py's -- both are passed
# explicitly whenever their camera is opened (regardless of camera_mode) so
# a single-camera deploy still binds the intended physical device even if
# both cameras happen to be connected, and 'both' mode's two pipelines never
# race to grab the same one (see demo_collect.py's camera connection comments).
_DEFAULT_CAMERA_SERIAL_GLOBAL = '827112072398'
_DEFAULT_CAMERA_SERIAL_WRIST  = '841512070635'

SERVO_LOOKAHEAD = 0.1   # s
SERVO_GAIN = 300


class _ReleaseDetected(Exception):
    """Internal sentinel: raised when op_mode [1,1] is predicted to exit episode loop."""


class DeploymentLogger:
    """
    Logs model predictions and robot states during a deployment episode.

    Saved per episode (.npz):
        timestamps         (T,)              wall-clock time of each executed step
        tcp_poses          (T, 6)            actual TCP pose read from robot at each step
        pwm_actual         (T, 3)            actual PWM values read from Flowbot at each step
        executed_actions   (T, 8)            full 8D action commanded at each step (denormalised)
        pwm_commanded      (T, 3)            integer PWM sent after clamping
        predicted_horizons (N_plans, P, 8)   full pred_horizon action sequence for each plan
        plan_times_ms      (N_plans,)        DDIM inference latency per plan (milliseconds)
        plan_step_indices  (N_plans,)        step index at which each plan was triggered

    Load later with:
        data = np.load('episode_000_20260228_120000.npz')
        tcp  = data['tcp_poses']          # (T, 6)
        pred = data['predicted_horizons'] # (N_plans, pred_horizon, 9)
    """

    def __init__(self, log_dir: str, checkpoint_path: str, tcp_dims: int = 3):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = str(checkpoint_path)
        self.tcp_dims = tcp_dims
        self._reset()

    def _reset(self):
        self._timestamps        = []
        self._tcp_poses         = []
        self._pwm_actual        = []
        self._executed_actions  = []
        self._pwm_commanded     = []
        self._predicted_horizons = []
        self._plan_times_ms     = []
        self._plan_step_indices = []

    def log_plan(self, step_idx: int, predicted_actions: np.ndarray, plan_time_s: float):
        """Call once per DDIM inference (before executing the action horizon)."""
        self._predicted_horizons.append(predicted_actions.copy())
        self._plan_times_ms.append(plan_time_s * 1000.0)
        self._plan_step_indices.append(step_idx)

    def log_step(self, state_raw: np.ndarray, action: np.ndarray, pwm_commanded: np.ndarray):
        """Call once per executed step (after _update_obs_buffer)."""
        d = self.tcp_dims
        self._timestamps.append(time.time())
        self._tcp_poses.append(state_raw[:d].copy())      # tcp_dims components
        self._pwm_actual.append(state_raw[d:d+3].copy())
        self._executed_actions.append(action.copy())
        self._pwm_commanded.append(pwm_commanded.copy())

    def save(self, episode_idx: int, total_steps: int, duration_s: float) -> Path:
        """Save collected data for one episode and reset buffers."""
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = self.log_dir / f"episode_{episode_idx:03d}_{ts}.npz"

        np.savez_compressed(
            out_path,
            timestamps         = np.array(self._timestamps),
            tcp_poses          = np.array(self._tcp_poses),
            pwm_actual         = np.array(self._pwm_actual),
            executed_actions   = np.array(self._executed_actions),
            pwm_commanded      = np.array(self._pwm_commanded),
            predicted_horizons = np.array(self._predicted_horizons),
            plan_times_ms      = np.array(self._plan_times_ms),
            plan_step_indices  = np.array(self._plan_step_indices),
            total_steps        = total_steps,
            duration_s         = duration_s,
            checkpoint_path    = self.checkpoint_path,
        )
        print(f"  Log saved: {out_path}")
        self._reset()
        return out_path


class RobotDeployment:
    """
    Main deployment class for UR5e + Flowbot with Diffusion Policy.

    Observation buffer keeps the last `obs_horizon` frames so the policy
    always sees a temporal window of states and images.
    """

    def __init__(
        self,
        checkpoint_path: str,
        robot_ip: str,
        arm: str = 'ur5',
        flowbot_port: str = '/dev/ttyACM0',
        flowbot_baud: int = 115200,
        image_size: tuple = (216, 288),
        device: str = 'cuda',
        verbose: bool = True,
        camera_height: int = 480,
        camera_width: int = 640,
        camera_serial_global: str = _DEFAULT_CAMERA_SERIAL_GLOBAL,
        camera_serial_wrist: str = _DEFAULT_CAMERA_SERIAL_WRIST,
        position_command_stride: int = 1,
        franka_position_velocity: float = FRANKA_POSITION_VELOCITY,
        franka_position_acceleration: float = FRANKA_POSITION_ACCEL,
    ):
        self.verbose = verbose
        self.arm = arm.lower()
        self.is_franka = self.arm == "franka"
        self.current_pwm = np.array([0, 0, 0], dtype=int)
        self.prev_pwm    = np.zeros(3, dtype=np.float32)   # command from previous step
        # Fixed rotation held when executing XYZ-only (tcp_dims=3) position
        # actions -- arm-specific, since UR5e's and Franka's start
        # orientations differ (ry, rz). See FRANKA_START_POSE/DEFAULT_START_POSE.
        self.tcp_fixed_rotation = FRANKA_START_POSE[3:] if self.is_franka else DEFAULT_START_POSE[3:]
        # Franka position mode only -- relative_dynamics_factor fractions
        # (0-1), see FRANKA_POSITION_VELOCITY/FRANKA_POSITION_ACCEL above.
        self.franka_position_velocity = franka_position_velocity
        self.franka_position_acceleration = franka_position_acceleration

        # ── Load policy ───────────────────────────────────────────────────────
        print(f"\n[1/4] Loading policy from: {checkpoint_path}")
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy = DiffusionPolicyInference(checkpoint_path, device=str(device_obj))
        self.config = self.policy.config
        self.obs_horizon = self.config['obs_horizon']
        self.action_horizon = self.config['action_horizon']
        self.tcp_dims = self.config.get('tcp_dims', 3)   # 3=xyz only, 6=xyz+rotation -- state only
        self.franka_action_space = self.config.get('franka_action_space', 'joint_velocity')
        if self.is_franka and self.franka_action_space not in ('joint_velocity', 'position'):
            raise ValueError(
                f"Checkpoint config has franka_action_space={self.franka_action_space!r} "
                "(expected 'joint_velocity' or 'position')"
            )
        # True iff this checkpoint's action is TCP position -- always for
        # UR5e, only for Franka when its config selected 'position'. Gates
        # which half of _execute_action runs (see dataset.py's identical flag).
        self.uses_position_action = (not self.is_franka) or self.franka_action_space == 'position'
        self.is_franka_joint_vel = self.is_franka and not self.uses_position_action
        # Franka joint_velocity: always 7D (tcp_dims doesn't apply to it, only
        # to state). Otherwise (UR5e, or Franka position mode): TCP[:tcp_dims].
        self.action_dim_arm = self.tcp_dims if self.uses_position_action else 7
        # Execute only every Nth predicted position waypoint instead of every
        # one -- e.g. stride=2 on [A,B,C,D,E] executes only [B,D,E]. Gives
        # each issued set_tcp_pose()/servo_tcp_pose() call N*DT instead of DT
        # to actually settle before the next one supersedes it, which is what
        # causes Franka position mode's jiggle (CartesianMotion plans to
        # arrive-and-stop, then gets interrupted mid-brake every tick).
        # No-op (every action still executed) for joint_velocity mode, which
        # has no arrive-and-stop semantics to begin with.
        self.position_command_stride = position_command_stride if self.uses_position_action else 1
        if self.position_command_stride < 1:
            raise ValueError(f"position_command_stride must be >= 1, got {position_command_stride}")
        self.camera_mode = self.config.get('camera_mode', 'global')
        if self.camera_mode not in ('global', 'wrist', 'both'):
            raise ValueError(
                f"Checkpoint config has camera_mode={self.camera_mode!r} "
                "(expected 'global', 'wrist', or 'both')"
            )
        self.num_cameras = self.policy.num_cameras   # 1 (single) or 2 (+ wrist), from checkpoint config
        print(f"      obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}")
        print(f"      tcp_dims={self.tcp_dims}  ({'xyz only' if self.tcp_dims == 3 else 'xyz+rotation'})")
        print(f"      action_dim_arm={self.action_dim_arm} ({'TCP pose' if self.uses_position_action else 'joint velocity'})"
              + (f"  franka_action_space={self.franka_action_space}" if self.is_franka else ""))
        if self.uses_position_action and self.position_command_stride > 1:
            print(f"      position_command_stride={self.position_command_stride} "
                  f"(executing 1 of every {self.position_command_stride} predicted waypoints)")
        if self.is_franka and self.uses_position_action:
            print(f"      franka_position_velocity={self.franka_position_velocity}, "
                  f"franka_position_acceleration={self.franka_position_acceleration} "
                  f"(relative_dynamics_factor fractions, 0-1)")
        print(f"      camera_mode={self.camera_mode}  (num_cameras={self.num_cameras})")
        print(f"      device={device_obj}")
        if image_size is None:
            self.image_size = tuple(self.policy.config['image_size'])
        else:
            self.image_size = image_size
        # Crop anchor must match training exactly (see image_utils.crop_and_resize).
        # Wrist camera can have its own image_size/crop settings -- see dataset.py's
        # identical global/wrist split; falls back to the global values when unset.
        self.crop_scale = self.config.get('crop_scale', 1.5)
        self.crop_x = self.config.get('crop_x', 0.5)
        self.crop_y = self.config.get('crop_y', 0.5)
        wrist_image_size = self.config.get('wrist_image_size', None)
        self.wrist_image_size = tuple(wrist_image_size) if wrist_image_size is not None else self.image_size
        self.wrist_crop_scale = self.config.get('wrist_crop_scale', self.crop_scale)
        self.wrist_crop_x = self.config.get('wrist_crop_x', self.crop_x)
        self.wrist_crop_y = self.config.get('wrist_crop_y', self.crop_y)
        # Settings for the primary/single-encoder slot (self.cam): global's,
        # unless this is wrist-only (self.cam IS the wrist camera in that case).
        if self.camera_mode == 'wrist':
            self._primary_image_size = self.wrist_image_size
            self._primary_crop_scale = self.wrist_crop_scale
            self._primary_crop_x = self.wrist_crop_x
            self._primary_crop_y = self.wrist_crop_y
        else:
            self._primary_image_size = self.image_size
            self._primary_crop_scale = self.crop_scale
            self._primary_crop_x = self.crop_x
            self._primary_crop_y = self.crop_y
        # ── Camera(s) ─────────────────────────────────────────────────────────
        # self.cam is always the primary/single-encoder feed (matches
        # dataset.py's obs_image routing): the global camera unless
        # camera_mode=='wrist', in which case the wrist camera fills that
        # same slot. self.cam_wrist is only opened for camera_mode=='both'
        # (second, independent encoder -- matches obs_image_wrist).
        print(f"\n[2/4] Opening RealSense camera(s) ({self.camera_mode}) ...")
        self.cam = None
        self.cam_wrist = None
        if self.camera_mode in ('global', 'both'):
            self.cam = RealSenseCamera(
                serial_number=camera_serial_global,
                width=camera_width,
                height=camera_height,
                enable_depth=False,
            )
        if self.camera_mode in ('wrist', 'both'):
            wrist_cam = RealSenseCamera(
                serial_number=camera_serial_wrist,
                width=camera_width,
                height=camera_height,
                enable_depth=False,
            )
            if self.camera_mode == 'wrist':
                self.cam = wrist_cam
            else:
                self.cam_wrist = wrist_cam
        print("      Camera(s) OK")

        # ── Robot arm ─────────────────────────────────────────────────────────
        print(f"\n[3/4] Connecting to {self.arm.upper()} at {robot_ip} ...")
        if self.is_franka:
            self.robot = FrankaRobot(robot_ip=robot_ip, frequency=CONTROL_FREQ, use_gripper=False)
        else:
            self.robot = UR5eRobot(robot_ip=robot_ip, frequency=CONTROL_FREQ)
        print(f"      {self.arm.upper()} connected")

        # ── Flowbot ───────────────────────────────────────────────────────────
        print(f"\n[4/4] Connecting to Flowbot on {flowbot_port} ...")
        self.fb = flowbot(serial_port=flowbot_port,
                          baud=flowbot_baud,
                          pwm_min=PWM_MIN,
                          pwm_max=PWM_MAX,
                          enable_plot=False,
                          frequency=FLOWBOT_FREQ,
                          max_pos_speed=30)
        self.fb.start()
        time.sleep(2.0)  # Arduino reset delay
        print("      Flowbot connected")

        # ── Observation buffers ───────────────────────────────────────────────
        self.state_buffer = deque(maxlen=self.obs_horizon)
        self.image_buffer = deque(maxlen=self.obs_horizon)
        self.image_buffer_wrist = deque(maxlen=self.obs_horizon) if self.num_cameras == 2 else None
        self.current_op_mode = np.zeros(2, dtype=np.float32)

        print("\n✅ All systems ready!\n")

    # ── Low-level observation ─────────────────────────────────────────────────

    def _crop_resize_primary(self, camera_frame: np.ndarray) -> np.ndarray:
        """Crop + resize the primary-slot (self.cam) frame — matches dataset.py's primary settings."""
        return crop_and_resize(
            camera_frame, self._primary_image_size,
            crop_scale=self._primary_crop_scale, crop_x=self._primary_crop_x, crop_y=self._primary_crop_y,
        )

    def _crop_resize_wrist(self, camera_frame: np.ndarray) -> np.ndarray:
        """Crop + resize the wrist-slot (self.cam_wrist, 'both' mode only) frame — matches dataset.py's wrist settings."""
        return crop_and_resize(
            camera_frame, self.wrist_image_size,
            crop_scale=self.wrist_crop_scale, crop_x=self.wrist_crop_x, crop_y=self.wrist_crop_y,
        )

    def _get_raw_observation(self):
        """
        Read current robot state and camera image(s).

        Returns:
            state_raw       : np.ndarray (tcp_dims+5,) — [tcp[:tcp_dims], pwm1,pwm2,pwm3, ur5_active, flowbot_active]
            image_raw       : np.ndarray (H,W,3) uint8 — cropped primary-camera frame
                               (global, unless camera_mode=='wrist')
            image_raw_wrist : np.ndarray (H,W,3) uint8, or None unless camera_mode=='both' — cropped wrist camera frame
        """
        # Robot TCP pose — slice to tcp_dims (3=xyz only, 6=xyz+rotation)
        tcp_pose = self.robot.get_tcp_pose()

        # PWM from previous step — matches the physical state visible in the current image
        pwm = self.prev_pwm.copy()                                                          # (3,)

        # Operation mode from last executed action (2D)
        state_raw = np.concatenate([tcp_pose[:self.tcp_dims], pwm, self.current_op_mode])  # (tcp_dims+5,)

        # Camera image(s)
        camera_frame, _ = self.cam.get_frames()
        if camera_frame is None:
            cam_role = 'Wrist' if self.camera_mode == 'wrist' else 'Global'
            raise RuntimeError(f"{cam_role} camera read failed")
        image_raw = self._crop_resize_primary(camera_frame)

        image_raw_wrist = None
        if self.camera_mode == 'both':
            camera_frame_wrist, _ = self.cam_wrist.get_frames()
            if camera_frame_wrist is None:
                raise RuntimeError("Wrist camera read failed")
            image_raw_wrist = self._crop_resize_wrist(camera_frame_wrist)

        return state_raw, image_raw, image_raw_wrist

    # ── Preprocessing (matching dataset.py) ──────────────────────────────────

    def _preprocess_state(self, state_raw: np.ndarray) -> np.ndarray:
        """Min-Max normalise state to [-1, 1]."""
        state_min   = self.policy.checkpoint['state_min']
        state_range = self.policy.checkpoint['state_range']
        return (2.0 * (state_raw - state_min) / state_range - 1.0).astype(np.float32)

    def _preprocess_image(self, image_raw: np.ndarray) -> np.ndarray:
        """uint8 (H,W,3) → float32 (3,H,W) in [-1,1]."""
        img = (image_raw.astype(np.float32) / 127.5) - 1.0
        return img.transpose(2, 0, 1)   # (3,H,W)

    # ── Observation buffer management ─────────────────────────────────────────

    def _fill_obs_buffer(self):
        """Fill observation buffers with obs_horizon copies of current obs."""
        state_raw, image_raw, image_raw_wrist = self._get_raw_observation()
        state_norm = self._preprocess_state(state_raw)
        image_norm = self._preprocess_image(image_raw)
        image_norm_wrist = self._preprocess_image(image_raw_wrist) if image_raw_wrist is not None else None
        for _ in range(self.obs_horizon):
            self.state_buffer.append(state_norm.copy())
            self.image_buffer.append(image_norm.copy())
            if self.image_buffer_wrist is not None:
                self.image_buffer_wrist.append(image_norm_wrist.copy())

    def _update_obs_buffer(self):
        """Append one new observation to the rolling buffer. Returns state_raw for logging."""
        state_raw, image_raw, image_raw_wrist = self._get_raw_observation()
        state_norm = self._preprocess_state(state_raw)
        image_norm = self._preprocess_image(image_raw)
        self.state_buffer.append(state_norm)
        self.image_buffer.append(image_norm)
        if self.image_buffer_wrist is not None:
            self.image_buffer_wrist.append(self._preprocess_image(image_raw_wrist))
        return state_raw

    def _get_obs_tensors(self):
        """
        Stack buffer contents into tensors for the policy.

        Returns:
            obs_state       : torch.Tensor (1, obs_horizon, state_dim)
            obs_image       : torch.Tensor (1, obs_horizon, 3, H, W) -- global
            obs_image_wrist : torch.Tensor (1, obs_horizon, 3, H, W), or None if num_cameras==1
        """
        obs_state = torch.from_numpy(
            np.stack(list(self.state_buffer), axis=0)   # (obs_horizon, state_dim)
        ).unsqueeze(0)                                   # (1, obs_horizon, state_dim)

        obs_image = torch.from_numpy(
            np.stack(list(self.image_buffer), axis=0)   # (obs_horizon, 3, H, W)
        ).unsqueeze(0)                                   # (1, obs_horizon, 3, H, W)

        obs_image_wrist = None
        if self.image_buffer_wrist is not None:
            obs_image_wrist = torch.from_numpy(
                np.stack(list(self.image_buffer_wrist), axis=0)
            ).unsqueeze(0)

        return obs_state, obs_image, obs_image_wrist

    # ── Policy inference ──────────────────────────────────────────────────────

    def _predict_actions(self):
        """
        Run one DDIM inference step.

        Returns:
            actions : np.ndarray (pred_horizon, action_dim_arm+5) — denormalised actions.
                      Position control (UR5e; Franka franka_action_space='position'):
                          [:tcp_dims] = TCP target, [tcp_dims:tcp_dims+3] = pwm, rest = op_mode
                      Franka joint_velocity: [:7] = joint velocities (rad/s), [7:10] = pwm, rest = op_mode
        """
        obs_state, obs_image, obs_image_wrist = self._get_obs_tensors()
        actions_norm = self.policy.predict(
            obs_state.squeeze(0),                                             # (obs_horizon, state_dim)
            obs_image.squeeze(0),                                             # (obs_horizon, 3, H, W)
            obs_image_wrist.squeeze(0) if obs_image_wrist is not None else None,
        ).numpy()                   # (pred_horizon, action_dim)

        # Denormalise: x = (x_norm + 1) * 0.5 * range + min
        action_min   = self.policy.checkpoint['action_min']
        action_range = self.policy.checkpoint['action_range']
        actions = (actions_norm + 1.0) * 0.5 * action_range + action_min
        return actions              # (pred_horizon, action_dim)

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_action(self, action: np.ndarray, execute_arm: bool = True, steps_covered: int = 1):
        """
        Send one action step to the robot and flowbot, gated by predicted operation mode.

        Args:
            action : np.ndarray (action_dim_arm+5,) — [action_dim_arm-wide arm command,
                     pwm1,pwm2,pwm3, ur5_active, flowbot_active]
                     Position control (UR5e always; Franka when
                             franka_action_space=='position'): action[:tcp_dims]
                             is an absolute TCP target (fixed rotation
                             self.tcp_fixed_rotation is appended when tcp_dims=3
                             to form a 6D target -- arm-specific, see __init__).
                     Franka joint_velocity: action[:7] is joint velocities
                             (rad/s) -- see hardware/franka_robot.py's
                             get_joint_velocities() docstring for why this is
                             joint space even though live teleoperation
                             commands Cartesian velocity.
            execute_arm : If False, decode and return PWM/op_mode as usual
                     (flowbot is unaffected by position_command_stride) but
                     don't touch the arm at all this tick -- used by
                     position_command_stride > 1 to skip intermediate
                     waypoints. The previously-issued command keeps running
                     on its own; see run_episode's execution loop.
            steps_covered : How many ticks (including this one) since the arm
                     was last actually commanded. Only meaningful when
                     execute_arm=True and position control (not Franka
                     joint_velocity): scales the MAX_TCP_DELTA safety clamp,
                     since a waypoint N ticks ahead in the model's own
                     predicted trajectory is expectedly ~N times as far from
                     the current position as a single tick's worth would be
                     -- that's real predicted motion, not something to clip
                     back down to a 1-tick-sized step.

        Returns:
            pwm_int      : np.ndarray (3,) int — clamped PWM actually sent
            op_mode_pred : np.ndarray (2,) int — [ur5_active, flowbot_active]
        """
        d = self.action_dim_arm
        franka_joint_vel = self.is_franka_joint_vel
        if franka_joint_vel:
            dq = np.array(action[:7], dtype=np.float64)
        elif d == 6:
            tcp_target = action[:6].tolist()
        else:  # d == 3: append fixed rotation so the robot holds its orientation
            tcp_target = action[:3].tolist() + self.tcp_fixed_rotation
        pwm_raw    = action[d:d+3]

        # Decode predicted operation mode (denorm ~[0,1] → binary)
        op_mode_pred = np.clip(np.round(action[d+3:d+5]), 0, 1).astype(int)

        # PWM offset, flowbot-active steps only.
        # if op_mode_pred[1] == 1:
        #     pwm_raw = pwm_raw + np.array([3, 0, 1])

        pwm_int    = np.clip(np.round(pwm_raw), PWM_MIN, PWM_MAX).astype(int)

        # Drop protection: if any channel decreases vs last sent PWM, hold previous value
        if np.any(pwm_int < self.current_pwm):
            pwm_int = self.current_pwm.copy()

        # Gate arm command: only move when ur5_active (field name kept for both arms),
        # and only if this tick actually owns the arm (position_command_stride
        # may have this tick's predicted waypoint skipped -- the previously
        # issued command keeps running on its own in that case).
        if execute_arm and op_mode_pred[0] == 1:
            if franka_joint_vel:
                # Safety clamp: cap each joint's speed to FRANKA_MAX_JOINT_VEL
                # regardless of what speed the policy saw in training data.
                # Elementwise, not a norm clamp -- a joint velocity limit is
                # inherently per-DOF.
                over = np.abs(dq) > FRANKA_MAX_JOINT_VEL
                if np.any(over) and self.verbose:
                    print(f"  ⚠️  Predicted joint speed(s) {np.round(dq[over], 3).tolist()} "
                          f"clamped to ±{FRANKA_MAX_JOINT_VEL:.2f} rad/s")
                dq = np.clip(dq, -FRANKA_MAX_JOINT_VEL, FRANKA_MAX_JOINT_VEL)
                # No Cartesian planning or Jacobian inversion here -- joint
                # velocities execute directly, so this can't trip a
                # Cartesian-singularity discontinuity reflex the way the
                # old set_ee_velocity path could.
                self.robot.set_joint_velocity(dq, max_vel=FRANKA_MAX_JOINT_VEL)
            else:
                # Position control -- UR5e always, Franka when
                # franka_action_space=='position'. Safety clamp: limit XYZ
                # displacement to steps_covered * MAX_TCP_DELTA (see
                # steps_covered's docstring above for why it's scaled).
                current_tcp = self.robot.get_tcp_pose()
                tcp_arr = np.array(tcp_target, dtype=np.float64)
                delta_xyz = tcp_arr[:3] - current_tcp[:3]
                dist = np.linalg.norm(delta_xyz)
                max_delta = steps_covered * MAX_TCP_DELTA
                if dist > max_delta:
                    tcp_arr[:3] = current_tcp[:3] + delta_xyz * (max_delta / dist)
                    if self.verbose:
                        print(f"  ⚠️  TCP delta {dist*1000:.1f}mm clamped to {max_delta*1000:.0f}mm")
                # Safety clamp: same idea, for rotation -- bounds an
                # accidental large rotation command (e.g. self.tcp_fixed_rotation
                # not actually matching the arm's current orientation) the way
                # the XYZ clamp above bounds an accidental large position jump.
                delta_rot = tcp_arr[3:] - current_tcp[3:]
                rot_dist = np.linalg.norm(delta_rot)
                max_rot_delta = steps_covered * MAX_TCP_ROT_DELTA
                if rot_dist > max_rot_delta:
                    tcp_arr[3:] = current_tcp[3:] + delta_rot * (max_rot_delta / rot_dist)
                    if self.verbose:
                        print(f"  ⚠️  TCP rotation delta {np.degrees(rot_dist):.1f}° "
                              f"clamped to {np.degrees(max_rot_delta):.1f}°")
                tcp_target = tcp_arr.tolist()
                if self.is_franka:
                    # set_tcp_pose (franky CartesianMotion) -- see
                    # hardware/franka_robot.py's docstring. velocity/acceleration
                    # are relative_dynamics_factor fractions (0-1), not m/s --
                    # self.franka_position_velocity/acceleration, independently
                    # tunable from UR5e's SERVO_SPEED/SERVO_ACCEL (see
                    # FRANKA_POSITION_VELOCITY/FRANKA_POSITION_ACCEL above).
                    self.robot.set_tcp_pose(tcp_target, velocity=self.franka_position_velocity,
                                           acceleration=self.franka_position_acceleration)
                else:
                    self.robot.servo_tcp_pose(target_pose=tcp_target, velocity=SERVO_SPEED,
                                            acceleration=SERVO_ACCEL, dt=DT,
                                            lookahead_time=SERVO_LOOKAHEAD, gain=SERVO_GAIN)
        elif execute_arm and franka_joint_vel:
            # ur5_active == 0 this step -- franky's set_joint_velocity has no
            # staleness watchdog: a JointVelocityMotion keeps running at its
            # last commanded velocity indefinitely until explicitly
            # superseded or stopped. Simply not calling it here would leave
            # the arm coasting at whatever speed the last active step
            # commanded, not just briefly but until something else happens
            # to stop it -- explicitly stop every tick the arm shouldn't be
            # moving. stop_joint_velocity(), not stop() -- see
            # hardware/franka_robot.py's stop_joint_velocity() docstring for
            # why the Cartesian-velocity stop doesn't reliably cover this.
            self.robot.stop_joint_velocity()
        # Franka position mode, inactive: no explicit stop needed here, same
        # as UR5e -- set_tcp_pose() is only called on active ticks (above),
        # not a continuously-running session the way set_joint_velocity() is.
        # execute_arm=False: arm untouched entirely this tick, by design.

        # Gate flowbot PWM: only send when flowbot_active
        if op_mode_pred[1] == 1 and np.any(pwm_int >= PWM_MIN):
            # pwm_int[0] = 0  # this is just a trick
            self.fb.serial_sending(pwm_int, wait_ack=True, ack_timeout=DT_FLOWBOT)
            self.current_pwm = pwm_int.copy()

        # Update tracked op_mode for next observation
        self.current_op_mode = op_mode_pred.astype(np.float32)

        if self.verbose:
            mode_str = ['idle', 'FB', 'UR5', 'release'][op_mode_pred[0] * 2 + op_mode_pred[1]]
            skip_str = "" if execute_arm else " (skipped -- stride, prior command still running)"
            if franka_joint_vel:
                dq_str = ', '.join(f'{v:.3f}' for v in dq)
                print(f"  [{mode_str}] Q_VEL: [{dq_str}] rad/s  PWM: {pwm_int.tolist()}{skip_str}")
            else:
                tcp = np.array(tcp_target, dtype=np.float32)
                print(
                    f"  [{mode_str}] TCP: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]  "
                    f"PWM: {pwm_int.tolist()}{skip_str}"
                )

        return pwm_int, op_mode_pred

    # ── Start position ────────────────────────────────────────────────────────

    def move_to_start(self, speed: float = None, accel: float = None):
        """
        Move the arm to its default start pose using a point-to-point move,
        then reset flowbot.

        speed, accel: None (default) resolves to self.franka_position_velocity/
        acceleration for Franka -- so --franka_position_speed/--franka_position_accel
        actually govern ALL Franka position-control motion, not just the
        active per-tick episode loop (this move is also franky CartesianMotion
        via move_tcp_pose, same relative_dynamics_factor semantics as
        set_tcp_pose) -- or 0.05/0.05 for UR5e, unchanged from before and
        independent of SERVO_SPEED/SERVO_ACCEL (those are for UR5e's active
        per-tick servo_tcp_pose loop only).
        """
        if speed is None:
            speed = self.franka_position_velocity if self.is_franka else 0.05
        if accel is None:
            accel = self.franka_position_acceleration if self.is_franka else 0.05
        print("\nMoving to start position ...")
        start_pose = FRANKA_START_POSE if self.is_franka else DEFAULT_START_POSE
        self.robot.move_tcp_pose(start_pose, velocity=speed, acceleration=accel)
        print(f"  TCP at: {start_pose}")

        print("Resetting Flowbot ...")
        self.fb.reset()
        time.sleep(0.5)
        print("  Flowbot reset OK")

    # ── Main episode loop ─────────────────────────────────────────────────────

    def run_episode(self, max_steps: int = 400, move_to_start: bool = True,
                    logger: 'DeploymentLogger | None' = None, episode_idx: int = 0):
        """
        Run one deployment episode with receding-horizon control.

        The policy produces `pred_horizon` actions; we execute `action_horizon`
        of them before re-planning — identical to training's receding horizon.

        Args:
            max_steps    : Hard step limit (safety stop)
            move_to_start: If True, move robot to start before running
            logger       : Optional DeploymentLogger; if provided, all predictions
                           and states are recorded and saved at episode end
            episode_idx  : Episode number (used for the log filename)
        """
        if move_to_start:
            self.move_to_start()

        # Reset internal state so the new episode starts clean (op_mode=[0,0], PWM=0).
        # This prevents carryover from the previous episode biasing the first observation.
        self.current_op_mode = np.zeros(2, dtype=np.float32)
        self.current_pwm     = np.array([0, 0, 0], dtype=int)
        self.prev_pwm        = np.zeros(3, dtype=np.float32)

        print("\n" + "="*30)
        print("Starting episode ...")
        print("="*30)

        # Fill the observation buffer with obs_horizon initial frames
        print("Filling observation buffer ...")
        self._fill_obs_buffer()

        total_steps = 0
        episode_start = time.time()

        try:
            while total_steps < max_steps:
                # ── Plan: run DDIM inference ──────────────────────────────────
                # Stop any live joint-velocity session before this heavy,
                # synchronous compute burst -- otherwise a JointVelocityMotion
                # left streaming from the previous action_horizon's last step
                # keeps running while DDIM inference (ResNet+UNet, CPU-bound
                # if --device cpu) competes for CPU scheduling time, which can
                # starve franky/libfranka's realtime communication thread long
                # enough to miss its deadline -- the same
                # communication_constraints_violation reflex already seen (and
                # fixed the same way) for flowbot's synchronous work in
                # demo_collect.py. UR5e's servo_tcp_pose has no persistent
                # session to stop here, so this only applies to Franka
                # joint_velocity mode. Franka position mode's set_tcp_pose()
                # is also a reactive/async call every tick (same pattern as
                # set_joint_velocity()), so the same GIL-starvation concern
                # could plausibly apply there too -- but there's no verified
                # equivalent stop call for an active CartesianMotion (as
                # opposed to CartesianVelocityMotion, which stop_joint_velocity()'s
                # docstring says needs its own distinct stop type) to reach for
                # here without confirming on hardware first, so this is left
                # as a known gap for franka_action_space='position' rather
                # than guessing at an unverified API call.
                if self.is_franka_joint_vel:
                    self.robot.stop_joint_velocity()
                t_plan_start = time.time()
                actions = self._predict_actions()   # (pred_horizon, 9)
                t_plan = time.time() - t_plan_start

                if self.verbose:
                    print(f"\nStep {total_steps} | Plan time: {t_plan*1000:.1f} ms")

                if logger is not None:
                    logger.log_plan(total_steps, actions, t_plan)

                # ── Execute: action_horizon steps from the plan ───────────────
                for step_i in range(self.action_horizon):
                    if total_steps >= max_steps:
                        break

                    t_step_start = time.time()

                    # Snapshot PWM before this step's command — image after sleep
                    # will reflect this value, not the command about to be issued.
                    self.prev_pwm = self.current_pwm.astype(np.float32)

                    action = actions[step_i]            # (8,)
                    # position_command_stride: execute only every Nth
                    # predicted waypoint (the last of each group of N), e.g.
                    # stride=2 on 5 steps [A,B,C,D,E] executes only [B,D,E] --
                    # always execute the very last step of the horizon too,
                    # even if it falls mid-group (a partial, smaller-than-N
                    # trailing group still needs to be acted on). No-op when
                    # stride=1 (every step executes, steps_covered always 1).
                    stride = self.position_command_stride
                    execute_arm = ((step_i + 1) % stride == 0) or (step_i == self.action_horizon - 1)
                    steps_covered = (step_i % stride) + 1
                    try:
                        pwm_int, op_mode_pred = self._execute_action(
                            action, execute_arm=execute_arm, steps_covered=steps_covered
                        )
                    except Exception as e:
                        # A transient motion fault (e.g. Franka's
                        # cartesian_motion_generator_*_discontinuity reflex --
                        # "Motion finished commanded, but the robot is still
                        # moving!", see _dyn_factor's docstring in
                        # hardware/franka_robot.py) is already recovered on
                        # the robot side inside set_tcp_pose/set_joint_velocity
                        # (recover_from_errors()) before this re-raises --
                        # without this catch that recovery was wasted, since
                        # the exception would otherwise crash the whole
                        # episode/deployment even though the arm is fine.
                        # Treat this tick as idle (safest default) and continue.
                        print(f"\n⚠️  Action execution error (arm recovered, continuing): {e}")
                        pwm_int = self.current_pwm.copy()
                        op_mode_pred = np.zeros(2, dtype=int)
                        self.current_op_mode = op_mode_pred.astype(np.float32)

                    # Release phase detected: hold 1 s then end episode immediately
                    if op_mode_pred[0] == 1 and op_mode_pred[1] == 1:
                        print("\n🔓 Release phase")
                        self.fb.release()   # sends 'r' to Arduino (triggers suction release)
                        total_steps += 1
                        raise _ReleaseDetected

                    # Use longer step time when flowbot is actively actuating
                    # so the soft actuator has enough time to inflate/deflate
                    step_dt = DT_FLOWBOT if op_mode_pred[1] == 1 else DT
                    elapsed = time.time() - t_step_start
                    sleep_time = step_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    # Update obs buffer AFTER robot has moved toward target
                    state_raw = self._update_obs_buffer()
                    if self.verbose:
                        _, image_raw, image_wrist = self._get_raw_observation()   # second read just for display
                        cv2.imshow("Live", cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if logger is not None:
                        logger.log_step(state_raw, action, pwm_int)

                    total_steps += 1

        except _ReleaseDetected:
            print("✅ Episode ended by release phase")
            # time.sleep(1)
            # print("Resetting Flowbot ...")
            # self.fb.reset()
            self.move_to_start()
        except KeyboardInterrupt:
            print("\n⚠️  Episode interrupted by user")

        elapsed_total = time.time() - episode_start
        print(f"\n✅ Episode finished: {total_steps} steps in {elapsed_total:.1f}s")

        # Stop arm servoing/velocity control and let it settle before any subsequent move.
        # Franka joint_velocity: needs stop_joint_velocity(), not stop() (Cartesian) --
        # see hardware/franka_robot.py's stop_joint_velocity() docstring.
        print("Resetting Flowbot ...")
        self.fb.reset()
        if self.is_franka:
            self.move_to_start()
            if self.is_franka_joint_vel:
                self.robot.stop_joint_velocity()
        else:
            self.robot.stop()
        time.sleep(0.5)

        # Save deployment log
        if logger is not None:
            logger.save(episode_idx, total_steps, elapsed_total)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self):
        """Safely disconnect all hardware."""
        print("\nShutting down ...")
        try:
            self.robot.disconnect()
        except Exception as e:
            print(f"  {self.arm.upper()} shutdown error: {e}")
        try:
            self.fb.reset()
        except Exception as e:
            print(f"  Flowbot reset error: {e}")
        try:
            self.cam.stop()
        except Exception:
            pass
        if self.cam_wrist is not None:
            try:
                self.cam_wrist.stop()
            except Exception:
                pass
        print("✅ Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='Deploy Diffusion Policy on UR5e/Franka + Flowbot')
    parser.add_argument('--checkpoint',    type=str,   required=True,
                        help='Path to trained checkpoint (.pt). Must match --arm: a checkpoint '
                             'trained on UR5e (TCP-pose actions) is not valid for Franka '
                             '(joint-velocity actions), and vice versa.')
    parser.add_argument('--arm',           type=str,   default='franka', choices=['ur5', 'franka'],
                        help='Which arm to deploy on -- must match the arm the checkpoint was trained for.')
    parser.add_argument('--robot_ip',      type=str, default=None,
                        help='Robot IP (default: 150.65.146.87 for ur5, 172.16.0.2 for franka)')
    parser.add_argument('--camera_serial_global', type=str, default=_DEFAULT_CAMERA_SERIAL_GLOBAL,
                        help='RealSense serial for the global camera. Only used when the checkpoint '
                             "was trained with camera_mode 'global' or 'both'.")
    parser.add_argument('--camera_serial_wrist',  type=str, default=_DEFAULT_CAMERA_SERIAL_WRIST,
                        help="RealSense serial for the wrist camera. Only used when the checkpoint "
                             "was trained with camera_mode 'wrist' or 'both'.")
    parser.add_argument('--flowbot_port',  type=str,   default='/dev/ttyACM0',
                        help='Arduino serial port for Flowbot')
    parser.add_argument('--flowbot_baud',  type=int,   default=115200,
                        help='Flowbot serial baud rate')
    parser.add_argument('--max_steps',     type=int,   default=450,
                        help='Max steps per episode')
    parser.add_argument('--num_episodes',  type=int,   default=1,
                        help='Number of episodes to run')
    parser.add_argument('--device',        type=str,   default='cuda',
                        help='Inference device (cuda/cpu)')
    parser.add_argument('--no_start_pose', action='store_true',
                        help='Skip moving to start pose at beginning of each episode')
    parser.add_argument('--quiet',         action='store_true',
                        help='Reduce per-step output')
    parser.add_argument('--log_dir',       type=str,   default=None,
                        help='Directory to save deployment logs (.npz per episode). ')
    parser.add_argument('--position_command_stride', '-skip', type=int, default=1,
                        help='Position control only (UR5e, or Franka with franka_action_space='
                             "'position'): execute only every Nth predicted waypoint instead of "
                             'every one, e.g. 2 on [A,B,C,D,E] executes only [B,D,E]. Gives each '
                             'issued command N*DT instead of DT to settle before the next '
                             'supersedes it -- fixes Franka position-mode jiggle (CartesianMotion '
                             'plans to arrive-and-stop, then gets interrupted mid-brake every '
                             'tick at stride 1). No-op (1) for UR5e/joint_velocity, which have no '
                             'arrive-and-stop semantics to begin with. Tune empirically on '
                             'hardware -- start at 2, increase if jiggle persists.')
    parser.add_argument('--franka_position_speed', type=float, default=FRANKA_POSITION_VELOCITY,
                        help='Franka position mode only (franka_action_space=\'position\'): '
                             'relative_dynamics_factor velocity fraction (0-1) for set_tcp_pose, '
                             'independent of UR5e\'s SERVO_SPEED. NOT literal m/s -- a fraction of '
                             "Franka's own max velocity. Lower = slower. Also the first thing to "
                             'try if set_tcp_pose keeps tripping the "Motion finished commanded, '
                             'but the robot is still moving!" discontinuity reflex, since lowering '
                             'it lowers jerk too (see FRANKA_POSITION_VELOCITY/FRANKA_POSITION_ACCEL '
                             'above).')
    parser.add_argument('--franka_position_accel', type=float, default=FRANKA_POSITION_ACCEL,
                        help='Franka position mode only: relative_dynamics_factor acceleration '
                             'fraction (0-1) for set_tcp_pose. Same caveats as '
                             '--franka_position_speed.')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    robot_ip = args.robot_ip or _DEFAULT_ROBOT_IP[args.arm]

    robot = None
    try:
        robot = RobotDeployment(
            checkpoint_path=args.checkpoint,
            robot_ip=robot_ip,
            arm=args.arm,
            flowbot_port=args.flowbot_port,
            flowbot_baud=args.flowbot_baud,
            device=args.device,
            verbose=not args.quiet,
            camera_serial_global=args.camera_serial_global,
            camera_serial_wrist=args.camera_serial_wrist,
            position_command_stride=args.position_command_stride,
            franka_position_velocity=args.franka_position_speed,
            franka_position_acceleration=args.franka_position_accel,
        )
        if args.log_dir:
            log_dir = Path(args.log_dir)
            if not log_dir.is_absolute():
                log_dir = Path(DEPLOY_DIR) / 'deploy_logs' / log_dir
            logger = DeploymentLogger(str(log_dir), args.checkpoint, tcp_dims=robot.tcp_dims)
            print(f"Logging enabled → {log_dir}")
        else:
            logger = None

        for ep in range(args.num_episodes):
            print(f"\n{'='*30}")
            print(f"EPISODE {ep + 1} / {args.num_episodes}")
            print(f"{'='*30}")

            robot.run_episode(
                max_steps=args.max_steps,
                move_to_start=(ep == 0) and not args.no_start_pose,
                logger=logger,
                episode_idx=ep,
            )

            if ep < args.num_episodes - 1:
                # Return to start so the scene is reset before the user confirms
                if not args.no_start_pose:
                    robot.move_to_start()
                input("\nPress Enter to start next episode (Ctrl+C to abort) ...")

        print("\n✅ All episodes complete!")

    except KeyboardInterrupt:
        print("\n⚠️  Deployment interrupted")
    finally:
        if robot is not None:
            robot.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
