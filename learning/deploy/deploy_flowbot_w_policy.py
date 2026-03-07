#!/usr/bin/env python3
"""
Deploy trained Diffusion Policy on real UR5e + Flowbot soft manipulator

Usage:
    python deploy/deploy_real_robot.py \
        --checkpoint train/checkpoints/best_model.pt \
        --robot_ip 192.168.1.100 \
        --flowbot_port /dev/ttyACM0

Hardware:
    - UR5e robot arm (RTDE control)
    - Flowbot soft pneumatic manipulator (3 valves via Arduino serial)
    - Intel RealSense camera (auto-detected)

State  (8D): robot TCP xyz (3D) + flowbot pwm (3D) + operation_mode (2D)  [rotation excluded]
Action (8D): target TCP xyz (3D) + target pwm (3D) + operation_mode (2D)  [rotation excluded]
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
from hardware.flowbot import flowbot
from hardware.realsense_camera import RealSenseCamera
from train.eval import DiffusionPolicyInference

# ── Constants ─────────────────────────────────────────────────────────────────
PWM_MIN = 0   # 0 = fully deflated (release); model must be able to command this
PWM_MAX = 26

# Default start pose (from collect_demos_with_camera.py)
DEFAULT_START_POSE = [0.20636, -0.46706, 0.44268, 3.14, -0.14, 0.0]

# Fixed TCP rotation used when executing XYZ-only actions from the policy.
# Rotation is not predicted by the model (action_dim=8) so we hold it constant.
TCP_FIXED_ROTATION = DEFAULT_START_POSE[3:]   # [rx, ry, rz]

# Control frequency (Hz)
CONTROL_FREQ =8.0
DT = 1.0 / CONTROL_FREQ
DT_FLOWBOT = 0.3     # Step time (s) when flowbot is actively actuating
FLOWBOT_FREQ = 10.0  # Flowbot command frequency — must match CONTROL_FREQ

# servo_l speed/acceleration (lower = smoother)
SERVO_SPEED = 0.05     # m/s
SERVO_ACCEL = 0.05     # m/s^2
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
        flowbot_port: str = '/dev/ttyACM0',
        flowbot_baud: int = 115200,
        image_size: tuple = (216, 288),
        device: str = 'cuda',
        verbose: bool = True,
        camera_height: int = 480,
        camera_width: int = 640,
    ):
        self.verbose = verbose
        self.current_pwm = np.array([0, 0, 0], dtype=int)
        self.prev_pwm    = np.zeros(3, dtype=np.float32)   # command from previous step

        # ── Load policy ───────────────────────────────────────────────────────
        print(f"\n[1/4] Loading policy from: {checkpoint_path}")
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy = DiffusionPolicyInference(checkpoint_path, device=str(device_obj))
        self.config = self.policy.config
        self.obs_horizon = self.config['obs_horizon']
        self.action_horizon = self.config['action_horizon']
        self.tcp_dims = self.config.get('tcp_dims', 3)   # 3=xyz only, 6=xyz+rotation
        print(f"      obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}")
        print(f"      tcp_dims={self.tcp_dims}  ({'xyz only' if self.tcp_dims == 3 else 'xyz+rotation'})")
        print(f"      device={device_obj}")
        if image_size is None:
            self.image_size = tuple(self.policy.config['image_size'])
        else:
            self.image_size = image_size
        # ── Camera ────────────────────────────────────────────────────────────
        print(f"\n[2/4] Opening RealSense camera ...")
        self.cam = RealSenseCamera(
            width=camera_width,
            height=camera_height,
            enable_depth=False,
        )
        print("      Camera OK")

        # ── UR5e RTDE ─────────────────────────────────────────────────────────
        print(f"\n[3/4] Connecting to UR5e at {robot_ip} ...")
        self.ur5 = UR5eRobot(robot_ip=robot_ip,frequency=CONTROL_FREQ)
        print("      UR5e connected")

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
        self.current_op_mode = np.zeros(2, dtype=np.float32)

        print("\n✅ All systems ready!\n")

    # ── Low-level observation ─────────────────────────────────────────────────

    def _get_raw_observation(self):
        """
        Read current robot state and camera image.

        Returns:
            state_raw  : np.ndarray (tcp_dims+5,) — [tcp[:tcp_dims], pwm1,pwm2,pwm3, ur5_active, flowbot_active]
            image_raw  : np.ndarray (H,W,3) uint8 — cropped camera frame
        """
        # Robot TCP pose — slice to tcp_dims (3=xyz only, 6=xyz+rotation)
        tcp_pose = self.ur5.get_tcp_pose()

        # PWM from previous step — matches the physical state visible in the current image
        pwm = self.prev_pwm.copy()                                                          # (3,)

        # Operation mode from last executed action (2D)
        state_raw = np.concatenate([tcp_pose[:self.tcp_dims], pwm, self.current_op_mode])  # (tcp_dims+5,)

        # Camera image
        camera_frame, _ = self.cam.get_frames()
        if camera_frame is None:
            raise RuntimeError("Camera read failed")
        
        # Centre-crop and resize (same as dataset.py)
        h, w = camera_frame.shape[:2]
        target_h, target_w = self.image_size
        crop_h = min(h, int(target_h * 1.5))
        crop_w = min(w, int(target_w * 1.5))
        sh = (h - crop_h) // 2
        sw = (w - crop_w) // 2
        image_raw = camera_frame[sh:sh + crop_h, sw:sw + crop_w]
        image_raw = cv2.resize(image_raw, (target_w, target_h))
        
        return state_raw, image_raw

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
        state_raw, image_raw = self._get_raw_observation()
        state_norm = self._preprocess_state(state_raw)
        image_norm = self._preprocess_image(image_raw)
        for _ in range(self.obs_horizon):
            self.state_buffer.append(state_norm.copy())
            self.image_buffer.append(image_norm.copy())

    def _update_obs_buffer(self):
        """Append one new observation to the rolling buffer. Returns state_raw for logging."""
        state_raw, image_raw = self._get_raw_observation()
        state_norm = self._preprocess_state(state_raw)
        image_norm = self._preprocess_image(image_raw)
        self.state_buffer.append(state_norm)
        self.image_buffer.append(image_norm)
        return state_raw

    def _get_obs_tensors(self):
        """
        Stack buffer contents into tensors for the policy.

        Returns:
            obs_state : torch.Tensor (1, obs_horizon, 8)
            obs_image : torch.Tensor (1, obs_horizon, 3, H, W)
        """
        obs_state = torch.from_numpy(
            np.stack(list(self.state_buffer), axis=0)   # (obs_horizon, 8)
        ).unsqueeze(0)                                   # (1, obs_horizon, 8)

        obs_image = torch.from_numpy(
            np.stack(list(self.image_buffer), axis=0)   # (obs_horizon, 3, H, W)
        ).unsqueeze(0)                                   # (1, obs_horizon, 3, H, W)

        return obs_state, obs_image

    # ── Policy inference ──────────────────────────────────────────────────────

    def _predict_actions(self):
        """
        Run one DDIM inference step.

        Returns:
            actions : np.ndarray (pred_horizon, 8) — denormalised actions
                      [:3] = TCP xyz, [3:6] = flowbot PWM (float), [6:8] = op_mode (~0 or ~1)
        """
        obs_state, obs_image = self._get_obs_tensors()
        actions_norm = self.policy.predict(
            obs_state.squeeze(0),   # (obs_horizon, 8)
            obs_image.squeeze(0),   # (obs_horizon, 3, H, W)
        ).numpy()                   # (pred_horizon, 8)

        # Denormalise: x = (x_norm + 1) * 0.5 * range + min
        action_min   = self.policy.checkpoint['action_min']
        action_range = self.policy.checkpoint['action_range']
        actions = (actions_norm + 1.0) * 0.5 * action_range + action_min
        return actions              # (pred_horizon, 8)

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_action(self, action: np.ndarray):
        """
        Send one action step to the robot and flowbot, gated by predicted operation mode.

        Args:
            action : np.ndarray (tcp_dims+5,) — [tcp[:tcp_dims], pwm1,pwm2,pwm3, ur5_active, flowbot_active]
                     When tcp_dims=3 the fixed rotation TCP_FIXED_ROTATION is appended to form 6D target.

        Returns:
            pwm_int      : np.ndarray (3,) int — clamped PWM actually sent
            op_mode_pred : np.ndarray (2,) int — [ur5_active, flowbot_active]
        """
        d = self.tcp_dims
        # Build 6D TCP target for servo_tcp_pose
        if d == 6:
            tcp_target = action[:6].tolist()
        else:  # d == 3: append fixed rotation so the robot holds its orientation
            tcp_target = action[:3].tolist() + TCP_FIXED_ROTATION
        pwm_raw    = action[d:d+3]
        pwm_int    = np.clip(np.round(pwm_raw), PWM_MIN, PWM_MAX).astype(int)

        # Drop protection: if any channel decreases vs last sent PWM, hold previous value
        if np.any(pwm_int < self.current_pwm):
            pwm_int = self.current_pwm.copy()

        # Decode predicted operation mode (denorm ~[0,1] → binary)
        op_mode_pred = np.clip(np.round(action[d+3:d+5]), 0, 1).astype(int)

        # Gate UR5 servo: only move when ur5_active
        if op_mode_pred[0] == 1:
            self.ur5.servo_tcp_pose(target_pose=tcp_target, velocity=SERVO_SPEED,
                                    acceleration=SERVO_ACCEL, dt=DT,
                                    lookahead_time=SERVO_LOOKAHEAD, gain=SERVO_GAIN)

        # Gate flowbot PWM: only send when flowbot_active
        if op_mode_pred[1] == 1 and np.any(pwm_int >= PWM_MIN):
            # pwm_int[0] = 0  # this is just a trick
            self.fb.serial_sending(pwm_int, wait_ack=True, ack_timeout=DT_FLOWBOT)
            self.current_pwm = pwm_int.copy()

        # Update tracked op_mode for next observation
        self.current_op_mode = op_mode_pred.astype(np.float32)

        if self.verbose:
            tcp = np.array(tcp_target, dtype=np.float32)
            mode_str = ['idle', 'FB', 'UR5', 'release'][op_mode_pred[0] * 2 + op_mode_pred[1]]
            print(
                f"  [{mode_str}] TCP: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]  "
                f"PWM: {pwm_int.tolist()}"
            )

        return pwm_int, op_mode_pred

    # ── Start position ────────────────────────────────────────────────────────

    def move_to_start(self, speed: float = 0.1, accel: float = 0.1):
        """
        Move UR5e to DEFAULT_START_POSE using moveL, then reset flowbot.
        """
        print("\nMoving to start position ...")
        self.ur5.move_tcp_pose(DEFAULT_START_POSE, velocity=speed, acceleration=accel)
        print(f"  TCP at: {DEFAULT_START_POSE}")

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
                    pwm_int, op_mode_pred = self._execute_action(action)

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
                        _, image_raw = self._get_raw_observation()   # second read just for display
                        cv2.imshow("Live", cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    if logger is not None:
                        logger.log_step(state_raw, action, pwm_int)

                    total_steps += 1

        except _ReleaseDetected:
            print("✅ Episode ended by release phase")
        except KeyboardInterrupt:
            print("\n⚠️  Episode interrupted by user")

        elapsed_total = time.time() - episode_start
        print(f"\n✅ Episode finished: {total_steps} steps in {elapsed_total:.1f}s")

        # Stop UR5e servoing and allow RTDE to settle before any subsequent moveL
        self.ur5.stop()
        time.sleep(0.5)

        # Reset Flowbot
        print("Resetting Flowbot ...")
        self.fb.reset()

        # Save deployment log
        if logger is not None:
            logger.save(episode_idx, total_steps, elapsed_total)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self):
        """Safely disconnect all hardware."""
        print("\nShutting down ...")
        try:
            self.ur5.disconnect()
        except Exception as e:
            print(f"  UR5e shutdown error: {e}")
        try:
            self.fb.reset()
        except Exception as e:
            print(f"  Flowbot reset error: {e}")
        try:
            self.cam.stop()
        except Exception:
            pass
        print("✅ Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='Deploy Diffusion Policy on UR5e + Flowbot')
    parser.add_argument('--checkpoint',    type=str,   required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--robot_ip',      type=str, default= "150.65.146.87",
                        help='UR5e IP address (e.g. 192.168.1.100)')
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
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    robot = None
    try:
        robot = RobotDeployment(
            checkpoint_path=args.checkpoint,
            robot_ip=args.robot_ip,
            flowbot_port=args.flowbot_port,
            flowbot_baud=args.flowbot_baud,
            device=args.device,
            verbose=not args.quiet,
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
