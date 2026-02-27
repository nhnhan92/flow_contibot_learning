#!/usr/bin/env python3
"""
Deploy trained Diffusion Policy on real UR5e + Flowbot soft manipulator

Usage:
    python deploy/deploy_real_robot.py \
        --checkpoint train/checkpoints/best_model.pt \
        --robot_ip 192.168.1.100 \
        --camera_id 0

Hardware:
    - UR5e robot arm (RTDE control)
    - Flowbot soft pneumatic manipulator (3 valves via Arduino serial)
    - USB camera (camera_id)

State  (9D): robot_eef_pose (6D: x,y,z,rx,ry,rz) + flowbot pwm (3D)
Action (9D): target robot_eef_pose (6D) + target pwm (3D)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import deque

# Add parent directory to path
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEPLOY_DIR)
sys.path.insert(0, PROJECT_DIR)

from train.eval import DiffusionPolicyInference

# ── Hardware imports ──────────────────────────────────────────────────────────
try:
    import rtde_control
    import rtde_receive
    HAS_RTDE = True
except ImportError:
    HAS_RTDE = False
    print("⚠️  rtde_control not found — robot control disabled")

try:
    FLOWBOT_DIR = os.path.join(PROJECT_DIR, 'flowbot')
    sys.path.insert(0, FLOWBOT_DIR)
    from flowbot import Flowbot
    HAS_FLOWBOT = True
except ImportError:
    HAS_FLOWBOT = False
    print("⚠️  flowbot module not found — PWM control disabled")

# ── Constants ─────────────────────────────────────────────────────────────────
PWM_MIN = 1
PWM_MAX = 26

# Default start pose (from collect_demos_with_camera.py)
DEFAULT_START_POSE = [0.20636, -0.46706, 0.44268, 3.14, -0.14, 0.0]

# Control frequency (Hz)
CONTROL_FREQ = 10.0
DT = 1.0 / CONTROL_FREQ

# servo_l speed/acceleration (lower = smoother)
SERVO_SPEED = 0.3       # m/s
SERVO_ACCEL = 0.3       # m/s^2
SERVO_LOOKAHEAD = 0.1   # s
SERVO_GAIN = 300


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
        camera_id: int = 0,
        flowbot_port: str = '/dev/ttyACM0',
        flowbot_baud: int = 115200,
        image_size: tuple = (96, 96),
        device: str = 'cuda',
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.image_size = image_size

        # ── Load policy ───────────────────────────────────────────────────────
        print(f"\n[1/4] Loading policy from: {checkpoint_path}")
        device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy = DiffusionPolicyInference(checkpoint_path, device=str(device_obj))
        self.config = self.policy.config
        self.obs_horizon = self.config['obs_horizon']
        self.action_horizon = self.config['action_horizon']
        print(f"      obs_horizon={self.obs_horizon}, action_horizon={self.action_horizon}")
        print(f"      device={device_obj}")

        # ── Camera ────────────────────────────────────────────────────────────
        print(f"\n[2/4] Opening camera (id={camera_id}) ...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera id={camera_id}")
        # Warmup
        for _ in range(5):
            self.cap.read()
        print("      Camera OK")

        # ── UR5e RTDE ─────────────────────────────────────────────────────────
        print(f"\n[3/4] Connecting to UR5e at {robot_ip} ...")
        if not HAS_RTDE:
            raise ImportError("rtde_control is required for robot control")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("      UR5e connected")

        # ── Flowbot ───────────────────────────────────────────────────────────
        print(f"\n[4/4] Connecting to Flowbot on {flowbot_port} ...")
        if not HAS_FLOWBOT:
            raise ImportError("flowbot module is required")
        self.fb = Flowbot(port=flowbot_port, baud=flowbot_baud)
        time.sleep(2.0)  # Arduino reset delay
        print("      Flowbot connected")

        # ── Observation buffers ───────────────────────────────────────────────
        self.state_buffer = deque(maxlen=self.obs_horizon)
        self.image_buffer = deque(maxlen=self.obs_horizon)

        print("\n✅ All systems ready!\n")

    # ── Low-level observation ─────────────────────────────────────────────────

    def _get_raw_observation(self):
        """
        Read current robot state and camera image.

        Returns:
            state_raw  : np.ndarray (9,)  — [x,y,z,rx,ry,rz, pwm1,pwm2,pwm3]
            image_raw  : np.ndarray (H,W,3) uint8 — cropped camera frame
        """
        # Robot TCP pose (6D)
        tcp_pose = np.array(self.rtde_r.getActualTCPPose(), dtype=np.float32)  # (6,)

        # Flowbot last sent PWM (3D)
        pwm = np.array(self.fb.last_pwm, dtype=np.float32)                     # (3,)

        state_raw = np.concatenate([tcp_pose, pwm])                            # (9,)

        # Camera image
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Centre-crop and resize (same as dataset.py)
        h, w = frame_rgb.shape[:2]
        target_h, target_w = self.image_size
        crop_h = min(h, int(target_h * 1.5))
        crop_w = min(w, int(target_w * 1.5))
        sh = (h - crop_h) // 2
        sw = (w - crop_w) // 2
        image_raw = frame_rgb[sh:sh + crop_h, sw:sw + crop_w]
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
        """Append one new observation to the rolling buffer."""
        state_raw, image_raw = self._get_raw_observation()
        state_norm = self._preprocess_state(state_raw)
        image_norm = self._preprocess_image(image_raw)
        self.state_buffer.append(state_norm)
        self.image_buffer.append(image_norm)

    def _get_obs_tensors(self):
        """
        Stack buffer contents into tensors for the policy.

        Returns:
            obs_state : torch.Tensor (1, obs_horizon, 9)
            obs_image : torch.Tensor (1, obs_horizon, 3, H, W)
        """
        obs_state = torch.from_numpy(
            np.stack(list(self.state_buffer), axis=0)   # (obs_horizon, 9)
        ).unsqueeze(0)                                   # (1, obs_horizon, 9)

        obs_image = torch.from_numpy(
            np.stack(list(self.image_buffer), axis=0)   # (obs_horizon, 3, H, W)
        ).unsqueeze(0)                                   # (1, obs_horizon, 3, H, W)

        return obs_state, obs_image

    # ── Policy inference ──────────────────────────────────────────────────────

    def _predict_actions(self):
        """
        Run one DDIM inference step.

        Returns:
            actions : np.ndarray (pred_horizon, 9) — denormalised actions
                      [:6] = TCP pose, [6:] = flowbot PWM (float)
        """
        obs_state, obs_image = self._get_obs_tensors()
        actions_norm = self.policy.predict(
            obs_state.squeeze(0),   # (obs_horizon, 9)
            obs_image.squeeze(0),   # (obs_horizon, 3, H, W)
        ).numpy()                   # (pred_horizon, 9)

        # Denormalise: x = (x_norm + 1) * 0.5 * range + min
        action_min   = self.policy.checkpoint['action_min']
        action_range = self.policy.checkpoint['action_range']
        actions = (actions_norm + 1.0) * 0.5 * action_range + action_min
        return actions              # (pred_horizon, 9)

    # ── Action execution ──────────────────────────────────────────────────────

    def _execute_action(self, action: np.ndarray):
        """
        Send one action step to the robot and flowbot.

        Args:
            action : np.ndarray (9,) — [x,y,z,rx,ry,rz, pwm1,pwm2,pwm3]
        """
        # UR5e: servo to target TCP pose (non-blocking)
        tcp_target = action[:6].tolist()
        self.rtde_c.servoL(
            tcp_target,
            SERVO_SPEED,
            SERVO_ACCEL,
            DT,
            SERVO_LOOKAHEAD,
            SERVO_GAIN,
        )

        # Flowbot: clamp, round to int, send
        pwm_raw   = action[6:]
        pwm_int   = np.clip(np.round(pwm_raw), PWM_MIN, PWM_MAX).astype(int)
        self.fb.serial_sending(pwm_int)

        if self.verbose:
            tcp = np.array(self.rtde_r.getActualTCPPose())
            print(
                f"  TCP: [{tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}]  "
                f"PWM: {pwm_int.tolist()}"
            )

    # ── Start position ────────────────────────────────────────────────────────

    def move_to_start(self, speed: float = 0.3, accel: float = 0.3):
        """
        Move UR5e to DEFAULT_START_POSE using moveL, then reset flowbot.
        """
        print("\nMoving to start position ...")
        self.rtde_c.moveL(DEFAULT_START_POSE, speed, accel)
        print(f"  TCP at: {DEFAULT_START_POSE}")

        print("Resetting Flowbot ...")
        self.fb.reset()
        time.sleep(0.5)
        print("  Flowbot reset OK")

    # ── Main episode loop ─────────────────────────────────────────────────────

    def run_episode(self, max_steps: int = 300, move_to_start: bool = True):
        """
        Run one deployment episode with receding-horizon control.

        The policy produces `pred_horizon` actions; we execute `action_horizon`
        of them before re-planning — identical to training's receding horizon.

        Args:
            max_steps    : Hard step limit (safety stop)
            move_to_start: If True, move robot to start before running
        """
        if move_to_start:
            self.move_to_start()

        print("\n" + "="*60)
        print("Starting episode ...")
        print("="*60)

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

                # ── Execute: action_horizon steps from the plan ───────────────
                for step_i in range(self.action_horizon):
                    if total_steps >= max_steps:
                        break

                    t_step_start = time.time()

                    action = actions[step_i]    # (9,)
                    self._execute_action(action)

                    # Update observation buffer after executing action
                    self._update_obs_buffer()

                    total_steps += 1

                    # Pace to CONTROL_FREQ
                    elapsed = time.time() - t_step_start
                    sleep_time = DT - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n⚠️  Episode interrupted by user")

        elapsed_total = time.time() - episode_start
        print(f"\n✅ Episode finished: {total_steps} steps in {elapsed_total:.1f}s")

        # Stop UR5e servoing
        self.rtde_c.servoStop()

        # Reset Flowbot
        print("Resetting Flowbot ...")
        self.fb.reset()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def shutdown(self):
        """Safely disconnect all hardware."""
        print("\nShutting down ...")
        try:
            self.rtde_c.servoStop()
            self.rtde_c.stopScript()
        except Exception as e:
            print(f"  UR5e shutdown error: {e}")
        try:
            self.fb.reset()
        except Exception as e:
            print(f"  Flowbot reset error: {e}")
        try:
            self.cap.release()
        except Exception:
            pass
        print("✅ Shutdown complete")


def main():
    parser = argparse.ArgumentParser(description='Deploy Diffusion Policy on UR5e + Flowbot')
    parser.add_argument('--checkpoint',    type=str,   required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--robot_ip',      type=str,   required=True,
                        help='UR5e IP address (e.g. 192.168.1.100)')
    parser.add_argument('--camera_id',     type=int,   default=0,
                        help='Camera device ID')
    parser.add_argument('--flowbot_port',  type=str,   default='/dev/ttyACM0',
                        help='Arduino serial port for Flowbot')
    parser.add_argument('--flowbot_baud',  type=int,   default=115200,
                        help='Flowbot serial baud rate')
    parser.add_argument('--max_steps',     type=int,   default=300,
                        help='Max steps per episode')
    parser.add_argument('--num_episodes',  type=int,   default=1,
                        help='Number of episodes to run')
    parser.add_argument('--device',        type=str,   default='cuda',
                        help='Inference device (cuda/cpu)')
    parser.add_argument('--no_start_pose', action='store_true',
                        help='Skip moving to start pose at beginning of each episode')
    parser.add_argument('--quiet',         action='store_true',
                        help='Reduce per-step output')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    robot = None
    try:
        robot = RobotDeployment(
            checkpoint_path=args.checkpoint,
            robot_ip=args.robot_ip,
            camera_id=args.camera_id,
            flowbot_port=args.flowbot_port,
            flowbot_baud=args.flowbot_baud,
            device=args.device,
            verbose=not args.quiet,
        )

        for ep in range(args.num_episodes):
            print(f"\n{'='*60}")
            print(f"EPISODE {ep + 1} / {args.num_episodes}")
            print(f"{'='*60}")

            robot.run_episode(
                max_steps=args.max_steps,
                move_to_start=not args.no_start_pose,
            )

            if ep < args.num_episodes - 1:
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