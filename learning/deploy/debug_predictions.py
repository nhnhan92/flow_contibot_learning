#!/usr/bin/env python3
"""
Debug Flowbot PWM predictions during deployment

Loads a checkpoint and runs inference on live camera + robot state,
printing and plotting the predicted TCP pose and PWM signals
WITHOUT actually commanding the robot.

Useful to verify the model is producing sensible outputs before
running the full deploy_real_robot.py.

Usage:
    # With real robot + camera (read-only, no commands sent)
    python deploy/debug_pwm_predictions.py \
        --checkpoint train/checkpoints/best_model.pt \
        --robot_ip 192.168.1.100 \
        --camera_id 0

    # Offline mode: use a zarr episode instead of live hardware
    python deploy/debug_pwm_predictions.py \
        --checkpoint train/checkpoints/best_model.pt \
        --dataset_path data/demo_data/dataset.zarr \
        --episode 0
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEPLOY_DIR)
FLOWBOT_DIR = os.path.join(PROJECT_DIR, 'flowbot')
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, FLOWBOT_DIR)

from train.eval import DiffusionPolicyInference


def _preprocess_state(state_raw, state_min, state_range):
    """Min-Max normalise to [-1, 1]."""
    return (2.0 * (state_raw - state_min) / state_range - 1.0).astype(np.float32)


def _preprocess_image(image_rgb, image_size):
    """uint8 (H,W,3) → float32 (3,H,W) in [-1,1], with centre crop."""
    import cv2
    h, w = image_rgb.shape[:2]
    target_h, target_w = image_size
    crop_h = min(h, int(target_h * 1.5))
    crop_w = min(w, int(target_w * 1.5))
    sh = (h - crop_h) // 2
    sw = (w - crop_w) // 2
    img = image_rgb[sh:sh + crop_h, sw:sw + crop_w]
    img = cv2.resize(img, (target_w, target_h))
    img = (img.astype(np.float32) / 127.5) - 1.0
    return img.transpose(2, 0, 1)   # (3,H,W)


def _denormalize_actions(actions_norm, action_min, action_range):
    """[-1,1] → original scale."""
    return (actions_norm + 1.0) * 0.5 * action_range + action_min


# ── Live mode ─────────────────────────────────────────────────────────────────

def debug_live(policy, robot_ip, camera_id, num_steps=20, sleep_s=0.5):
    """
    Continuously read live robot state + camera and print model predictions.
    The robot receives NO commands.
    """
    import cv2
    try:
        import rtde_receive
    except ImportError:
        print("❌ rtde_receive not installed.")
        return

    config      = policy.config
    obs_horizon = config['obs_horizon']
    image_size  = tuple(config['image_size'])
    state_min   = policy.checkpoint['state_min']
    state_range = policy.checkpoint['state_range']
    action_min  = policy.checkpoint['action_min']
    action_range= policy.checkpoint['action_range']

    try:
        from flowbot import Flowbot
        fb = Flowbot()    # default port; read-only, no commands
        time.sleep(2.0)
        has_flowbot = True
    except Exception:
        has_flowbot = False
        print("⚠️  Flowbot not connected — using zero PWM for state")

    print(f"\nConnecting to UR5e at {robot_ip} ...")
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

    print(f"Opening camera id={camera_id} ...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    for _ in range(5):
        cap.read()

    from collections import deque
    state_buf = deque(maxlen=obs_horizon)
    image_buf = deque(maxlen=obs_horizon)

    # ── Fill buffers ──────────────────────────────────────────────────────────
    print("Filling observation buffer ...")
    for _ in range(obs_horizon):
        tcp = np.array(rtde_r.getActualTCPPose(), dtype=np.float32)
        pwm = np.array(fb.last_pwm if has_flowbot else [0, 0, 0], dtype=np.float32)
        state_raw = np.concatenate([tcp, pwm])
        state_norm = _preprocess_state(state_raw, state_min, state_range)
        state_buf.append(state_norm)

        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_norm = _preprocess_image(image_rgb, image_size)
        image_buf.append(image_norm)

    print(f"\nRunning {num_steps} prediction steps (robot NOT commanded) ...\n")

    import torch
    for step in range(num_steps):
        obs_state = torch.from_numpy(np.stack(list(state_buf))).unsqueeze(0)   # (1,T,9)
        obs_image = torch.from_numpy(np.stack(list(image_buf))).unsqueeze(0)   # (1,T,3,H,W)

        t0 = time.time()
        actions_norm = policy.predict(
            obs_state.squeeze(0), obs_image.squeeze(0)
        ).numpy()
        infer_ms = (time.time() - t0) * 1000

        actions = _denormalize_actions(actions_norm, action_min, action_range)

        # Print first action in the plan
        a0 = actions[0]
        tcp_now = np.array(rtde_r.getActualTCPPose())
        pwm_now = np.array(fb.last_pwm if has_flowbot else [0, 0, 0])

        print(f"Step {step+1:3d}  [{infer_ms:.0f} ms]")
        print(f"  Current  TCP: [{tcp_now[0]:.3f}, {tcp_now[1]:.3f}, {tcp_now[2]:.3f}]  "
              f"PWM: {pwm_now.tolist()}")
        print(f"  Pred[0]  TCP: [{a0[0]:.3f}, {a0[1]:.3f}, {a0[2]:.3f}]  "
              f"PWM: [{a0[6]:.1f}, {a0[7]:.1f}, {a0[8]:.1f}]")

        # Update buffers
        tcp = np.array(rtde_r.getActualTCPPose(), dtype=np.float32)
        pwm = np.array(fb.last_pwm if has_flowbot else [0, 0, 0], dtype=np.float32)
        state_raw = np.concatenate([tcp, pwm])
        state_buf.append(_preprocess_state(state_raw, state_min, state_range))

        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_buf.append(_preprocess_image(image_rgb, image_size))

        time.sleep(sleep_s)

    cap.release()
    rtde_r.disconnect()


# ── Offline mode ──────────────────────────────────────────────────────────────

def debug_offline(policy, dataset_path, episode_idx=0):
    """
    Run inference on a recorded zarr episode and plot predictions vs ground truth.
    No hardware required.
    """
    import zarr
    import torch

    config       = policy.config
    obs_horizon  = config['obs_horizon']
    pred_horizon = config['pred_horizon']
    action_horizon = config['action_horizon']
    image_size   = tuple(config['image_size'])
    state_min    = policy.checkpoint['state_min']
    state_range  = policy.checkpoint['state_range']
    action_min   = policy.checkpoint['action_min']
    action_range = policy.checkpoint['action_range']

    print(f"\nLoading dataset: {dataset_path}")
    root = zarr.open(dataset_path, mode='r')
    episode_ends = root['meta/episode_ends'][:]

    ep_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    ep_end   = int(episode_ends[episode_idx])
    ep_len   = ep_end - ep_start
    print(f"Episode {episode_idx}: steps {ep_start} → {ep_end}  (len={ep_len})")

    from collections import deque
    state_buf = deque(maxlen=obs_horizon)
    image_buf = deque(maxlen=obs_horizon)

    all_pred_xyz  = []
    all_true_xyz  = []
    all_pred_pwm  = []
    all_true_pwm  = []

    stride = action_horizon
    steps  = range(obs_horizon - 1, ep_len - pred_horizon, stride)

    print(f"Running inference on {len(list(steps))} windows ...")

    for step in steps:
        abs_idx = ep_start + step

        # Build obs window
        obs_start = abs_idx - (obs_horizon - 1)
        robot_obs = root['data/robot_eef_pose'][obs_start:abs_idx + 1].astype(np.float32)
        pwm_obs   = root['data/pwm_signals'][obs_start:abs_idx + 1].astype(np.float32)
        state_raw_seq = np.concatenate([robot_obs, pwm_obs], axis=-1)  # (T, 9)

        import cv2
        images = root['data/camera_0'][obs_start:abs_idx + 1]           # (T, H, W, 3)
        proc_images = []
        for img in images:
            proc_images.append(_preprocess_image(img, image_size))

        obs_state = torch.from_numpy(
            np.stack([_preprocess_state(s, state_min, state_range) for s in state_raw_seq])
        ).unsqueeze(0)   # (1, T, 9)
        obs_image = torch.from_numpy(
            np.stack(proc_images)
        ).unsqueeze(0)   # (1, T, 3, H, W)

        actions_norm = policy.predict(
            obs_state.squeeze(0), obs_image.squeeze(0)
        ).numpy()
        actions = _denormalize_actions(actions_norm, action_min, action_range)

        # Ground truth future
        gt_start = abs_idx
        gt_end   = abs_idx + pred_horizon
        gt_robot = root['data/robot_eef_pose'][gt_start:gt_end].astype(np.float32)
        gt_pwm   = root['data/pwm_signals'][gt_start:gt_end].astype(np.float32)

        all_pred_xyz.append(actions[:, :3])
        all_true_xyz.append(gt_robot[:, :3])
        all_pred_pwm.append(actions[:, 6:])
        all_true_pwm.append(gt_pwm)

    pred_xyz = np.concatenate(all_pred_xyz)
    true_xyz = np.concatenate(all_true_xyz)
    pred_pwm = np.concatenate(all_pred_pwm)
    true_pwm = np.concatenate(all_true_pwm)

    t = np.arange(len(pred_xyz))

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'PWM Predictions Debug — Episode {episode_idx}', fontsize=14)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # XYZ trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        ax1.plot(t, pred_xyz[:, i], color=color, linestyle='-',  label=f'Pred {label}', alpha=0.8)
        ax1.plot(t, true_xyz[:, i], color=color, linestyle='--', label=f'True {label}', alpha=0.5)
    ax1.set_title('TCP Position (m)')
    ax1.set_xlabel('Timestep')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Position error
    ax2 = fig.add_subplot(gs[0, 1])
    pos_err = np.linalg.norm(pred_xyz - true_xyz, axis=1) * 100  # cm
    ax2.plot(t, pos_err, 'r-', linewidth=1.5)
    ax2.axhline(pos_err.mean(), color='b', linestyle='--',
                label=f'Mean: {pos_err.mean():.2f} cm')
    ax2.set_title('Position Error (cm)')
    ax2.set_xlabel('Timestep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # PWM signals
    ax3 = fig.add_subplot(gs[1, 0])
    colors_pwm = ['tab:blue', 'tab:orange', 'tab:green']
    for i, color in enumerate(colors_pwm):
        ax3.plot(t, pred_pwm[:, i], color=color, linestyle='-',
                 label=f'Pred PWM{i+1}', alpha=0.8)
        ax3.plot(t, true_pwm[:, i], color=color, linestyle='--',
                 label=f'True PWM{i+1}', alpha=0.5)
    ax3.set_title('Flowbot PWM Signals')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('PWM value')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # PWM error per channel
    ax4 = fig.add_subplot(gs[1, 1])
    for i, color in enumerate(colors_pwm):
        pwm_err_i = np.abs(pred_pwm[:, i] - true_pwm[:, i])
        ax4.plot(t, pwm_err_i, color=color, label=f'PWM{i+1} MAE={pwm_err_i.mean():.1f}')
    ax4.set_title('PWM Absolute Error per Channel')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('|Pred - True|')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'debug_pwm_ep{episode_idx}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: {out_path}")

    # Summary statistics
    print(f"\nPosition Error: mean={pos_err.mean():.2f} cm, max={pos_err.max():.2f} cm")
    for i in range(3):
        pwm_err = np.abs(pred_pwm[:, i] - true_pwm[:, i])
        print(f"PWM{i+1} Error:   mean={pwm_err.mean():.2f}, max={pwm_err.max():.2f}")

    try:
        plt.show()
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Debug PWM predictions from trained policy')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--device',       type=str, default='cpu',
                        help='Inference device (cpu recommended for debugging)')

    # Live mode
    parser.add_argument('--robot_ip',     type=str, default=None,
                        help='UR5e IP for live mode')
    parser.add_argument('--camera_id',    type=int, default=0,
                        help='Camera device ID for live mode')
    parser.add_argument('--num_steps',    type=int, default=20,
                        help='Number of prediction steps (live mode)')
    parser.add_argument('--sleep',        type=float, default=0.5,
                        help='Sleep between steps in seconds (live mode)')

    # Offline mode
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to zarr dataset for offline mode')
    parser.add_argument('--episode',      type=int, default=0,
                        help='Episode index for offline mode')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    print(f"Loading policy from: {args.checkpoint}")
    import torch
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    policy = DiffusionPolicyInference(args.checkpoint, device=str(device))
    print("✅ Policy loaded")

    if args.dataset_path is not None:
        # Offline mode
        debug_offline(policy, args.dataset_path, args.episode)
    elif args.robot_ip is not None:
        # Live mode
        debug_live(policy, args.robot_ip, args.camera_id,
                   num_steps=args.num_steps, sleep_s=args.sleep)
    else:
        print("❌ Provide either --robot_ip (live) or --dataset_path (offline)")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
