#!/usr/bin/env python3
"""
Visualize predicted vs actual trajectories from a zarr episode

Shows:
  - 3D TCP position trajectory (predicted vs actual)
  - XYZ time-series
  - Flowbot PWM signals (predicted vs actual, 3 channels)
  - Position & PWM error over time

Usage:
    python deploy/visualize_predictions.py \
        --checkpoint train/checkpoints/best_model.pt \
        --dataset_path data/demo_data/dataset.zarr \
        --episode 0

Output:
    Saves 'visualization_ep{N}.png' in the current directory.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3D projection)

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DEPLOY_DIR)
sys.path.insert(0, PROJECT_DIR)

from train.eval import DiffusionPolicyInference


def _preprocess_state(state_raw, state_min, state_range):
    return (2.0 * (state_raw - state_min) / state_range - 1.0).astype(np.float32)


def _preprocess_image(image_rgb, image_size):
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


def _denormalize(data_norm, data_min, data_range):
    return (data_norm + 1.0) * 0.5 * data_range + data_min


def run_episode_inference(policy, dataset_path, episode_idx):
    """
    Run model inference across an entire episode and collect
    predicted + actual trajectories.

    Returns dicts with:
        'xyz'  : (N, 3)
        'pwm'  : (N, 3)
    """
    import zarr
    import torch

    config         = policy.config
    obs_horizon    = config['obs_horizon']
    pred_horizon   = config['pred_horizon']
    action_horizon = config['action_horizon']
    image_size     = tuple(config['image_size'])
    state_min      = policy.checkpoint['state_min']
    state_range    = policy.checkpoint['state_range']
    action_min     = policy.checkpoint['action_min']
    action_range   = policy.checkpoint['action_range']

    root = zarr.open(dataset_path, mode='r')
    episode_ends = root['meta/episode_ends'][:]
    ep_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    ep_end   = int(episode_ends[episode_idx])
    ep_len   = ep_end - ep_start

    print(f"Episode {episode_idx}: {ep_len} steps")

    pred_xyz_all = []
    pred_pwm_all = []
    true_xyz_all = []
    true_pwm_all = []

    stride = action_horizon
    steps  = list(range(obs_horizon - 1, ep_len - pred_horizon, stride))

    for step in steps:
        abs_idx   = ep_start + step
        obs_start = abs_idx - (obs_horizon - 1)

        # State observation window
        robot_obs = root['data/robot_eef_pose'][obs_start:abs_idx + 1].astype(np.float32)
        pwm_obs   = root['data/pwm_signals'][obs_start:abs_idx + 1].astype(np.float32)
        states_raw = np.concatenate([robot_obs, pwm_obs], axis=-1)   # (T,9)
        states_norm = np.stack([_preprocess_state(s, state_min, state_range)
                                for s in states_raw])                  # (T,9)

        # Image observation window
        import cv2
        imgs_raw = root['data/camera_0'][obs_start:abs_idx + 1]       # (T,H,W,3)
        imgs_norm = np.stack([_preprocess_image(img, image_size) for img in imgs_raw])

        obs_state = torch.from_numpy(states_norm).unsqueeze(0)  # (1,T,9)
        obs_image = torch.from_numpy(imgs_norm).unsqueeze(0)    # (1,T,3,H,W)

        actions_norm = policy.predict(
            obs_state.squeeze(0), obs_image.squeeze(0)
        ).numpy()                                                # (pred_horizon,9)
        actions = _denormalize(actions_norm, action_min, action_range)

        # Ground truth
        gt_robot = root['data/robot_eef_pose'][abs_idx:abs_idx + pred_horizon].astype(np.float32)
        gt_pwm   = root['data/pwm_signals'][abs_idx:abs_idx + pred_horizon].astype(np.float32)

        pred_xyz_all.append(actions[:, :3])
        pred_pwm_all.append(actions[:, 6:])
        true_xyz_all.append(gt_robot[:, :3])
        true_pwm_all.append(gt_pwm)

    return {
        'pred': {
            'xyz': np.concatenate(pred_xyz_all),
            'pwm': np.concatenate(pred_pwm_all),
        },
        'true': {
            'xyz': np.concatenate(true_xyz_all),
            'pwm': np.concatenate(true_pwm_all),
        },
    }


def visualize(data: dict, episode_idx: int, output_path: str):
    """
    Create a 6-panel figure showing predicted vs actual trajectories.

    Panels:
        [0,0] 3D XYZ trajectory
        [0,1] XYZ time-series
        [1,0] PWM signals (all 3 channels)
        [1,1] Position error over time
        [2,0] PWM per-channel error
        [2,1] PWM error histogram
    """
    pred_xyz = data['pred']['xyz']
    true_xyz = data['true']['xyz']
    pred_pwm = data['pred']['pwm']
    true_pwm = data['true']['pwm']

    t = np.arange(len(pred_xyz))
    pos_err = np.linalg.norm(pred_xyz - true_xyz, axis=1) * 100   # cm
    pwm_err = [np.abs(pred_pwm[:, i] - true_pwm[:, i]) for i in range(3)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Diffusion Policy — Episode {episode_idx}: Predicted vs Actual', fontsize=15)

    # ── Panel 1: 3D trajectory ────────────────────────────────────────────────
    ax3d = fig.add_subplot(3, 2, 1, projection='3d')
    ax3d.plot(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2],
              'b-', linewidth=2, alpha=0.7, label='Actual')
    ax3d.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2],
              'r--', linewidth=2, alpha=0.7, label='Predicted')
    ax3d.scatter(*true_xyz[0],  c='green', s=60, zorder=5, label='Start')
    ax3d.scatter(*true_xyz[-1], c='red',   s=60, zorder=5, label='End')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D TCP Position Trajectory')
    ax3d.legend(fontsize=7)

    # ── Panel 2: XYZ time-series ──────────────────────────────────────────────
    ax_xyz = fig.add_subplot(3, 2, 2)
    for i, (axis, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        ax_xyz.plot(t, pred_xyz[:, i], color=color, linestyle='-',
                    label=f'Pred {axis}', alpha=0.8)
        ax_xyz.plot(t, true_xyz[:, i], color=color, linestyle='--',
                    label=f'True {axis}', alpha=0.5)
    ax_xyz.set_title('TCP Position Time-series')
    ax_xyz.set_xlabel('Timestep')
    ax_xyz.set_ylabel('Position (m)')
    ax_xyz.legend(fontsize=7, ncol=2)
    ax_xyz.grid(True, alpha=0.3)

    # ── Panel 3: PWM signals ──────────────────────────────────────────────────
    ax_pwm = fig.add_subplot(3, 2, 3)
    colors_pwm = ['tab:blue', 'tab:orange', 'tab:green']
    for i, color in enumerate(colors_pwm):
        ax_pwm.plot(t, pred_pwm[:, i], color=color, linestyle='-',
                    label=f'Pred PWM{i+1}', alpha=0.8)
        ax_pwm.plot(t, true_pwm[:, i], color=color, linestyle='--',
                    label=f'True PWM{i+1}', alpha=0.5)
    ax_pwm.set_title('Flowbot PWM Signals')
    ax_pwm.set_xlabel('Timestep')
    ax_pwm.set_ylabel('PWM value')
    ax_pwm.legend(fontsize=7, ncol=2)
    ax_pwm.grid(True, alpha=0.3)

    # ── Panel 4: Position error over time ─────────────────────────────────────
    ax_perr = fig.add_subplot(3, 2, 4)
    ax_perr.plot(t, pos_err, 'r-', linewidth=1.5)
    ax_perr.axhline(pos_err.mean(), color='b', linestyle='--',
                    label=f'Mean: {pos_err.mean():.2f} cm')
    ax_perr.fill_between(t, 0, pos_err, alpha=0.2, color='red')
    ax_perr.set_title('Position Error Over Time')
    ax_perr.set_xlabel('Timestep')
    ax_perr.set_ylabel('Error (cm)')
    ax_perr.legend()
    ax_perr.grid(True, alpha=0.3)

    # ── Panel 5: PWM per-channel error ────────────────────────────────────────
    ax_pwmerr = fig.add_subplot(3, 2, 5)
    for i, color in enumerate(colors_pwm):
        ax_pwmerr.plot(t, pwm_err[i], color=color,
                       label=f'PWM{i+1} (mean={pwm_err[i].mean():.1f})', alpha=0.8)
    ax_pwmerr.set_title('PWM Absolute Error per Channel')
    ax_pwmerr.set_xlabel('Timestep')
    ax_pwmerr.set_ylabel('|Pred - True|')
    ax_pwmerr.legend(fontsize=8)
    ax_pwmerr.grid(True, alpha=0.3)

    # ── Panel 6: PWM error distribution ──────────────────────────────────────
    ax_hist = fig.add_subplot(3, 2, 6)
    combined_pwm_err = np.abs(pred_pwm - true_pwm).mean(axis=1)
    ax_hist.hist(combined_pwm_err, bins=30, edgecolor='black', alpha=0.7, color='tab:purple')
    ax_hist.axvline(combined_pwm_err.mean(), color='r', linestyle='--',
                    label=f'Mean: {combined_pwm_err.mean():.2f}')
    ax_hist.set_title('PWM Error Distribution (mean across 3 channels)')
    ax_hist.set_xlabel('Error')
    ax_hist.set_ylabel('Count')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved: {output_path}")

    try:
        plt.show()
    except Exception:
        pass

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Position Error (cm): mean={pos_err.mean():.2f}, "
          f"median={np.median(pos_err):.2f}, max={pos_err.max():.2f}")
    for i in range(3):
        print(f"PWM{i+1} Error:        mean={pwm_err[i].mean():.2f}, "
              f"max={pwm_err[i].max():.2f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Diffusion Policy predictions')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to zarr dataset')
    parser.add_argument('--episode',      type=int, default=0,
                        help='Episode index to visualize')
    parser.add_argument('--device',       type=str, default='cpu',
                        help='Inference device (cpu/cuda)')
    parser.add_argument('--output',       type=str, default=None,
                        help='Output image path (default: visualization_ep{N}.png)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset not found: {args.dataset_path}")
        return 1

    import torch
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == 'cpu') else 'cpu')
    print(f"Loading policy (device={device}) ...")
    policy = DiffusionPolicyInference(args.checkpoint, device=str(device))
    print("✅ Policy loaded")

    data = run_episode_inference(policy, args.dataset_path, args.episode)

    output_path = args.output or f'visualization_ep{args.episode}.png'
    visualize(data, args.episode, output_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
