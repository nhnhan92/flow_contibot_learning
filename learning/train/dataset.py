"""
Dataset loader for Flowbot demonstrations
Loads data from zarr format collected by collect_demos_with_camera.py

Robot: UR5e + Flowbot soft manipulator (3 pneumatic valves via PWM)
"""

import numpy as np
import zarr
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class PickPlaceDataset(Dataset):
    """
    Dataset for robot demonstrations with Flowbot soft manipulator.

    Data format from zarr (collected by collect_demos_with_camera.py):
        - robot_eef_pose: (T, 6) - UR5e end-effector TCP pose [x, y, z, rx, ry, rz]
        - robot_joint:    (T, 6) - UR5e joint angles (not used for training)
        - pwm_signals:    (T, 3) - Flowbot PWM signals [pwm1, pwm2, pwm3]
        - action:         (T, 6) - commanded UR5e target TCP pose (spacemouse target)
        - camera_0:       (T, H, W, 3) - RGB images
        - timestamp:      (T,) - timestamps

    State  (11D): robot_eef_pose (6D) + pwm_signals (3D) + operation_mode (2D)
    Action (11D): target TCP pose from data/action (6D) + pwm_signals (3D) + operation_mode (2D)

    operation_mode encoding per frame:
        [0, 0] = idle / holding
        [1, 0] = UR5 being controlled
        [0, 1] = flowbot being controlled
        [1, 1] = release phase

    Using data/action (commanded target_pose) rather than data/robot_eef_pose for action
    labels ensures action[0] != obs[-1]: the first predicted action is the command that
    moves the robot forward, not a copy of the current position.
    """

    def __init__(
        self,
        dataset_path,
        obs_horizon=2,      # Number of observation frames
        pred_horizon=16,    # Number of action predictions
        action_horizon=8,   # Number of actions to execute
        image_size=(96, 96),  # Resize images to this size
        use_images=True,
        normalize=True,
        exclude_episodes=None,  # List of episode indices to exclude
    ):
        self.dataset_path = Path(dataset_path)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_images = use_images
        self.normalize = normalize
        self.exclude_episodes = exclude_episodes if exclude_episodes is not None else []

        # Load zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')

        # Get episode boundaries
        self.episode_ends = self.zarr_root['meta/episode_ends'][:]
        self.n_episodes = len(self.episode_ends)

        # Calculate valid samples (need enough frames for obs + pred)
        self.samples = []
        excluded_count = 0
        for ep_idx in range(self.n_episodes):
            # Skip excluded episodes
            if ep_idx in self.exclude_episodes:
                excluded_count += 1
                continue

            start_idx = 0 if ep_idx == 0 else int(self.episode_ends[ep_idx-1])
            end_idx = int(self.episode_ends[ep_idx])
            episode_length = end_idx - start_idx

            # Each sample needs obs_horizon past frames + pred_horizon future actions
            for i in range(episode_length):
                if i < obs_horizon - 1:
                    continue
                if i + pred_horizon > episode_length:
                    continue
                self.samples.append({
                    'episode_idx': ep_idx,
                    'start_idx': start_idx,
                    'sample_idx': start_idx + i
                })

        if excluded_count > 0:
            print(f"Loaded {self.n_episodes} episodes ({excluded_count} excluded), "
                  f"{len(self.samples)} samples")
        else:
            print(f"Loaded {self.n_episodes} episodes, {len(self.samples)} samples")

        # Compute normalization stats
        if self.normalize:
            self._compute_stats()

    def _compute_stats(self):
        """Compute min/max for normalization (Min-Max to [-1, 1]).

        x_norm = 2.0 * (x - min) / (max - min) - 1.0

        Uses ALL frames to guarantee correct min/max (no sampling bias).
        For large datasets (>10k frames) a seeded random sample is used
        to keep loading time reasonable while being fully reproducible.
        """
        print("Computing normalization statistics (Min-Max to [-1, 1])...")

        total_len = int(self.episode_ends[-1])
        FULL_SCAN_THRESHOLD = 10_000  # use all frames below this size

        if total_len <= FULL_SCAN_THRESHOLD:
            # Load everything — guaranteed correct min/max
            robot_states  = self.zarr_root['data/robot_eef_pose'][:]  # (T, 6)
            pwm_states    = self.zarr_root['data/pwm_signals'][:]     # (T, 3)
            robot_actions = self.zarr_root['data/action'][:]          # (T, 6) target_pose
            print(f"  Using all {total_len} frames for stats")
        else:
            # Seeded random sample — reproducible across runs
            rng = np.random.RandomState(42)
            sample_indices = sorted(rng.choice(total_len, 5000, replace=False))
            robot_states  = self.zarr_root['data/robot_eef_pose'].oindex[sample_indices]
            pwm_states    = self.zarr_root['data/pwm_signals'].oindex[sample_indices]
            robot_actions = self.zarr_root['data/action'].oindex[sample_indices]
            print(f"  Using 5000/{total_len} seeded-random frames for stats")

        robot_states  = np.array(robot_states)   # (N, 6)
        pwm_states    = np.array(pwm_states)     # (N, 3)
        robot_actions = np.array(robot_actions)  # (N, 6)

        eps = 1e-6

        # State: actual robot_eef_pose (6D) + pwm (3D) = 9D base
        self.state_min = np.concatenate([robot_states.min(0), pwm_states.min(0)])
        self.state_max = np.concatenate([robot_states.max(0), pwm_states.max(0)])
        self.state_range = self.state_max - self.state_min + eps

        # Action: commanded target_pose (6D) + pwm (3D) = 9D base  — stats from data/action
        self.action_min = np.concatenate([robot_actions.min(0), pwm_states.min(0)])
        self.action_max = np.concatenate([robot_actions.max(0), pwm_states.max(0)])
        self.action_range = self.action_max - self.action_min + eps

        # Append hardcoded stats for operation_mode (2D): always in {0, 1}
        # Hardcoded to avoid wrong range when dataset only has one mode
        op_min   = np.array([0.0, 0.0])
        op_max   = np.array([1.0, 1.0])
        op_range = np.array([1.0 + eps, 1.0 + eps])
        self.state_min   = np.concatenate([self.state_min,   op_min])
        self.state_max   = np.concatenate([self.state_max,   op_max])
        self.state_range = np.concatenate([self.state_range, op_range])
        self.action_min   = np.concatenate([self.action_min,   op_min])
        self.action_max   = np.concatenate([self.action_max,   op_max])
        self.action_range = np.concatenate([self.action_range, op_range])

        print(f"  State  range (XYZ): "
              f"X=[{self.state_min[0]:.4f}, {self.state_max[0]:.4f}], "
              f"Y=[{self.state_min[1]:.4f}, {self.state_max[1]:.4f}], "
              f"Z=[{self.state_min[2]:.4f}, {self.state_max[2]:.4f}]")
        print(f"  Action range (XYZ): "
              f"X=[{self.action_min[0]:.4f}, {self.action_max[0]:.4f}], "
              f"Y=[{self.action_min[1]:.4f}, {self.action_max[1]:.4f}], "
              f"Z=[{self.action_min[2]:.4f}, {self.action_max[2]:.4f}]")
        print(f"  PWM range: "
              f"[{self.state_min[6]:.1f}, {self.state_max[6]:.1f}], "
              f"[{self.state_min[7]:.1f}, {self.state_max[7]:.1f}], "
              f"[{self.state_min[8]:.1f}, {self.state_max[8]:.1f}]")
        print(f"  op_mode: hardcoded [0,0]→[-1,-1], [1,1]→[+1,+1]")

    def _normalize_state(self, state):
        """Normalize state using Min-Max to [-1, 1]"""
        if self.normalize:
            return 2.0 * (state - self.state_min) / self.state_range - 1.0
        return state

    def _normalize_action(self, action):
        """Normalize action using Min-Max to [-1, 1]"""
        if self.normalize:
            return 2.0 * (action - self.action_min) / self.action_range - 1.0
        return action

    def _denormalize_action(self, action):
        """Denormalize action from [-1, 1] to original range"""
        if self.normalize:
            return (action + 1.0) * 0.5 * self.action_range + self.action_min
        return action

    def _denormalize_state(self, state):
        """Denormalize state from [-1, 1] to original range"""
        if self.normalize:
            return (state + 1.0) * 0.5 * self.state_range + self.state_min
        return state

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample_idx = sample_info['sample_idx']

        # Observation window indices
        obs_start = sample_idx - (self.obs_horizon - 1)
        obs_end = sample_idx + 1

        # Robot TCP states (obs_horizon, 6)
        robot_states = self.zarr_root['data/robot_eef_pose'][obs_start:obs_end]

        # Flowbot PWM states (obs_horizon, 3)
        pwm_states = self.zarr_root['data/pwm_signals'][obs_start:obs_end].astype(np.float32)

        # Operation mode (obs_horizon, 2): [ur5_active, flowbot_active]
        op_mode_states = self.zarr_root['data/operation_mode'][obs_start:obs_end].astype(np.float32)

        # Combined state (obs_horizon, 11): robot_pose + pwm + op_mode
        states = np.concatenate([robot_states, pwm_states, op_mode_states], axis=-1)
        states = self._normalize_state(states)

        # Images
        if self.use_images:
            images = self.zarr_root['data/camera_0'][obs_start:obs_end]

            processed_images = []
            for img in images:
                h, w = img.shape[:2]
                target_h, target_w = self.image_size

                crop_h = min(h, int(target_h * 1.5))
                crop_w = min(w, int(target_w * 1.5))

                start_h = (h - crop_h) // 2
                start_w = (w - crop_w) // 2
                img_cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w]

                img_resized = cv2.resize(img_cropped, (target_w, target_h))
                img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
                processed_images.append(img_normalized)

            images = np.array(processed_images)
            images = images.transpose(0, 3, 1, 2)  # (obs_horizon, C, H, W)
        else:
            images = np.zeros((self.obs_horizon, 3, *self.image_size), dtype=np.float32)

        # Future actions: target_pose (6D) + pwm (3D) + op_mode (2D) = 11D
        # Using data/action (commanded target_pose) instead of data/robot_eef_pose so that
        # action[0] != obs[-1]: the spacemouse target is always ahead of the actual TCP.
        action_start = sample_idx
        action_end = sample_idx + self.pred_horizon
        robot_actions  = self.zarr_root['data/action'][action_start:action_end]
        pwm_actions    = self.zarr_root['data/pwm_signals'][action_start:action_end].astype(np.float32)
        op_mode_actions = self.zarr_root['data/operation_mode'][action_start:action_end].astype(np.float32)

        actions = np.concatenate([robot_actions, pwm_actions, op_mode_actions], axis=-1)  # (pred_horizon, 11)
        actions = self._normalize_action(actions)

        return {
            'obs_state': torch.from_numpy(states).float(),    # (obs_horizon, 11)
            'obs_image': torch.from_numpy(images).float(),    # (obs_horizon, 3, H, W)
            'actions':   torch.from_numpy(actions).float(),   # (pred_horizon, 11)
        }

    def get_normalizer(self):
        """Get action/state normalizer for inference"""
        return {
            'action_min':   self.action_min,
            'action_max':   self.action_max,
            'action_range': self.action_range,
            'state_min':    self.state_min,
            'state_max':    self.state_max,
            'state_range':  self.state_range,
        }


def test_dataset():
    """Test dataset loading"""
    dataset = PickPlaceDataset(
        dataset_path='/home/nhnhan/Desktop/flow_contibot_learning/data/demo_data/dataset.zarr',
        use_images=True
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  obs_state shape: {sample['obs_state'].shape}")   # (2, 11)
    print(f"  obs_image shape: {sample['obs_image'].shape}")   # (2, 3, H, W)
    print(f"  actions shape:   {sample['actions'].shape}")     # (16, 11)

    print(f"\n  State (t):   pose={sample['obs_state'][-1, :6]}, pwm={sample['obs_state'][-1, 6:9]}, op_mode={sample['obs_state'][-1, 9:]}")
    print(f"  Action [0]:  pose={sample['actions'][0, :6]},    pwm={sample['actions'][0, 6:9]},    op_mode={sample['actions'][0, 9:]}")
    print(f"\n  Δpose (action[0] - obs[-1]): {sample['actions'][0, :6] - sample['obs_state'][-1, :6]}")
    print(f"  (should be non-zero — action[0] is target_pose, not current position)")


if __name__ == '__main__':
    test_dataset()
