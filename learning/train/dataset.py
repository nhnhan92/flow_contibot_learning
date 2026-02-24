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
        - action:         (T, 6) - commanded UR5e target TCP pose (not used; we use future eef_pose)
        - camera_0:       (T, H, W, 3) - RGB images
        - timestamp:      (T,) - timestamps

    State  (9D): robot_eef_pose (6D) + pwm_signals (3D)
    Action (9D): future robot_eef_pose (6D) + future pwm_signals (3D)
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
        """
        print("Computing normalization statistics (Min-Max to [-1, 1])...")

        total_len = int(self.episode_ends[-1])
        num_samples = min(1000, total_len)
        sample_indices = np.random.choice(total_len, num_samples, replace=False)
        sample_indices = sorted(sample_indices)

        robot_states = []
        pwm_states = []
        for idx in sample_indices:
            robot_states.append(self.zarr_root['data/robot_eef_pose'][idx])
            pwm_states.append(self.zarr_root['data/pwm_signals'][idx])

        robot_states = np.array(robot_states)  # (N, 6)
        pwm_states = np.array(pwm_states)      # (N, 3)

        eps = 1e-6

        # State: robot_pose (6D) + pwm (3D) = 9D
        self.state_min = np.concatenate([robot_states.min(0), pwm_states.min(0)])
        self.state_max = np.concatenate([robot_states.max(0), pwm_states.max(0)])
        self.state_range = self.state_max - self.state_min + eps

        # Action uses the same structure (future robot_pose + future pwm)
        self.action_min = self.state_min.copy()
        self.action_max = self.state_max.copy()
        self.action_range = self.state_range.copy()

        print(f"  State/Action range (XYZ): "
              f"X=[{self.state_min[0]:.4f}, {self.state_max[0]:.4f}], "
              f"Y=[{self.state_min[1]:.4f}, {self.state_max[1]:.4f}], "
              f"Z=[{self.state_min[2]:.4f}, {self.state_max[2]:.4f}]")
        print(f"  PWM range: "
              f"[{self.state_min[6]:.1f}, {self.state_max[6]:.1f}], "
              f"[{self.state_min[7]:.1f}, {self.state_max[7]:.1f}], "
              f"[{self.state_min[8]:.1f}, {self.state_max[8]:.1f}]")

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

        # Combined state (obs_horizon, 9): robot_pose + pwm
        states = np.concatenate([robot_states, pwm_states], axis=-1)
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

        # Future actions: robot_eef_pose (6D) + pwm_signals (3D) = 9D
        action_start = sample_idx
        action_end = sample_idx + self.pred_horizon
        robot_actions = self.zarr_root['data/robot_eef_pose'][action_start:action_end]
        pwm_actions = self.zarr_root['data/pwm_signals'][action_start:action_end].astype(np.float32)

        actions = np.concatenate([robot_actions, pwm_actions], axis=-1)  # (pred_horizon, 9)
        actions = self._normalize_action(actions)

        return {
            'obs_state': torch.from_numpy(states).float(),    # (obs_horizon, 9)
            'obs_image': torch.from_numpy(images).float(),    # (obs_horizon, 3, H, W)
            'actions':   torch.from_numpy(actions).float(),   # (pred_horizon, 9)
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
        dataset_path='data/data_demo/dataset.zarr',
        use_images=True
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  obs_state shape: {sample['obs_state'].shape}")   # (2, 9)
    print(f"  obs_image shape: {sample['obs_image'].shape}")   # (2, 3, H, W)
    print(f"  actions shape:   {sample['actions'].shape}")     # (16, 9)

    print(f"\n  State (t):  pose={sample['obs_state'][-1, :6]}, pwm={sample['obs_state'][-1, 6:]}")
    print(f"  Action [0]: pose={sample['actions'][0, :6]},     pwm={sample['actions'][0, 6:]}")


if __name__ == '__main__':
    test_dataset()
