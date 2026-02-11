"""
Dataset loader for Pick-Place demonstrations
Loads data from zarr format collected by collect_demos_with_camera.py
"""

import numpy as np
import zarr
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class PickPlaceDataset(Dataset):
    """
    Dataset for pick-place demonstrations

    Data format from zarr:
        - robot_eef_pose: (T, 6) - end-effector pose [x, y, z, rx, ry, rz] (USED AS ACTIONS)
        - robot_joint: (T, 6) - joint angles
        - gripper_position: (T,) - gripper state [0, 1]
        - action: (T, 6) - commanded poses during collection (NOT USED - relative commands)
        - camera_0: (T, H, W, 3) - RGB images
        - timestamp: (T,) - timestamps

    NOTE: For Diffusion Policy, actions are ABSOLUTE TARGET POSES, not relative commands.
          We use future robot_eef_pose as action labels.
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
                # Can we get obs_horizon frames? (including current)
                if i < obs_horizon - 1:
                    continue

                # Can we get pred_horizon future actions?
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
        """Compute min/max for normalization (following original diffusion_policy)

        Uses Min-Max normalization to [-1, 1] range:
            x_norm = 2.0 * (x - min) / (max - min) - 1.0

        This is the standard approach in diffusion models, as it provides:
        - Bounded inputs/outputs in [-1, 1]
        - Better stability during training
        - Consistent with image normalization
        """
        print("Computing normalization statistics (Min-Max to [-1, 1])...")

        # Sample 1000 random timesteps
        total_len = int(self.episode_ends[-1])
        num_samples = min(1000, total_len)
        sample_indices = np.random.choice(total_len, num_samples, replace=False)
        sample_indices = sorted(sample_indices)  # Sort for better performance

        # Get robot states (zarr doesn't support fancy indexing, load individually)
        robot_states = []
        gripper_states = []
        actions = []
        for idx in sample_indices:
            robot_states.append(self.zarr_root['data/robot_eef_pose'][idx])
            gripper_states.append(self.zarr_root['data/gripper_position'][idx])
            actions.append(self.zarr_root['data/action'][idx])

        robot_states = np.array(robot_states)
        gripper_states = np.array(gripper_states)
        actions = np.array(actions)

        # Compute min/max for Min-Max normalization
        eps = 1e-6  # Small epsilon to avoid division by zero

        self.state_min = np.concatenate([robot_states.min(0), [gripper_states.min()]])
        self.state_max = np.concatenate([robot_states.max(0), [gripper_states.max()]])
        self.state_range = self.state_max - self.state_min + eps

        # Action includes robot pose (6D) + gripper (1D) = 7D
        self.action_min = np.concatenate([actions.min(0), [gripper_states.min()]])
        self.action_max = np.concatenate([actions.max(0), [gripper_states.max()]])
        self.action_range = self.action_max - self.action_min + eps

        print(f"  State range: [{self.state_min[0]:.4f}, {self.state_max[0]:.4f}] (X)")
        print(f"               [{self.state_min[1]:.4f}, {self.state_max[1]:.4f}] (Y)")
        print(f"               [{self.state_min[2]:.4f}, {self.state_max[2]:.4f}] (Z)")
        print(f"  Action range: [{self.action_min[0]:.4f}, {self.action_max[0]:.4f}] (X)")
        print(f"                [{self.action_min[1]:.4f}, {self.action_max[1]:.4f}] (Y)")
        print(f"                [{self.action_min[2]:.4f}, {self.action_max[2]:.4f}] (Z)")

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

        # Get observation window
        obs_start = sample_idx - (self.obs_horizon - 1)
        obs_end = sample_idx + 1

        # Robot states (obs_horizon, 6)
        robot_states = self.zarr_root['data/robot_eef_pose'][obs_start:obs_end]

        # Gripper states (obs_horizon,)
        gripper_states = self.zarr_root['data/gripper_position'][obs_start:obs_end]

        # Combine into state (obs_horizon, 7)
        states = np.concatenate([
            robot_states,
            gripper_states[:, None]
        ], axis=-1)

        # Normalize
        states = self._normalize_state(states)

        # Get images if using
        if self.use_images:
            # Load images (obs_horizon, H, W, 3)
            images = self.zarr_root['data/camera_0'][obs_start:obs_end]

            # Process images: center crop then resize
            processed_images = []
            for img in images:
                # Center crop (matches original diffusion_policy approach)
                # Original camera: 480x640 (H x W)
                # Target size: self.image_size (H, W)
                h, w = img.shape[:2]
                target_h, target_w = self.image_size

                # Calculate crop boundaries for center crop
                crop_h = min(h, int(target_h * 1.5))  # Allow some margin for cropping
                crop_w = min(w, int(target_w * 1.5))

                # Center crop
                start_h = (h - crop_h) // 2
                start_w = (w - crop_w) // 2
                img_cropped = img[start_h:start_h + crop_h, start_w:start_w + crop_w]

                # Resize to target size (cv2.resize takes (width, height))
                img_resized = cv2.resize(img_cropped, (target_w, target_h))

                # Normalize to [-1, 1]
                img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
                processed_images.append(img_normalized)

            images = np.array(processed_images)
            # Transpose to (obs_horizon, C, H, W) for PyTorch
            images = images.transpose(0, 3, 1, 2)
        else:
            images = np.zeros((self.obs_horizon, 3, *self.image_size), dtype=np.float32)

        # Get future actions (pred_horizon, 6) and gripper (pred_horizon,)
        # IMPORTANT: Actions should be target poses (absolute), not relative commands!
        action_start = sample_idx
        action_end = sample_idx + self.pred_horizon
        robot_actions = self.zarr_root['data/robot_eef_pose'][action_start:action_end]  # Use future poses as actions
        gripper_actions = self.zarr_root['data/gripper_position'][action_start:action_end]

        # Combine into actions (pred_horizon, 7): robot pose + gripper
        actions = np.concatenate([
            robot_actions,
            gripper_actions[:, None]
        ], axis=-1)

        # Normalize
        actions = self._normalize_action(actions)

        return {
            'obs_state': torch.from_numpy(states).float(),        # (obs_horizon, 7)
            'obs_image': torch.from_numpy(images).float(),        # (obs_horizon, 3, H, W)
            'actions': torch.from_numpy(actions).float(),         # (pred_horizon, 7)
        }

    def get_normalizer(self):
        """Get action normalizer for inference"""
        return {
            'action_min': self.action_min,
            'action_max': self.action_max,
            'action_range': self.action_range,
            'state_min': self.state_min,
            'state_max': self.state_max,
            'state_range': self.state_range,
        }


def test_dataset():
    """Test dataset loading"""
    dataset = PickPlaceDataset(
        dataset_path='/home/protac/Desktop/my_pickplace/data/camera_demos/dataset.zarr',
        use_images=True
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  obs_state shape: {sample['obs_state'].shape}")
    print(f"  obs_image shape: {sample['obs_image'].shape}")
    print(f"  action shape: {sample['action'].shape}")

    # Print some values
    print(f"\n  State (first timestep): {sample['obs_state'][0]}")
    print(f"  Image (first pixel): {sample['obs_image'][0, :, 0, 0]}")
    print(f"  Action (first step): {sample['action'][0]}")


if __name__ == '__main__':
    test_dataset()
