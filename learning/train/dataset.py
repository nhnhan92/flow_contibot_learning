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


class DiffusionDataset(Dataset):
    """
    Dataset for robot demonstrations with Flowbot soft manipulator.

    Data format from zarr (collected by demo_collect.py):
        - robot_eef_pose: (T, 6) - arm end-effector TCP pose [x, y, z, rx, ry, rz]
        - robot_joint:    (T, 6 or 7) - arm joint angles (not used for training)
        - pwm_signals:    (T, 3) - Flowbot PWM signals [pwm1, pwm2, pwm3]
        - action:         (T, 6) UR5e or (T, 7) Franka -- see `arm` below
        - camera_0:       (T, H, W, 3) - RGB images, global (scene) camera
        - camera_1:       (T, H, W, 3) - RGB images, wrist camera (optional,
                          only present in datasets collected with a wrist
                          camera connected -- see `camera_mode`)
        - timestamp:      (T,) - timestamps

    State  (tcp_dims+5 D):  robot_eef_pose[:tcp_dims] + pwm_signals (3D) + operation_mode (2D)
                             Always the Cartesian TCP pose, regardless of `arm` --
                             only the ACTION space differs per arm (below).

    Action, depends on `arm`:
        UR5e   (tcp_dims+5 D): target TCP[:tcp_dims] from data/action + pwm (3D) + op_mode (2D)
        Franka (7+5=12 D):     data/action in full (7D joint velocities, rad/s --
                                see demo_collect.py's _servo_toward docstring and
                                hardware/franka_robot.py's get_joint_velocities())
                                + pwm (3D) + op_mode (2D). tcp_dims does not apply
                                to a Franka checkpoint's action (only its state).

    tcp_dims controls how many TCP *state* components are used (both arms):
        tcp_dims=3  →  xyz only
        tcp_dims=6  →  xyz + rx,ry,rz
    Set via config key 'tcp_dims' (default: 3).

    operation_mode encoding per frame:
        [0, 0] = idle / holding
        [1, 0] = arm being controlled
        [0, 1] = flowbot being controlled
        [1, 1] = release phase

    camera_mode selects which camera(s) feed the model, matching model.py's
    num_cameras (1 = single vision encoder, 2 = two independent encoders):
        'global' (default): camera_0 only  -> sample['obs_image']
        'wrist':             camera_1 only  -> sample['obs_image'] (single
                              encoder still -- the pixels just come from the
                              wrist camera instead of the global one)
        'both':               camera_0      -> sample['obs_image']
                               camera_1      -> sample['obs_image_wrist']
    Set via config key 'camera_mode'.

    Using data/action (commanded target_pose for UR5e; executed joint velocity
    for Franka) rather than data/robot_eef_pose for action labels ensures
    action[0] != obs[-1]: the first predicted action is the command that moves
    the robot forward, not a copy of the current position/velocity.
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
        tcp_dims=3,         # TCP *state* components used: 3=xyz only, 6=xyz+rotation
        arm='ur5',          # 'ur5' (target TCP pose action) or 'franka' (7D joint velocity action)
        camera_mode='global',  # 'global' (camera_0 only), 'wrist' (camera_1 only), or 'both'
    ):
        self.dataset_path = Path(dataset_path)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_images = use_images
        self.normalize = normalize
        self.exclude_episodes = exclude_episodes if exclude_episodes is not None else []
        self.tcp_dims = tcp_dims
        self.arm = arm.lower()
        self.is_franka = self.arm == 'franka'
        # Franka's action is always 7D joint velocities (tcp_dims doesn't apply
        # to it); UR5e's action is the TCP pose, sliced like state is.
        self.action_dim_raw = 7 if self.is_franka else self.tcp_dims

        self.camera_mode = camera_mode.lower()
        if self.camera_mode not in ('global', 'wrist', 'both'):
            raise ValueError(f"camera_mode must be 'global', 'wrist', or 'both', got {camera_mode!r}")
        self.use_global_camera = self.camera_mode in ('global', 'both')
        self.use_wrist_camera  = self.camera_mode in ('wrist', 'both')
        self.num_cameras = int(self.use_global_camera) + int(self.use_wrist_camera)
        # Which raw camera key feeds sample['obs_image'] (the single/primary
        # vision encoder in model.py): camera_0 unless this is wrist-only,
        # in which case camera_1's pixels go through that same slot -- the
        # model doesn't care which physical camera a single-encoder path's
        # pixels came from. 'both' additionally routes camera_1 through
        # obs_image_wrist (second, independent encoder) -- see __getitem__.
        self._primary_camera_key = 'data/camera_0' if self.use_global_camera else 'data/camera_1'

        # Load zarr dataset
        self.zarr_root = zarr.open(str(self.dataset_path), mode='r')

        if self.use_global_camera and 'camera_0' not in self.zarr_root['data']:
            raise ValueError(
                f"camera_mode={self.camera_mode!r} needs data/camera_0, but "
                f"{self.dataset_path} has none."
            )
        if self.use_wrist_camera and 'camera_1' not in self.zarr_root['data']:
            raise ValueError(
                f"camera_mode={self.camera_mode!r} needs data/camera_1, but "
                f"{self.dataset_path} has none -- this dataset was collected "
                "without a wrist camera (or with --no_camera_wrist). Either "
                "recollect with the wrist camera connected, or use camera_mode='global'."
            )
        if self.is_franka:
            action_shape = self.zarr_root['data/action'].shape
            if action_shape[1] != 7:
                raise ValueError(
                    f"arm='franka' expects data/action to be 7D (joint velocities), "
                    f"but {self.dataset_path} has action shape {action_shape}. This "
                    f"dataset was likely collected with an older demo_collect.py "
                    f"that recorded Cartesian velocity (6D) as the Franka action -- "
                    f"recollect with the current demo_collect.py."
                )

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
        robot_actions = np.array(robot_actions)  # (N, 6) UR5e / (N, 7) Franka

        eps = 1e-6
        d = self.tcp_dims        # state TCP width: 3 or 6, both arms
        a = self.action_dim_raw  # action raw width: tcp_dims (UR5e) or 7 (Franka)

        # State: robot_eef_pose[:tcp_dims] + pwm (3D) -- Cartesian pose,
        # regardless of arm/action space.
        self.state_min = np.concatenate([robot_states[:, :d].min(0), pwm_states.min(0)])
        self.state_max = np.concatenate([robot_states[:, :d].max(0), pwm_states.max(0)])
        self.state_range = self.state_max - self.state_min + eps

        # Action: UR5e target_pose[:tcp_dims], Franka full 7D joint velocity + pwm (3D)
        self.action_min = np.concatenate([robot_actions[:, :a].min(0), pwm_states.min(0)])
        self.action_max = np.concatenate([robot_actions[:, :a].max(0), pwm_states.max(0)])
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

        tcp_labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'][:d]
        tcp_str = ', '.join(f"{l}=[{self.state_min[i]:.4f}, {self.state_max[i]:.4f}]"
                            for i, l in enumerate(tcp_labels))
        print(f"  State  range (TCP {d}D): {tcp_str}")

        if self.is_franka:
            joint_labels = [f'q{i+1}' for i in range(a)]
            action_str = ', '.join(f"{l}=[{self.action_min[i]:.4f}, {self.action_max[i]:.4f}]"
                                    for i, l in enumerate(joint_labels))
            print(f"  Action range (joint velocity {a}D, rad/s): {action_str}")
        else:
            action_str = ', '.join(f"{l}=[{self.action_min[i]:.4f}, {self.action_max[i]:.4f}]"
                                    for i, l in enumerate(tcp_labels))
            print(f"  Action range (TCP {a}D): {action_str}")

        print(f"  PWM range (state):  "
              f"[{self.state_min[d]:.1f}, {self.state_max[d]:.1f}], "
              f"[{self.state_min[d+1]:.1f}, {self.state_max[d+1]:.1f}], "
              f"[{self.state_min[d+2]:.1f}, {self.state_max[d+2]:.1f}]")
        print(f"  PWM range (action): "
              f"[{self.action_min[a]:.1f}, {self.action_max[a]:.1f}], "
              f"[{self.action_min[a+1]:.1f}, {self.action_max[a+1]:.1f}], "
              f"[{self.action_min[a+2]:.1f}, {self.action_max[a+2]:.1f}]")
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

        # Combined state (obs_horizon, tcp_dims+5): tcp[:tcp_dims] + pwm + op_mode
        states = np.concatenate([robot_states[:, :self.tcp_dims], pwm_states, op_mode_states], axis=-1)
        states = self._normalize_state(states)

        # Images
        if self.use_images:
            images = self._load_and_process_images(self._primary_camera_key, obs_start, obs_end)
        else:
            images = np.zeros((self.obs_horizon, 3, *self.image_size), dtype=np.float32)

        sample = {
            'obs_state': torch.from_numpy(states).float(),    # (obs_horizon, d+5)
            'obs_image': torch.from_numpy(images).float(),    # (obs_horizon, 3, H, W)
        }

        if self.camera_mode == 'both':
            if self.use_images:
                images_wrist = self._load_and_process_images('data/camera_1', obs_start, obs_end)
            else:
                images_wrist = np.zeros((self.obs_horizon, 3, *self.image_size), dtype=np.float32)
            sample['obs_image_wrist'] = torch.from_numpy(images_wrist).float()

        # Future actions: UR5e target TCP[:tcp_dims], Franka full 7D joint
        # velocity, + pwm (3D) + op_mode (2D). Using data/action (commanded
        # target_pose / executed joint velocity) instead of data/robot_eef_pose
        # so that action[0] != obs[-1]: it's always the command that moves the
        # robot forward, not a copy of the current position/velocity.
        action_start = sample_idx
        action_end = sample_idx + self.pred_horizon
        robot_actions  = self.zarr_root['data/action'][action_start:action_end]
        pwm_actions    = self.zarr_root['data/pwm_signals'][action_start:action_end].astype(np.float32)
        op_mode_actions = self.zarr_root['data/operation_mode'][action_start:action_end].astype(np.float32)

        actions = np.concatenate([robot_actions[:, :self.action_dim_raw], pwm_actions, op_mode_actions], axis=-1)
        actions = self._normalize_action(actions)

        sample['actions'] = torch.from_numpy(actions).float()   # (pred_horizon, action_dim_raw+5)
        return sample

    def _load_and_process_images(self, camera_key, obs_start, obs_end):
        """Center-crop + resize + normalize one camera's frames to (obs_horizon, C, H, W)."""
        images = self.zarr_root[camera_key][obs_start:obs_end]

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
        return images.transpose(0, 3, 1, 2)  # (obs_horizon, C, H, W)

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
    print(f"  obs_state shape: {sample['obs_state'].shape}")   # (2, 8)
    print(f"  obs_image shape: {sample['obs_image'].shape}")   # (2, 3, H, W)
    print(f"  actions shape:   {sample['actions'].shape}")     # (16, 8)

    d = dataset.tcp_dims
    print(f"\n  State (t):   tcp={sample['obs_state'][-1, :d]}, pwm={sample['obs_state'][-1, d:d+3]}, op_mode={sample['obs_state'][-1, d+3:]}")
    print(f"  Action [0]:  tcp={sample['actions'][0, :d]},    pwm={sample['actions'][0, d:d+3]},    op_mode={sample['actions'][0, d+3:]}")
    print(f"\n  Δtcp (action[0] - obs[-1]): {sample['actions'][0, :d] - sample['obs_state'][-1, :d]}")
    print(f"  (should be non-zero — action[0] is target_pose, not current position)")


if __name__ == '__main__':
    test_dataset()
