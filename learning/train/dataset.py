"""
Dataset loader for Flowbot demonstrations
Loads data from zarr format collected by collect_demos_with_camera.py

Robot: UR5e + Flowbot soft manipulator (3 pneumatic valves via PWM)
"""

import os
import sys
import numpy as np
import zarr
import torch
from torch.utils.data import Dataset
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware.image_utils import crop_and_resize


# state_keys component name -> zarr dataset key. 'tcp' additionally gets
# sliced to :tcp_dims (its width varies with tcp_dims); the others are used
# in full at their fixed width below. See DiffusionDataset's docstring.
STATE_COMPONENT_ZARR_KEY = {
    'tcp':      'data/robot_eef_pose',
    'pwm':      'data/pwm_signals',
    'flowrate': 'data/flowrate',
    'op_mode':  'data/operation_mode',
}
STATE_COMPONENT_WIDTH = {'pwm': 3, 'flowrate': 3, 'op_mode': 2}  # 'tcp' width = tcp_dims
# Components whose normalization range is fixed, not derived from the data:
# op_mode is always exactly {0, 1} per dim -- deriving min/max from data
# would risk a degenerate (zero-width) range if a given dataset only ever
# visited one op_mode value.
STATE_COMPONENT_HARDCODED_STATS = {'op_mode'}


class DiffusionDataset(Dataset):
    """
    Dataset for robot demonstrations with Flowbot soft manipulator.

    Data format from zarr (collected by demo_collect.py):
        - robot_eef_pose: (T, 6) - arm end-effector TCP pose [x, y, z, rx, ry, rz]
        - robot_joint:    (T, 6 or 7) - arm joint angles (not used for training)
        - pwm_signals:    (T, 3) - Flowbot PWM signals [pwm1, pwm2, pwm3]
        - flowrate:       (T, 3) - Flowbot flow-sensor readings, L/min per
                          actuator (optional -- only present in datasets
                          collected after flowrate recording was added; see
                          state_keys below)
        - action:         (T, 6) UR5e or (T, 7) Franka -- see `arm` below
        - camera_0:       (T, H, W, 3) - RGB images, global (scene) camera
        - camera_1:       (T, H, W, 3) - RGB images, wrist camera (optional,
                          only present in datasets collected with a wrist
                          camera connected -- see `camera_mode`)
        - timestamp:      (T,) - timestamps

    State: freely composable from state_keys, a list of component names in
    the order they're concatenated. Available components:
        'tcp'      : robot_eef_pose[:tcp_dims]  (width = tcp_dims)
        'pwm'      : pwm_signals                (width 3)
        'flowrate' : flowrate                   (width 3) -- raises a clear
                     error at load time if this dataset predates flowrate
                     recording and doesn't have data/flowrate
        'op_mode'  : operation_mode              (width 2) -- always a fixed
                     {0,1} range (see _compute_stats), never data-derived
    Default state_keys=('tcp', 'pwm', 'op_mode') reproduces the original
    fixed tcp_dims+5 composition exactly. Examples from the class's actual
    use: state_keys=('tcp', 'flowrate', 'op_mode') swaps pwm for flowrate
    (tcp_dims+5 D); state_keys=('tcp', 'pwm', 'flowrate', 'op_mode') uses
    all four (tcp_dims+8 D). Set via config key 'state_keys'. Always the
    Cartesian TCP pose for 'tcp', regardless of `arm` -- only the ACTION
    space differs per arm (below), which is NOT affected by state_keys.

    Action, depends on `arm` and (Franka only) `franka_action_space`:
        UR5e             (tcp_dims+5 D): target TCP[:tcp_dims] from data/action
                                + pwm (3D) + op_mode (2D)
        Franka
          'joint_velocity' (7+5=12 D):   data/action in full (7D joint velocities,
                                rad/s -- see demo_collect.py's _servo_toward
                                docstring and hardware/franka_robot.py's
                                get_joint_velocities()) + pwm (3D) + op_mode (2D).
                                tcp_dims does not apply to this action (only state).
          'position'      (tcp_dims+5 D): TCP[:tcp_dims] one step ahead, read
                                from data/robot_eef_pose (NOT data/action --
                                demo_collect.py drives Franka via velocity control,
                                so there is no commanded position target to record;
                                this uses the actual measured next position
                                instead) + pwm (3D) + op_mode (2D). Same action
                                shape/meaning as UR5e's -- lets a Franka checkpoint
                                reuse UR5e's TCP-position execution path at deploy
                                time (see deploy_flowbot_w_policy.py).
        Set via config key 'franka_action_space' (default 'joint_velocity').
        A checkpoint's action space is fixed at training time -- it doesn't
        change what demo_collect.py needs to record; both are always available
        in any Franka dataset, so you can retrain with the other choice later
        without recollecting.

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
        image_size=(240, 320),  # Resize images to this size
        use_images=True,
        cache_images=True,  # Precompute+cache cropped/resized frames in RAM once
                             # (__init__), instead of re-decoding+resizing from zarr
                             # on every __getitem__ call of every epoch. Cropping is
                             # deterministic (fixed crop_scale/crop_x/crop_y -- the
                             # actual random augmentation happens later, in model.py,
                             # on already-loaded GPU tensors), so the cached result is
                             # valid for the whole training run. Set False only if the
                             # dataset is too large to fit the cache in RAM (see the
                             # printed size estimate at construction time).
        normalize=True,
        exclude_episodes=None,  # List of episode indices to exclude
        tcp_dims=3,         # TCP components used: 3=xyz only, 6=xyz+rotation
        state_keys=('tcp', 'pwm', 'op_mode'),  # Which components make up the
                                # state vector, in order -- any subset/order
                                # of 'tcp', 'pwm', 'flowrate', 'op_mode'. See
                                # class docstring. Default reproduces the
                                # original fixed composition exactly.
        crop_scale=1.5,     # Crop window size as a multiple of image_size
        crop_x=0.5,         # Crop anchor in [0,1]: 0=left, 0.5=center, 1=right
        crop_y=0.5,         # Crop anchor in [0,1]: 0=top, 0.5=center, 1=bottom
        wrist_image_size=None,  # Wrist camera override; None -> same as image_size
        wrist_crop_scale=None,  # Wrist camera override; None -> same as crop_scale
        wrist_crop_x=None,      # Wrist camera override; None -> same as crop_x
        wrist_crop_y=None,      # Wrist camera override; None -> same as crop_y
        arm='ur5',          # 'ur5' (target TCP pose action) or 'franka' (see franka_action_space)
        camera_mode='global',  # 'global' (camera_0 only), 'wrist' (camera_1 only), or 'both'
        franka_action_space='joint_velocity',  # Franka only: 'joint_velocity' (7D, data/action) or
                                                # 'position' (TCP[:tcp_dims], data/robot_eef_pose)
    ):
        self.dataset_path = Path(dataset_path)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.use_images = use_images
        self.cache_images = cache_images
        self.normalize = normalize
        self.exclude_episodes = exclude_episodes if exclude_episodes is not None else []
        self.tcp_dims = tcp_dims
        self.state_keys = list(state_keys)
        _unknown = set(self.state_keys) - set(STATE_COMPONENT_ZARR_KEY)
        if _unknown:
            raise ValueError(
                f"Unknown state_keys entry/entries {sorted(_unknown)}. "
                f"Valid: {sorted(STATE_COMPONENT_ZARR_KEY)}."
            )
        self.state_dim = sum(
            self.tcp_dims if key == 'tcp' else STATE_COMPONENT_WIDTH[key]
            for key in self.state_keys
        )
        self.crop_scale = crop_scale
        self.crop_x = crop_x
        self.crop_y = crop_y
        # Wrist camera can have its own image_size/crop settings (independent
        # vision encoder in model.py -- see SpatialSoftmax/AvgPool, both
        # resolution-agnostic, so the two cameras need not match). Any
        # wrist_* left as None falls back to the global-camera value above.
        self.wrist_image_size = tuple(wrist_image_size) if wrist_image_size is not None else self.image_size
        self.wrist_crop_scale = wrist_crop_scale if wrist_crop_scale is not None else self.crop_scale
        self.wrist_crop_x     = wrist_crop_x     if wrist_crop_x     is not None else self.crop_x
        self.wrist_crop_y     = wrist_crop_y     if wrist_crop_y     is not None else self.crop_y
        self.arm = arm.lower()
        self.is_franka = self.arm == 'franka'
        self.franka_action_space = franka_action_space.lower() if self.is_franka else None
        if self.is_franka and self.franka_action_space not in ('joint_velocity', 'position'):
            raise ValueError(
                f"franka_action_space must be 'joint_velocity' or 'position', "
                f"got {franka_action_space!r}"
            )
        # True iff this checkpoint's action is TCP position -- true for UR5e
        # always, and for Franka only when explicitly selected. Determines
        # both the action width/source below and how deploy executes it.
        self.uses_position_action = (not self.is_franka) or self.franka_action_space == 'position'
        # Franka joint_velocity: always 7D (tcp_dims doesn't apply to it, only
        # to state). Otherwise (UR5e, or Franka position mode): TCP[:tcp_dims],
        # sliced the same as state.
        self.action_dim_raw = self.tcp_dims if self.uses_position_action else 7
        # data/action for joint_velocity (a genuine commanded/executed value,
        # see demo_collect.py); data/robot_eef_pose for Franka position mode --
        # there's no commanded position target to record under velocity-control
        # teleop, so this uses the actual measured next position instead (see
        # class docstring). __getitem__ applies an extra +1 shift for this
        # source specifically, since (unlike data/action) it's recorded at the
        # SAME index as the state it would otherwise trivially duplicate.
        self._action_source_key = (
            'data/robot_eef_pose' if (self.is_franka and self.franka_action_space == 'position')
            else 'data/action'
        )
        self._action_index_shift = 1 if self._action_source_key == 'data/robot_eef_pose' else 0

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
        # Crop/resize settings for the primary slot: global's when the primary
        # is camera_0, wrist's when this is wrist-only (primary is camera_1).
        if self.use_global_camera:
            self._primary_image_size = self.image_size
            self._primary_crop_scale = self.crop_scale
            self._primary_crop_x     = self.crop_x
            self._primary_crop_y     = self.crop_y
        else:
            self._primary_image_size = self.wrist_image_size
            self._primary_crop_scale = self.wrist_crop_scale
            self._primary_crop_x     = self.wrist_crop_x
            self._primary_crop_y     = self.wrist_crop_y

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
        for key in self.state_keys:
            zarr_key = STATE_COMPONENT_ZARR_KEY[key]
            if zarr_key.split('/', 1)[1] not in self.zarr_root['data']:
                raise ValueError(
                    f"state_keys includes {key!r}, but {self.dataset_path} has no "
                    f"{zarr_key} -- this dataset was collected with an older "
                    f"demo_collect.py that didn't record it. Recollect with the "
                    f"current demo_collect.py, or remove {key!r} from state_keys."
                )
        if self.is_franka and self.franka_action_space == 'joint_velocity':
            action_shape = self.zarr_root['data/action'].shape
            if action_shape[1] != 7:
                raise ValueError(
                    f"arm='franka', franka_action_space='joint_velocity' expects data/action "
                    f"to be 7D (joint velocities), but {self.dataset_path} has action shape "
                    f"{action_shape}. This dataset was likely collected with an older "
                    f"demo_collect.py that recorded Cartesian velocity (6D) as the Franka "
                    f"action -- recollect with the current demo_collect.py."
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
            # (+ one more frame when the action source is index-shifted, i.e.
            # Franka position mode reading data/robot_eef_pose -- see
            # self._action_index_shift).
            for i in range(episode_length):
                if i < obs_horizon - 1:
                    continue
                if i + pred_horizon + self._action_index_shift > episode_length:
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

        # Precompute cropped/resized frames once (see cache_images above) --
        # keyed by camera_key since 'both' mode caches camera_0 and camera_1
        # independently, each under its own crop settings.
        self._image_caches = {}
        if self.use_images and self.cache_images:
            self._image_caches[self._primary_camera_key] = self._build_image_cache(
                self._primary_camera_key, self._primary_image_size,
                self._primary_crop_scale, self._primary_crop_x, self._primary_crop_y,
            )
            if self.camera_mode == 'both':
                self._image_caches['data/camera_1'] = self._build_image_cache(
                    'data/camera_1', self.wrist_image_size,
                    self.wrist_crop_scale, self.wrist_crop_x, self.wrist_crop_y,
                )

    def _build_image_cache(self, camera_key, image_size, crop_scale, crop_x, crop_y):
        """Decode + crop + resize every frame of `camera_key` once, up front,
        and hold the result (uint8, pre-normalization) in RAM. See
        cache_images's docstring above for why this is safe (deterministic
        crop) and worthwhile (eliminates ~num_epochs redundant re-decodes).
        """
        total_len = int(self.episode_ends[-1])
        target_h, target_w = image_size
        size_mb = total_len * target_h * target_w * 3 / (1024 ** 2)
        print(f"Caching {camera_key} in RAM: {total_len} frames @ {target_h}x{target_w} "
              f"(~{size_mb:.0f} MB)...")

        cache = np.empty((total_len, target_h, target_w, 3), dtype=np.uint8)
        zarr_arr = self.zarr_root[camera_key]
        BATCH = 256
        for start in range(0, total_len, BATCH):
            end = min(start + BATCH, total_len)
            raw_batch = zarr_arr[start:end]  # decodes this batch's chunks from disk
            for i, img in enumerate(raw_batch):
                cache[start + i] = crop_and_resize(img, image_size, crop_scale=crop_scale,
                                                    crop_x=crop_x, crop_y=crop_y)
            if (start // BATCH) % 8 == 0 or end == total_len:
                print(f"  {end}/{total_len}", end='\r', flush=True)
        print(f"  {total_len}/{total_len} cached.")
        return cache

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
        needs_flowrate = 'flowrate' in self.state_keys

        if total_len <= FULL_SCAN_THRESHOLD:
            # Load everything — guaranteed correct min/max
            robot_states  = self.zarr_root['data/robot_eef_pose'][:]  # (T, 6)
            pwm_states    = self.zarr_root['data/pwm_signals'][:]     # (T, 3)
            robot_actions = self.zarr_root[self._action_source_key][:]  # (T, 6 or 7)
            flowrate_states = self.zarr_root['data/flowrate'][:] if needs_flowrate else None
            print(f"  Using all {total_len} frames for stats")
        else:
            # Seeded random sample — reproducible across runs
            rng = np.random.RandomState(42)
            sample_indices = sorted(rng.choice(total_len, 5000, replace=False))
            robot_states  = self.zarr_root['data/robot_eef_pose'].oindex[sample_indices]
            pwm_states    = self.zarr_root['data/pwm_signals'].oindex[sample_indices]
            robot_actions = self.zarr_root[self._action_source_key].oindex[sample_indices]
            flowrate_states = self.zarr_root['data/flowrate'].oindex[sample_indices] if needs_flowrate else None
            print(f"  Using 5000/{total_len} seeded-random frames for stats")

        robot_states  = np.array(robot_states)   # (N, 6)
        pwm_states    = np.array(pwm_states)     # (N, 3)
        robot_actions = np.array(robot_actions)  # (N, tcp_dims-compatible 6) or (N, 7) joint velocity
        if needs_flowrate:
            flowrate_states = np.array(flowrate_states)  # (N, 3)

        eps = 1e-6
        d = self.tcp_dims        # state TCP width: 3 or 6, both arms
        a = self.action_dim_raw  # action raw width: tcp_dims (UR5e) or 7 (Franka)

        # State: generic composition from self.state_keys, in order (see
        # class docstring). Reproduces the original fixed
        # tcp+pwm+op_mode computation exactly when state_keys is left default.
        _state_source = {'tcp': robot_states[:, :d], 'pwm': pwm_states, 'flowrate': flowrate_states}
        state_mins, state_maxs = [], []
        for key in self.state_keys:
            if key in STATE_COMPONENT_HARDCODED_STATS:  # op_mode: always {0,1}, never data-derived
                width = STATE_COMPONENT_WIDTH[key]
                state_mins.append(np.zeros(width))
                state_maxs.append(np.ones(width))
            else:
                comp = _state_source[key]
                state_mins.append(comp.min(0))
                state_maxs.append(comp.max(0))
        self.state_min = np.concatenate(state_mins)
        self.state_max = np.concatenate(state_maxs)
        self.state_range = self.state_max - self.state_min + eps

        # Action: UR5e target_pose[:tcp_dims], Franka full 7D joint velocity
        # + pwm (3D) + op_mode (2D) -- always this fixed composition,
        # regardless of state_keys (state_keys only affects the STATE above).
        self.action_min = np.concatenate([robot_actions[:, :a].min(0), pwm_states.min(0)])
        self.action_max = np.concatenate([robot_actions[:, :a].max(0), pwm_states.max(0)])
        self.action_range = self.action_max - self.action_min + eps

        # Append hardcoded stats for operation_mode (2D): always in {0, 1}
        # Hardcoded to avoid wrong range when dataset only has one mode
        op_min   = np.array([0.0, 0.0])
        op_max   = np.array([1.0, 1.0])
        op_range = np.array([1.0 + eps, 1.0 + eps])
        self.action_min   = np.concatenate([self.action_min,   op_min])
        self.action_max   = np.concatenate([self.action_max,   op_max])
        self.action_range = np.concatenate([self.action_range, op_range])

        print(f"  State composition {self.state_keys} ({self.state_dim}D):")
        offset = 0
        for key in self.state_keys:
            width = d if key == 'tcp' else STATE_COMPONENT_WIDTH[key]
            rng_str = ', '.join(
                f"[{self.state_min[offset+i]:.4f}, {self.state_max[offset+i]:.4f}]"
                for i in range(width)
            )
            tag = " (hardcoded)" if key in STATE_COMPONENT_HARDCODED_STATS else ""
            print(f"    {key}: {rng_str}{tag}")
            offset += width

        tcp_labels = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'][:d]
        if self.is_franka and self.franka_action_space == 'joint_velocity':
            joint_labels = [f'q{i+1}' for i in range(a)]
            action_str = ', '.join(f"{l}=[{self.action_min[i]:.4f}, {self.action_max[i]:.4f}]"
                                    for i, l in enumerate(joint_labels))
            print(f"  Action range (joint velocity {a}D, rad/s): {action_str}")
        else:
            action_str = ', '.join(f"{l}=[{self.action_min[i]:.4f}, {self.action_max[i]:.4f}]"
                                    for i, l in enumerate(tcp_labels))
            print(f"  Action range (TCP {a}D): {action_str}")

        print(f"  PWM range (action): "
              f"[{self.action_min[a]:.1f}, {self.action_max[a]:.1f}], "
              f"[{self.action_min[a+1]:.1f}, {self.action_max[a+1]:.1f}], "
              f"[{self.action_min[a+2]:.1f}, {self.action_max[a+2]:.1f}]")
        print(f"  op_mode (action): hardcoded [0,0]→[-1,-1], [1,1]→[+1,+1]")

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

        # Read only the raw zarr components self.state_keys actually needs
        # (obs_horizon, width) each -- see class docstring / state_keys.
        _state_raw = {}
        if 'tcp' in self.state_keys:
            _state_raw['tcp'] = self.zarr_root['data/robot_eef_pose'][obs_start:obs_end][:, :self.tcp_dims] \
                .astype(np.float32)
        if 'pwm' in self.state_keys:
            _state_raw['pwm'] = self.zarr_root['data/pwm_signals'][obs_start:obs_end].astype(np.float32)
        if 'flowrate' in self.state_keys:
            _state_raw['flowrate'] = self.zarr_root['data/flowrate'][obs_start:obs_end].astype(np.float32)
        if 'op_mode' in self.state_keys:  # [ur5_active, flowbot_active]
            _state_raw['op_mode'] = self.zarr_root['data/operation_mode'][obs_start:obs_end].astype(np.float32)

        # Combined state (obs_horizon, state_dim): components in self.state_keys order.
        states = np.concatenate([_state_raw[key] for key in self.state_keys], axis=-1)
        states = self._normalize_state(states)

        # Images
        if self.use_images:
            images = self._load_and_process_images(
                self._primary_camera_key, obs_start, obs_end,
                self._primary_image_size, self._primary_crop_scale,
                self._primary_crop_x, self._primary_crop_y,
            )
        else:
            images = np.zeros((self.obs_horizon, 3, *self._primary_image_size), dtype=np.float32)

        sample = {
            'obs_state': torch.from_numpy(states).float(),    # (obs_horizon, state_dim)
            'obs_image': torch.from_numpy(images).float(),    # (obs_horizon, 3, H, W)
        }

        if self.camera_mode == 'both':
            if self.use_images:
                images_wrist = self._load_and_process_images(
                    'data/camera_1', obs_start, obs_end,
                    self.wrist_image_size, self.wrist_crop_scale,
                    self.wrist_crop_x, self.wrist_crop_y,
                )
            else:
                images_wrist = np.zeros((self.obs_horizon, 3, *self.wrist_image_size), dtype=np.float32)
            sample['obs_image_wrist'] = torch.from_numpy(images_wrist).float()

        # Future actions: TCP[:tcp_dims] (UR5e, or Franka position mode) or
        # 7D joint velocity (Franka joint_velocity mode), + pwm (3D) +
        # op_mode (2D). self._action_source_key is data/action (a genuine
        # commanded/executed value) unless this is Franka position mode, in
        # which case it's data/robot_eef_pose shifted one extra step ahead
        # (self._action_index_shift) so action[0] != obs[-1] -- see
        # self._action_source_key's assignment above and the class docstring.
        action_start = sample_idx
        action_end = sample_idx + self.pred_horizon
        arm_action_start = action_start + self._action_index_shift
        arm_action_end = action_end + self._action_index_shift
        robot_actions  = self.zarr_root[self._action_source_key][arm_action_start:arm_action_end]
        pwm_actions    = self.zarr_root['data/pwm_signals'][action_start:action_end].astype(np.float32)
        op_mode_actions = self.zarr_root['data/operation_mode'][action_start:action_end].astype(np.float32)

        actions = np.concatenate([robot_actions[:, :self.action_dim_raw], pwm_actions, op_mode_actions], axis=-1)
        actions = self._normalize_action(actions)

        sample['actions'] = torch.from_numpy(actions).float()   # (pred_horizon, action_dim_raw+5)
        return sample

    def _load_and_process_images(self, camera_key, obs_start, obs_end,
                                  image_size, crop_scale, crop_x, crop_y):
        """Crop + resize + normalize one camera's frames to (obs_horizon, C, H, W).

        Uses the precomputed RAM cache (see _build_image_cache) when
        available -- only decodes/crops/resizes from zarr on a cache miss
        (cache_images=False, e.g. a dataset too large to fit in RAM).
        """
        cache = self._image_caches.get(camera_key)
        if cache is not None:
            images_uint8 = cache[obs_start:obs_end]
        else:
            raw = self.zarr_root[camera_key][obs_start:obs_end]
            images_uint8 = np.array([
                crop_and_resize(img, image_size, crop_scale=crop_scale, crop_x=crop_x, crop_y=crop_y)
                for img in raw
            ])

        images = (images_uint8.astype(np.float32) / 127.5) - 1.0
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
