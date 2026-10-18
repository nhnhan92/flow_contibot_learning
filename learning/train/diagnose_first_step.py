#!/usr/bin/env python3
"""
diagnose_first_step.py — offline check of what a trained policy does at the
very FIRST step of each episode, to debug "the flowbot moves first instead of
the UR5" without touching the robot.

For every episode in the dataset, the policy is fed that episode's starting
observation and sampled --n_samples times (diffusion sampling is stochastic,
so one sample proves nothing). The op_mode of the first executed action
(actions[0], same as deploy_flowbot_w_policy.py's execution loop) is counted:
UR5 [1,0], FB [0,1], idle [0,0], release [1,1].

Check 1 — replay start frames:
    Two observation variants per episode:
      data   : the dataset's own first observation window, exactly as in training.
      deploy : what deployment actually feeds at step 0 -- the first frame
               repeated obs_horizon times (_fill_obs_buffer), with the pwm and
               op_mode state components reset to 0 (run_episode resets
               prev_pwm / current_op_mode before filling the buffer).
    FB on 'data' too  -> the model itself is wrong on its own training starts.
    FB on 'deploy' only -> train/deploy mismatch at step 0 (e.g. the recorded
                           frame-0 op_mode/pwm state isn't [0,0]/0 in the data).

Check 2 — EMA swap:
    Two weight variants per checkpoint:
      ema  : EMA UNet + live encoders -- exactly what eval.py/deployment loads.
      live : the live (non-EMA) weights for the whole model.
    FB with ema but not live -> EMA-UNet / live-encoder mismatch confirmed.

Also printed per object: the ground-truth op_mode of the first recorded action,
and the RECORDED frame-0 op_mode state -- if that isn't idle [0,0] in the
data, deployment's step-0 reset to [0,0] is itself out of distribution.

Usage:
    python learning/train/diagnose_first_step.py \\
        --checkpoints path/to/proj/best_model.pt path/to/noproj/best_model.pt \\
        --dataset data/combined/dataset.zarr \\
        --n_samples 20 --out_csv first_step_report.csv

    # only one object, fewer episodes (faster)
    python learning/train/diagnose_first_step.py --checkpoints ckpt.pt \\
        --dataset data/combined/dataset.zarr --object red2 --max_episodes_per_object 10
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.eval import DiffusionPolicyInference
from train.dataset import DiffusionDataset

MODES = ['UR5', 'FB', 'idle', 'release']
_OP_MODE_NAMES = {(0, 0): 'idle', (0, 1): 'FB', (1, 0): 'UR5', (1, 1): 'release'}


def op_mode_name(pair) -> str:
    return _OP_MODE_NAMES[(int(pair[0]), int(pair[1]))]


def build_dataset(policy, dataset_path):
    """Dataset built with the checkpoint's own config, normalized with the
    checkpoint's own stats (not stats recomputed from this dataset)."""
    cfg = policy.config
    ds = DiffusionDataset(
        dataset_path=dataset_path,
        obs_horizon=cfg['obs_horizon'],
        pred_horizon=cfg['pred_horizon'],
        action_horizon=cfg['action_horizon'],
        image_size=tuple(cfg['image_size']),
        cache_images=False,   # only the start frame of each episode is read
        exclude_episodes=cfg.get('exclude_episodes', []),
        tcp_dims=cfg.get('tcp_dims', 3),
        state_keys=cfg.get('state_keys', ('tcp', 'pwm', 'op_mode')),
        crop_scale=cfg.get('crop_scale', 1.5),
        crop_x=cfg.get('crop_x', 0.5),
        crop_y=cfg.get('crop_y', 0.5),
        wrist_image_size=cfg.get('wrist_image_size', None),
        wrist_crop_scale=cfg.get('wrist_crop_scale', None),
        wrist_crop_x=cfg.get('wrist_crop_x', None),
        wrist_crop_y=cfg.get('wrist_crop_y', None),
        arm=cfg.get('arm', 'ur5'),
        camera_mode=cfg.get('camera_mode', 'global'),
        franka_action_space=cfg.get('franka_action_space', 'joint_velocity'),
    )
    ckpt = policy.checkpoint
    ds.state_min, ds.state_max, ds.state_range = ckpt['state_min'], ckpt['state_max'], ckpt['state_range']
    return ds


def episode_start_indices(ds):
    """episode_idx -> dataset index of that episode's first sample."""
    starts = {}
    for i, s in enumerate(ds.samples):
        ep = s['episode_idx']
        if ep not in starts or s['sample_idx'] < ds.samples[starts[ep]]['sample_idx']:
            starts[ep] = i
    return starts


def episode_object_names(ds):
    """episode_idx -> object name, read straight from the zarr meta (works even
    if this checkpoint doesn't use 'object_id' in state_keys)."""
    meta = ds.zarr_root['meta']
    if 'episode_object_id' not in meta or 'object_names' not in meta:
        return defaultdict(lambda: 'all')
    ids = np.asarray(meta['episode_object_id'][:], dtype=np.int64)
    names = [str(n) for n in meta['object_names'][:]]
    return {ep: names[ids[ep]] for ep in range(len(ids))}


def state_offsets(ds):
    """state_keys component -> (start, end) slice within the state vector."""
    out, off = {}, 0
    for key in ds.state_keys:
        w = ds._component_width(key)
        out[key] = (off, off + w)
        off += w
    return out


def make_obs(ds, sample, variant, offsets):
    """Returns (obs_state, obs_image, obs_image_wrist) tensors, (obs_horizon, ...)."""
    obs_state = sample['obs_state'].clone()
    obs_image = sample['obs_image'].clone()
    obs_image_wrist = sample['obs_image_wrist'].clone() if 'obs_image_wrist' in sample else None
    if variant == 'deploy':
        T = obs_state.shape[0]
        obs_state = obs_state[:1].repeat(T, 1)
        obs_image = obs_image[:1].repeat(T, 1, 1, 1)
        if obs_image_wrist is not None:
            obs_image_wrist = obs_image_wrist[:1].repeat(T, 1, 1, 1)
        # Reset pwm/op_mode to 0, as run_episode does before _fill_obs_buffer.
        raw = ds._denormalize_state(obs_state.numpy())
        for key in ('pwm', 'op_mode'):
            if key in offsets:
                a, b = offsets[key]
                raw[:, a:b] = 0.0
        obs_state = torch.from_numpy(ds._normalize_state(raw)).float()
    return obs_state, obs_image, obs_image_wrist


@torch.no_grad()
def first_step_modes(policy, obs, n_samples, action_horizon):
    """Sample n_samples action chunks; return (first executed op_mode names,
    whether any FB appears within the executed action_horizon steps)."""
    obs_state, obs_image, obs_image_wrist = obs
    batch = lambda t: None if t is None else t.unsqueeze(0).repeat(n_samples, *([1] * t.ndim))
    actions_norm = policy.predict(batch(obs_state), batch(obs_image), batch(obs_image_wrist)).numpy()
    ckpt = policy.checkpoint
    actions = (actions_norm + 1.0) * 0.5 * ckpt['action_range'] + ckpt['action_min']
    op = np.clip(np.round(actions[:, :, -2:]), 0, 1).astype(int)    # op_mode = last 2 dims
    first = [op_mode_name(op[i, 0]) for i in range(n_samples)]
    any_fb = [bool(np.any((op[i, :action_horizon, 0] == 0) & (op[i, :action_horizon, 1] == 1)))
              for i in range(n_samples)]
    return first, any_fb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', required=True, help='One or more policy checkpoints')
    parser.add_argument('--dataset', required=True, help='The (labeled) dataset.zarr the policies were trained on')
    parser.add_argument('--n_samples', type=int, default=20, help='Diffusion samples per episode start (default 20)')
    parser.add_argument('--object', type=str, default=None, help='Only check episodes of this object name')
    parser.add_argument('--max_episodes_per_object', type=int, default=None)
    parser.add_argument('--out_csv', type=str, default=None, help='Optional per-episode results CSV')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    csv_rows = []

    for ckpt_path in args.checkpoints:
        print('\n' + '=' * 78)
        print(f'Checkpoint: {ckpt_path}')
        print('=' * 78)
        policy = DiffusionPolicyInference(ckpt_path, device=args.device)
        cfg = policy.config
        print(f"spatial_softmax_proj={cfg.get('spatial_softmax_proj', False)}  "
              f"num_keypoints={cfg.get('num_keypoints', 32)}  state_keys={cfg.get('state_keys')}")

        # Weight variants: what deployment loads (EMA UNet) vs fully live weights.
        weight_sets = {'ema': {k: v.clone() for k, v in policy.model.state_dict().items()}}
        if 'ema' in policy.checkpoint and policy.checkpoint.get('model') is not None:
            weight_sets['live'] = policy.checkpoint['model']
        else:
            print("  (no EMA in this checkpoint -- 'ema' and 'live' would be identical, only 'ema' run)")

        ds = build_dataset(policy, args.dataset)
        offsets = state_offsets(ds)
        starts = episode_start_indices(ds)
        ep_obj = episode_object_names(ds)
        zarr_op = ds.zarr_root['data/operation_mode']

        # Group episodes by object (optionally filtered / capped).
        by_obj = defaultdict(list)
        for ep in sorted(starts):
            obj = ep_obj[ep]
            if args.object is not None and obj != args.object:
                continue
            if args.max_episodes_per_object is None or len(by_obj[obj]) < args.max_episodes_per_object:
                by_obj[obj].append(ep)
        if not by_obj:
            print(f'  No episodes matched (object={args.object!r}).')
            continue

        # Ground truth per object: first recorded action op_mode, and the
        # recorded frame-0 op_mode STATE (what deployment resets to idle).
        print('\nGround truth from the data (per episode, frame 0):')
        print(f"  {'object':<10} {'n_ep':>5}   first action op_mode        recorded frame-0 op_mode state")
        for obj, eps in sorted(by_obj.items()):
            first_act, state0 = Counter(), Counter()
            for ep in eps:
                start = 0 if ep == 0 else int(ds.episode_ends[ep - 1])
                sidx = ds.samples[starts[ep]]['sample_idx']
                first_act[op_mode_name(zarr_op[sidx])] += 1
                state0[op_mode_name(zarr_op[start])] += 1
            fmt = lambda c: ', '.join(f'{m} {c[m]}' for m in MODES if c[m])
            print(f'  {obj:<10} {len(eps):>5}   {fmt(first_act):<27} {fmt(state0)}')

        # Model predictions: weights x observation variant.
        print(f'\nPredicted op_mode of the FIRST executed action ({args.n_samples} samples per episode):')
        print(f"  {'object':<10} {'weights':<6} {'obs':<7} " + ' '.join(f'{m:>8}' for m in MODES)
              + f"   {'FB in exec chunk':>16}")
        for obj, eps in sorted(by_obj.items()):
            for wname, sd in weight_sets.items():
                policy.model.load_state_dict(sd)
                policy.model.eval()
                for variant in ('data', 'deploy'):
                    torch.manual_seed(args.seed)
                    counts, fb_chunk, total = Counter(), 0, 0
                    for ep in eps:
                        sample = ds[starts[ep]]
                        obs = make_obs(ds, sample, variant, offsets)
                        first, any_fb = first_step_modes(policy, obs, args.n_samples, cfg['action_horizon'])
                        ep_counts = Counter(first)
                        counts.update(ep_counts)
                        fb_chunk += sum(any_fb)
                        total += len(first)
                        csv_rows.append({
                            'checkpoint': ckpt_path, 'object': obj, 'episode': ep,
                            'weights': wname, 'obs': variant, 'n_samples': len(first),
                            **{f'first_{m}': ep_counts[m] for m in MODES},
                            'fb_in_exec_chunk': sum(any_fb),
                        })
                    pct = lambda n: f'{100.0 * n / total:7.1f}%'
                    print(f'  {obj:<10} {wname:<6} {variant:<7} ' + ' '.join(pct(counts[m]) for m in MODES)
                          + f'   {pct(fb_chunk):>16}')
        # Leave the policy in the deployment configuration.
        policy.model.load_state_dict(weight_sets['ema'])

    if args.out_csv and csv_rows:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f'\nPer-episode results: {args.out_csv}')

    print('\nHow to read this:')
    print("  FB high for ema+deploy only                -> train/deploy step-0 mismatch "
          "(check the recorded frame-0 op_mode state column)")
    print("  FB high for ema (data and deploy), not live -> EMA-UNet / live-encoder mismatch")
    print("  FB high in every row                        -> the model itself mis-decides on this "
          "object's start (ambiguous start vs handoff)")


if __name__ == '__main__':
    main()
