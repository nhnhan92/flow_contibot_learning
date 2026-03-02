#!/usr/bin/env python3
"""
Dataset management utility for the zarr demonstration dataset.

Usage:
    # Preview last episode (no changes made)
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --info

    # Remove the last episode
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_last

    # Remove a specific episode by index
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_episode 3
"""

import argparse
import numpy as np
import zarr
from pathlib import Path


def dataset_info(zarr_root):
    """Print summary of all episodes in the dataset."""
    episode_ends = zarr_root['meta/episode_ends'][:]
    n_eps = len(episode_ends)
    total_steps = int(episode_ends[-1]) if n_eps > 0 else 0

    print(f"\nDataset: {n_eps} episodes, {total_steps} total steps")
    print(f"{'Ep':>4}  {'Start':>7}  {'End':>7}  {'Steps':>6}")
    print("-" * 32)
    for i, end in enumerate(episode_ends):
        start = 0 if i == 0 else int(episode_ends[i - 1])
        steps = int(end) - start
        marker = " ← LAST" if i == n_eps - 1 else ""
        print(f"{i:>4}  {start:>7}  {int(end):>7}  {steps:>6}{marker}")

    print(f"\nData arrays:")
    for key in zarr_root['data'].keys():
        arr = zarr_root['data'][key]
        print(f"  data/{key}: {arr.shape}  dtype={arr.dtype}")


def remove_episode(zarr_root, ep_idx, dry_run=False):
    """
    Remove a single episode from the dataset by index.

    All episodes AFTER ep_idx are shifted down (data is compacted).
    """
    episode_ends = zarr_root['meta/episode_ends'][:]
    n_eps = len(episode_ends)

    if n_eps == 0:
        print("❌ No episodes in dataset.")
        return False

    if ep_idx < 0 or ep_idx >= n_eps:
        print(f"❌ Episode index {ep_idx} out of range (0–{n_eps - 1}).")
        return False

    ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
    ep_end   = int(episode_ends[ep_idx])
    ep_steps = ep_end - ep_start

    print(f"\nEpisode {ep_idx}: steps {ep_start} → {ep_end} ({ep_steps} frames)")

    if dry_run:
        print("(dry run — no changes made)")
        return True

    # ── Compact each data array: shift data after ep_end down by ep_steps ─────
    for key in zarr_root['data'].keys():
        arr = zarr_root['data'][key]
        total = arr.shape[0]

        if ep_end < total:
            # Shift trailing data into the gap left by the removed episode
            trailing = arr[ep_end:]
            arr[ep_start:ep_start + len(trailing)] = trailing

        # Truncate to new length
        new_total = total - ep_steps
        arr.resize((new_total,) + arr.shape[1:])

    # ── Update episode_ends ────────────────────────────────────────────────────
    # Rebuild: remove ep_idx entry and subtract ep_steps from all later entries
    new_ends = np.concatenate([
        episode_ends[:ep_idx],
        episode_ends[ep_idx + 1:] - ep_steps
    ]).astype(np.int64)

    zarr_root['meta/episode_ends'].resize(len(new_ends))
    if len(new_ends) > 0:
        zarr_root['meta/episode_ends'][:] = new_ends

    print(f"✅ Episode {ep_idx} removed. Dataset now has {n_eps - 1} episodes.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Manage zarr demonstration dataset')
    parser.add_argument('--dataset', type=str,
                        default='data/demo_data/dataset.zarr',
                        help='Path to dataset.zarr')
    parser.add_argument('--info', action='store_true',
                        help='Show episode summary (read-only)')
    parser.add_argument('--remove_last', action='store_true',
                        help='Remove the last episode')
    parser.add_argument('--remove_episode', type=int, default=None,
                        metavar='IDX',
                        help='Remove episode by index')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt')
    args = parser.parse_args()

    zarr_path = Path(args.dataset)
    if not zarr_path.exists():
        print(f"❌ Dataset not found: {zarr_path}")
        return 1

    zarr_root = zarr.open(str(zarr_path), mode='a')

    # ── Info ──────────────────────────────────────────────────────────────────
    dataset_info(zarr_root)

    if args.info:
        return 0

    # ── Determine which episode to remove ─────────────────────────────────────
    episode_ends = zarr_root['meta/episode_ends'][:]
    n_eps = len(episode_ends)

    if args.remove_last:
        ep_idx = n_eps - 1
    elif args.remove_episode is not None:
        ep_idx = args.remove_episode
    else:
        print("\nNo action specified. Use --info, --remove_last, or --remove_episode IDX.")
        return 0

    # Preview
    remove_episode(zarr_root, ep_idx, dry_run=True)

    # Confirm
    if not args.yes:
        answer = input(f"\nRemove episode {ep_idx}? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            return 0

    remove_episode(zarr_root, ep_idx)
    print("\nDataset after removal:")
    dataset_info(zarr_root)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
