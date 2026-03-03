#!/usr/bin/env python3
"""
Dataset management utility for the zarr demonstration dataset.

Usage:
    # Show episode summary
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --info

    # Remove the last episode
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_last

    # Remove the last N episodes
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_last 3

    # Remove specific episodes by index (space-separated)
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_episodes 2 5 7

    # Remove a contiguous range  (e.g. episodes 3, 4, 5)
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_episodes 3-5

    # Mix of singles and ranges
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_episodes 0 3-5 9

    # Skip confirmation
    python manage_dataset.py --dataset data/demo_data/dataset.zarr --remove_episodes 2 5 --yes
"""

import sys
import argparse
import numpy as np
import zarr
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_indices(tokens, n_eps):
    """
    Parse a list of index tokens into a sorted list of episode indices.

    Each token can be:
      - an integer              e.g. "3"
      - a closed range A-B      e.g. "3-5"  → [3, 4, 5]
      - a Python-style slice    e.g. "3:6"  → [3, 4, 5]
    """
    indices = set()
    for token in tokens:
        token = str(token)
        if '-' in token and not token.startswith('-'):
            # A-B range (inclusive both ends)
            parts = token.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid range '{token}'. Use A-B format.")
            a, b = int(parts[0]), int(parts[1])
            indices.update(range(a, b + 1))
        elif ':' in token:
            # Python slice  A:B  (exclusive end, like range)
            parts = token.split(':')
            a = int(parts[0]) if parts[0] else 0
            b = int(parts[1]) if parts[1] else n_eps
            indices.update(range(a, b))
        else:
            indices.add(int(token))

    out = sorted(indices)
    bad = [i for i in out if i < 0 or i >= n_eps]
    if bad:
        raise ValueError(f"Episode indices out of range (0–{n_eps-1}): {bad}")
    return out


def dataset_info(zarr_root, mark=None):
    """
    Print episode summary table.

    Args:
        mark : set of episode indices to highlight (e.g. ones to be removed)
    """
    episode_ends = zarr_root['meta/episode_ends'][:]
    n_eps        = len(episode_ends)
    total_steps  = int(episode_ends[-1]) if n_eps > 0 else 0

    print(f"\nDataset: {n_eps} episodes, {total_steps} total frames")
    print(f"{'Ep':>4}  {'Start':>7}  {'End':>7}  {'Frames':>7}  {'Note'}")
    print("─" * 45)
    for i, end in enumerate(episode_ends):
        start  = 0 if i == 0 else int(episode_ends[i - 1])
        steps  = int(end) - start
        note   = ""
        if mark and i in mark:
            note = "← REMOVE"
        elif i == n_eps - 1:
            note = "← LAST"
        print(f"{i:>4}  {start:>7}  {int(end):>7}  {steps:>7}  {note}")

    print(f"\nData arrays:")
    for key in zarr_root['data'].keys():
        arr = zarr_root['data'][key]
        print(f"  data/{key}: {arr.shape}  dtype={arr.dtype}")


# ── Core ──────────────────────────────────────────────────────────────────────

def remove_episodes(zarr_root, ep_indices, dry_run=False):
    """
    Remove a list of episodes from the dataset in a single compact pass.

    Episodes are processed in order; kept episodes are shifted down to fill
    gaps left by removed ones. No episode is read/written more than once.

    Args:
        zarr_root  : zarr.Group opened in write mode ('a')
        ep_indices : sorted list of episode indices to remove (0-based)
        dry_run    : if True, print plan but make no changes
    Returns:
        True on success, False on validation failure
    """
    episode_ends = zarr_root['meta/episode_ends'][:]
    n_eps        = len(episode_ends)

    if n_eps == 0:
        print("❌ No episodes in dataset.")
        return False

    # Validate
    bad = [i for i in ep_indices if i < 0 or i >= n_eps]
    if bad:
        print(f"❌ Episode indices out of range (0–{n_eps-1}): {bad}")
        return False

    remove_set   = set(ep_indices)
    ep_starts    = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    ep_ends_arr  = episode_ends.astype(np.int64)

    # Summary
    removed_frames = sum(int(ep_ends_arr[i]) - int(ep_starts[i]) for i in remove_set)
    kept_frames    = int(ep_ends_arr[-1]) - removed_frames
    print(f"\nRemoving {len(remove_set)} episode(s)  ({removed_frames} frames):")
    for i in sorted(remove_set):
        s, e = int(ep_starts[i]), int(ep_ends_arr[i])
        print(f"  Episode {i:>3}:  frames {s:>7} → {e:>7}  ({e-s} frames)")
    print(f"Kept: {n_eps - len(remove_set)} episodes  ({kept_frames} frames)")

    if dry_run:
        print("\n(dry run — no changes made)")
        return True

    # ── Single-pass compaction ────────────────────────────────────────────────
    # Walk episodes in order. Keep a write pointer (dst) that advances only
    # for kept episodes. Removed episodes create a gap; subsequent kept
    # episodes are copied into the gap. Because dst ≤ src always (we only
    # remove, never insert), reads never overlap with writes.

    new_ep_ends = []
    dst = 0   # write pointer in frames

    for i in range(n_eps):
        src_start = int(ep_starts[i])
        src_end   = int(ep_ends_arr[i])
        ep_len    = src_end - src_start

        if i in remove_set:
            continue   # skip: just advance src implicitly

        # Copy this episode's data into position dst
        if dst != src_start:
            for key in zarr_root['data'].keys():
                arr = zarr_root['data'][key]
                arr[dst:dst + ep_len] = arr[src_start:src_end]

        dst += ep_len
        new_ep_ends.append(dst)

    # Truncate all arrays to new total length
    for key in zarr_root['data'].keys():
        arr = zarr_root['data'][key]
        arr.resize((dst,) + arr.shape[1:])

    # Rewrite episode_ends
    new_ends_arr = np.array(new_ep_ends, dtype=np.int64)
    zarr_root['meta/episode_ends'].resize(len(new_ends_arr))
    if len(new_ends_arr) > 0:
        zarr_root['meta/episode_ends'][:] = new_ends_arr

    remaining = n_eps - len(remove_set)
    print(f"\n✅ Done. Dataset now has {remaining} episodes ({kept_frames} frames).")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Manage zarr demonstration dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--dataset', type=str,
                        default='data/demo_data/dataset.zarr',
                        help='Path to dataset.zarr')
    parser.add_argument('--info', action='store_true',
                        help='Show episode summary (read-only)')

    rem = parser.add_mutually_exclusive_group()
    rem.add_argument('--remove_last', type=int, nargs='?', const=1, metavar='N',
                     help='Remove the last N episodes (default N=1)')
    rem.add_argument('--remove_episodes', nargs='+', metavar='IDX',
                     help='Remove episodes by index. Accepts individual indices '
                          '(e.g. 2 5 7), closed ranges (e.g. 3-5), or a mix '
                          '(e.g. 0 3-5 9). Indices are 0-based.')

    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt')
    args = parser.parse_args()

    zarr_path = Path(args.dataset)
    if not zarr_path.exists():
        print(f"❌ Dataset not found: {zarr_path}")
        return 1

    zarr_root  = zarr.open(str(zarr_path), mode='a')
    ep_ends    = zarr_root['meta/episode_ends'][:]
    n_eps      = len(ep_ends)

    # ── Determine indices to remove ───────────────────────────────────────────
    if args.remove_last is not None:
        n = args.remove_last
        if n < 1 or n > n_eps:
            print(f"❌ Cannot remove {n} episodes (only {n_eps} exist).")
            return 1
        ep_indices = list(range(n_eps - n, n_eps))

    elif args.remove_episodes is not None:
        try:
            ep_indices = _parse_indices(args.remove_episodes, n_eps)
        except ValueError as e:
            print(f"❌ {e}")
            return 1

    else:
        # No removal requested — just show info
        dataset_info(zarr_root)
        if not args.info:
            print("\nNo action specified. Use --info, --remove_last [N], "
                  "or --remove_episodes IDX [IDX ...].")
        return 0

    # ── Show current state with removals highlighted ──────────────────────────
    dataset_info(zarr_root, mark=set(ep_indices))

    # ── Confirm ───────────────────────────────────────────────────────────────
    if not args.yes:
        answer = input(f"\nRemove {len(ep_indices)} episode(s)? [y/N] ").strip().lower()
        if answer != 'y':
            print("Aborted.")
            return 0

    ok = remove_episodes(zarr_root, ep_indices)
    if ok:
        print("\nDataset after removal:")
        dataset_info(zarr_root)

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
