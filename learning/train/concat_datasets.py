#!/usr/bin/env python3
"""
Concatenate multiple zarr demonstration datasets into a single dataset.

Episodes are merged in the order the source paths are given, so you control
which episodes appear first in the combined dataset.

Usage:
    # Combine two sessions into one
    python concat_datasets.py \\
        data/session1/dataset.zarr \\
        data/session2/dataset.zarr \\
        --output data/combined/dataset.zarr

    # Explicit order (session3 first, then session1, then session2)
    python concat_datasets.py \\
        data/session3/dataset.zarr \\
        data/session1/dataset.zarr \\
        data/session2/dataset.zarr \\
        --output data/combined/dataset.zarr

    # Preview without writing (--dry_run)
    python concat_datasets.py \\
        data/session1/dataset.zarr \\
        data/session2/dataset.zarr \\
        --output data/combined/dataset.zarr --dry_run
"""

import sys
import argparse
import shutil
import numpy as np
import zarr
from pathlib import Path
from zarr.codecs.numcodecs import Blosc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan(path):
    """Return summary dict for one source dataset."""
    root = zarr.open(str(path), mode='r')
    ep_ends = root['meta/episode_ends'][:]
    n_frames = int(ep_ends[-1]) if len(ep_ends) > 0 else 0
    n_eps    = len(ep_ends)
    keys     = set(root['data'].keys())
    shapes   = {k: root['data'][k].shape[1:] for k in keys}
    dtypes   = {k: root['data'][k].dtype      for k in keys}
    return dict(root=root, path=str(path), ep_ends=ep_ends,
                n_frames=n_frames, n_eps=n_eps,
                keys=keys, shapes=shapes, dtypes=dtypes)


def _print_summary(sources):
    total_eps    = sum(s['n_eps']    for s in sources)
    total_frames = sum(s['n_frames'] for s in sources)
    print(f"\n{'─'*62}")
    print(f"{'#':>3}  {'Path':<38}  {'Eps':>5}  {'Frames':>7}")
    print(f"{'─'*62}")
    for i, s in enumerate(sources):
        print(f"{i+1:>3}  {s['path']:<38}  {s['n_eps']:>5}  {s['n_frames']:>7}")
    print(f"{'─'*62}")
    print(f"{'':>3}  {'TOTAL':<38}  {total_eps:>5}  {total_frames:>7}")
    print(f"{'─'*62}\n")


# ── Core ──────────────────────────────────────────────────────────────────────

def concat_datasets(source_paths, output_path, chunk_size=100, overwrite=False,
                    dry_run=False):
    """
    Concatenate zarr datasets in the given order.

    Args:
        source_paths : list[str | Path]  source zarr directories, in merge order
        output_path  : str | Path        destination zarr directory
        chunk_size   : int               frames per I/O chunk (tune for RAM)
        overwrite    : bool              delete existing output if present
        dry_run      : bool              scan and validate only; write nothing
    """
    output_path = Path(output_path)

    # ── 1. Scan ───────────────────────────────────────────────────────────────
    print("Scanning source datasets ...")
    sources = []
    for p in source_paths:
        p = Path(p)
        if not p.exists():
            print(f"  ❌ Not found: {p}")
            return False
        s = _scan(p)
        print(f"  ✓  {p}  →  {s['n_eps']} episodes, {s['n_frames']} frames  "
              f"keys={sorted(s['keys'])}")
        sources.append(s)

    _print_summary(sources)

    # ── 2. Validate ───────────────────────────────────────────────────────────
    all_keys = [s['keys'] for s in sources]
    common_keys = set.intersection(*all_keys)
    dropped_keys = set.union(*all_keys) - common_keys
    if dropped_keys:
        print(f"  ⚠️  Keys not present in ALL datasets will be skipped: {dropped_keys}")
    print(f"  Keys to merge: {sorted(common_keys)}\n")

    # Validate per-key shapes match across sources
    for key in common_keys:
        shapes = [s['shapes'][key] for s in sources]
        if len(set(shapes)) > 1:
            print(f"  ❌ Shape mismatch for '{key}': {shapes}")
            return False
        dtypes = [str(s['dtypes'][key]) for s in sources]
        if len(set(dtypes)) > 1:
            print(f"  ⚠️  dtype mismatch for '{key}': {dtypes}  (will use first)")

    total_frames = sum(s['n_frames'] for s in sources)
    total_eps    = sum(s['n_eps']    for s in sources)
    print(f"Output: {output_path}")
    print(f"  {total_eps} episodes  |  {total_frames} frames total")

    if dry_run:
        print("\n✅ Dry run complete — no files written.")
        return True

    # ── 3. Prepare output ─────────────────────────────────────────────────────
    if output_path.exists():
        if overwrite:
            print(f"\nRemoving existing output: {output_path}")
            shutil.rmtree(output_path)
        else:
            print(f"\n❌ Output already exists: {output_path}")
            print("   Use --overwrite to replace it.")
            return False

    output_path.mkdir(parents=True)
    out        = zarr.open(str(output_path), mode='w')
    data_group = out.require_group('data')
    meta_group = out.require_group('meta')

    # ── 4. Pre-allocate output arrays ─────────────────────────────────────────
    print("\nPre-allocating output arrays ...")
    first = sources[0]
    for key in sorted(common_keys):
        arr       = first['root']['data'][key]
        out_shape = (total_frames,) + arr.shape[1:]
        dtype     = arr.dtype

        if key == 'camera_0':
            data_group.create_dataset(
                key, shape=out_shape, dtype=dtype,
                chunks=(1,) + arr.shape[1:],
                compressor=Blosc(cname='lz4', clevel=3),
            )
        else:
            data_group.create_dataset(
                key, shape=out_shape, dtype=dtype,
                chunks=(min(chunk_size, total_frames),) + arr.shape[1:],
            )
        print(f"  {key:20s}  {str(out_shape):30s}  {dtype}")

    # ── 5. Copy data ──────────────────────────────────────────────────────────
    offset       = 0
    all_ep_ends  = []

    for src_i, src in enumerate(sources):
        n = src['n_frames']
        print(f"\nCopying source {src_i+1}/{len(sources)}: {src['path']}"
              f"  ({n} frames → [{offset}:{offset+n}])")

        # Update episode_ends with cumulative offset
        all_ep_ends.extend((src['ep_ends'] + offset).tolist())

        # Copy each key in chunks
        for key in sorted(common_keys):
            src_arr = src['root']['data'][key]
            dst_arr = data_group[key]

            copied = 0
            while copied < n:
                end = min(copied + chunk_size, n)
                dst_arr[offset + copied : offset + end] = src_arr[copied:end]
                copied = end

            print(f"  ✓  {key}")

        offset += n

    # ── 6. Write episode_ends ─────────────────────────────────────────────────
    ep_ends_arr = np.array(all_ep_ends, dtype=np.int64)
    meta_group.create_dataset('episode_ends', data=ep_ends_arr,
                              chunks=(max(1, len(ep_ends_arr)),))

    # ── 7. Copy metadata (camera_info etc.) from first source if present ──────
    for group_name in ('camera_info',):
        if group_name in sources[0]['root']:
            zarr.copy(sources[0]['root'][group_name], out, name=group_name)

    print(f"\n✅ Done!  {total_eps} episodes | {total_frames} frames")
    print(f"   Saved to: {output_path}\n")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Concatenate zarr demonstration datasets in the given order.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('sources', nargs='+', metavar='SOURCE',
                        help='Source zarr paths, in the order they should be merged')
    parser.add_argument('--output', '-o', required=True,
                        help='Output zarr path')
    parser.add_argument('--chunk_size', type=int, default=100, metavar='N',
                        help='Frames to copy per I/O chunk (default 100). '
                             'Larger = faster but uses more RAM.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output if it already exists')
    parser.add_argument('--dry_run', action='store_true',
                        help='Scan and validate only; do not write anything')
    args = parser.parse_args()

    ok = concat_datasets(
        source_paths=args.sources,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
