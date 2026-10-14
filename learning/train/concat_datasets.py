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

    # If each source is a separate per-object collection session (e.g.
    # session1 = all ObjectA episodes, session2 = all ObjectB episodes),
    # label them in the SAME merge step with --object_names -- one name per
    # source, same order -- instead of a separate label_object_id.py pass:
    python concat_datasets.py \\
        data/session_objectA/dataset.zarr \\
        data/session_objectB/dataset.zarr \\
        --output data/combined/dataset.zarr \\
        --object_names ObjectA ObjectB
    # This writes meta/object_names + meta/episode_object_id directly --
    # see dataset.py's 'object_id' state_keys component. If your sources
    # DON'T cleanly map one-to-one to objects (e.g. a session mixed several
    # objects, or you built the combined dataset some other way), use
    # label_object_id.py on the combined output instead.
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
                    dry_run=False, object_names=None):
    """
    Concatenate zarr datasets in the given order.

    Args:
        source_paths : list[str | Path]  source zarr directories, in merge order
        output_path  : str | Path        destination zarr directory
        chunk_size   : int               frames per I/O chunk (tune for RAM)
        overwrite    : bool              delete existing output if present
        dry_run      : bool              scan and validate only; write nothing
        object_names : list[str] | None  one object name per source_paths entry,
                       same order -- if given, every episode from source i is
                       labeled object_names[i], written as meta/object_names +
                       meta/episode_object_id (see dataset.py's 'object_id'
                       state_keys component). Must be len(source_paths); repeat
                       a name if two sources are the same object.
    """
    output_path = Path(output_path)
    if object_names is not None and len(object_names) != len(source_paths):
        raise ValueError(
            f"--object_names has {len(object_names)} entries but there are "
            f"{len(source_paths)} source datasets -- need exactly one name per source "
            f"(repeat a name if two sources share an object)."
        )

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
            data_group.create_array(
                key, shape=out_shape, dtype=dtype,
                chunks=(1,) + arr.shape[1:],
                compressors=Blosc(cname='lz4', clevel=3),
            )
        else:
            data_group.create_array(
                key, shape=out_shape, dtype=dtype,
                chunks=(min(chunk_size, total_frames),) + arr.shape[1:],
            )
        print(f"  {key:20s}  {str(out_shape):30s}  {dtype}")

    # ── 5. Copy data ──────────────────────────────────────────────────────────
    offset       = 0
    all_ep_ends  = []
    all_ep_object_names = []  # one entry per episode, only used if object_names given

    for src_i, src in enumerate(sources):
        n = src['n_frames']
        print(f"\nCopying source {src_i+1}/{len(sources)}: {src['path']}"
              f"  ({n} frames → [{offset}:{offset+n}])")

        # Update episode_ends with cumulative offset
        all_ep_ends.extend((src['ep_ends'] + offset).tolist())
        if object_names is not None:
            all_ep_object_names.extend([object_names[src_i]] * src['n_eps'])

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
    ep_ends_ds = meta_group.create_dataset(
        'episode_ends',
        shape=(len(ep_ends_arr),),
        dtype=np.int64,
        chunks=(max(1, len(ep_ends_arr)),),
    )
    ep_ends_ds[:] = ep_ends_arr

    # ── 6b. Write object labels, if requested ─────────────────────────────────
    if object_names is not None:
        unique_names = sorted(set(object_names))   # stable, alphabetical -> deterministic indices
        name_to_id = {name: i for i, name in enumerate(unique_names)}
        episode_object_id = np.array([name_to_id[n] for n in all_ep_object_names], dtype=np.int64)
        meta_group.create_array('episode_object_id', shape=episode_object_id.shape, dtype=np.int64)
        meta_group['episode_object_id'][:] = episode_object_id
        # Fixed-width unicode (not variable-length/object dtype) -- a plain
        # native numpy dtype, needs no special codec, works the same in
        # zarr v2 and v3 without chasing the exact VLenUTF8 import path.
        names_arr = np.array(unique_names, dtype=f'<U{max(len(n) for n in unique_names)}')
        meta_group.create_array('object_names', shape=names_arr.shape, dtype=names_arr.dtype)
        meta_group['object_names'][:] = names_arr
        counts = np.bincount(episode_object_id, minlength=len(unique_names))
        print(f"\nLabeled {total_eps} episodes -> {len(unique_names)} objects: {unique_names}")
        for name, count in zip(unique_names, counts):
            print(f"  {name}: {count} episodes")

    # ── 7. Copy metadata (camera_info etc.) from first source if present ──────
    for group_name in ('camera_info',):
        src_root = sources[0]['root']
        if group_name in src_root:
            src_grp = src_root[group_name]
            dst_grp = out.require_group(group_name)
            for key in src_grp.keys():
                arr = np.array(src_grp[key])
                ds = dst_grp.create_array(key, shape=arr.shape, dtype=arr.dtype)
                ds[:] = arr

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
    parser.add_argument('--object_names', nargs='+', default=None, metavar='NAME',
                        help='One object name per SOURCE, same order/count -- labels every '
                             'episode from that source with that object (see dataset.py\'s '
                             "'object_id' state_keys component). Omit if sources don't map "
                             'one-to-one to objects (use label_object_id.py on the output instead).')
    args = parser.parse_args()

    ok = concat_datasets(
        source_paths=args.sources,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        object_names=args.object_names,
    )
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
