#!/usr/bin/env python3
"""
Check a demo_collect.py dataset for the camera-shape-corruption bug: an
earlier `save_episode` resized data/camera_0 (or camera_1) without checking
it matched the array's existing per-frame shape, so if camera settings
(--camera_width/--camera_height) differed across separate demo_collect.py
sessions that wrote into the same --output directory, earlier episodes'
frames can be laid out on disk under a shape the array no longer declares --
reading them back (e.g. via concat_datasets.py) can then stall or error.

This reads just the FIRST frame of every episode (cheap) and reports which
ones fail to decompress cleanly.

Usage:
    python system_verification/check_camera_shape_consistency.py \
        data/your_dataset/dataset.zarr [--key camera_0]
"""
import argparse
import zarr
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('--key', default='camera_0')
    args = parser.parse_args()

    root = zarr.open(args.dataset_path, mode='r')
    arr = root[f'data/{args.key}']
    ep_ends = root['meta/episode_ends'][:]
    print(f"Declared array shape: {arr.shape}, dtype={arr.dtype}")
    print(f"{len(ep_ends)} episodes, {int(ep_ends[-1])} total frames\n")

    start = 0
    n_bad = 0
    for i, end in enumerate(ep_ends):
        start_i, end_i = start, int(end)
        try:
            frame = np.asarray(arr[start_i])
            ok = frame.shape == arr.shape[1:] and np.isfinite(frame.astype(np.float64)).all()
            status = "OK" if ok else "SUSPICIOUS (unexpected content)"
        except Exception as e:
            ok = False
            status = f"READ FAILED: {e}"
        if not ok:
            n_bad += 1
        print(f"Episode {i:3d}  frames [{start_i}:{end_i}]  first-frame {status}")
        start = end_i

    print(f"\n{'✅ All episodes read cleanly.' if n_bad == 0 else f'⚠️  {n_bad} episode(s) failed -- see above.'}")
    if n_bad > 0:
        print("Affected episodes were likely recorded with different camera settings than the "
              "rest of this dataset (see demo_collect.py's save_episode fix). You'll need to "
              "either exclude them (dataset.py's exclude_episodes) or recollect them.")


if __name__ == '__main__':
    main()
