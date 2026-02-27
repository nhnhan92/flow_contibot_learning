#!/usr/bin/env python3
"""
Extract images from a zarr dataset to disk.

Usage:
    # Extract every frame of every episode
    python train/extract_images.py --dataset data/demo_data/Task0/dataset.zarr --output /tmp/frames

    # Extract only episode 0
    python train/extract_images.py --dataset data/demo_data/Task0/dataset.zarr --output /tmp/frames --episode 0

    # Extract one frame per second (subsample every N frames)
    python train/extract_images.py --dataset data/demo_data/Task0/dataset.zarr --output /tmp/frames --step 10

    # Save as a video (MP4) instead of individual images
    python train/extract_images.py --dataset data/demo_data/Task0/dataset.zarr --output /tmp/videos --video --fps 10
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, LEARNING_DIR)


def extract_images(dataset_path, output_dir, episode=None, step=1, as_video=False, fps=10):
    import zarr
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening dataset: {dataset_path}")
    z = zarr.open(str(dataset_path), mode='r')

    episode_ends = z['meta/episode_ends'][:]
    n_episodes = len(episode_ends)
    total_frames = int(episode_ends[-1])

    print(f"Dataset: {n_episodes} episodes, {total_frames} total frames")
    print(f"Image shape per frame: {z['data/camera_0'].shape[1:]}")

    # Determine which episodes to process
    if episode is not None:
        episodes = [episode]
    else:
        episodes = list(range(n_episodes))

    for ep_idx in episodes:
        ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        ep_end   = int(episode_ends[ep_idx])
        ep_len   = ep_end - ep_start
        frame_indices = list(range(ep_start, ep_end, step))

        print(f"\nEpisode {ep_idx:03d}: frames {ep_start}–{ep_end-1} ({ep_len} frames, extracting {len(frame_indices)})")

        if as_video:
            video_path = output_dir / f"episode_{ep_idx:03d}.mp4"
            first_frame = z['data/camera_0'][ep_start]
            H, W = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

            for i, fi in enumerate(frame_indices):
                img_rgb = z['data/camera_0'][fi]           # (H, W, 3) uint8 RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                writer.write(img_bgr)
                if (i + 1) % 50 == 0 or i == len(frame_indices) - 1:
                    print(f"  {i+1}/{len(frame_indices)} frames written", end='\r')

            writer.release()
            print(f"\n  Saved: {video_path}")

        else:
            ep_dir = output_dir / f"episode_{ep_idx:03d}"
            ep_dir.mkdir(exist_ok=True)

            for i, fi in enumerate(frame_indices):
                img_rgb = z['data/camera_0'][fi]           # (H, W, 3) uint8 RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_path = ep_dir / f"frame_{i:05d}_global{fi:06d}.jpg"
                cv2.imwrite(str(img_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if (i + 1) % 100 == 0 or i == len(frame_indices) - 1:
                    print(f"  {i+1}/{len(frame_indices)} images saved", end='\r')

            print(f"\n  Saved to: {ep_dir}/")

    print("\nDone.")


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Extract images from zarr dataset")
    parser.add_argument('--dataset', type=str,
                        default='Task3',
                        help='Path to dataset.zarr')
    parser.add_argument('--output', type=str, default='data/extracted_images',
                        help='Output directory')
    parser.add_argument('--episode', '-ep', type=int, default=None,
                        help='Extract only this episode index (default: all)')
    parser.add_argument('--step', type=int, default=1,
                        help='Save every N-th frame (1 = all frames)')
    parser.add_argument('--video', action='store_true',
                        help='Save as MP4 video instead of individual images')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for video output (default: 10)')
    args = parser.parse_args()
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'demo_data'
    dataset = data_dir / args.dataset / 'dataset.zarr'

    output_dir = Path(args.output) / args.dataset
    extract_images(
        dataset_path=dataset,
        output_dir=output_dir,
        episode=args.episode,
        step=args.step,
        as_video=args.video,
        fps=args.fps,
    )


if __name__ == '__main__':
    main()
