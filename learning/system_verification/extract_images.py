#!/usr/bin/env python3
"""
Extract images from a zarr dataset to disk.

Usage:
    # Raw frames (original 480x640)
    python system_verification/extract_images.py --dataset Task0 --output /tmp/frames

    # Preprocessed frames — same center-crop + resize as training (216x288 by default)
    python system_verification/extract_images.py --dataset Task0 --output /tmp/frames --preprocessed

    # Custom target size (must match config image_size)
    python system_verification/extract_images.py --dataset Task0 --output /tmp/frames --preprocessed --image_size 216 288

    # Extract only episode 0
    python system_verification/extract_images.py --dataset Task0 --output /tmp/frames --episode 0

    # Subsample every 10th frame
    python system_verification/extract_images.py --dataset Task0 --output /tmp/frames --step 10

    # Save as MP4 video
    python system_verification/extract_images.py --dataset Task0 --output /tmp/videos --video --fps 10

    # Preprocessed video
    python system_verification/extract_images.py --dataset Task0 --output /tmp/videos --video --preprocessed --fps 10
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

SYSVER_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(SYSVER_DIR)
sys.path.insert(0, LEARNING_DIR)

from hardware.image_utils import crop_and_resize


def preprocess_image(img_rgb: np.ndarray, target_h: int, target_w: int,
                      crop_scale: float = 1.5, crop_x: float = 0.5, crop_y: float = 0.5) -> np.ndarray:
    """
    Apply the same crop + resize pipeline as dataset.py (see image_utils.crop_and_resize).

    Args:
        img_rgb  : (H, W, 3) uint8 RGB image
        target_h : target height in pixels
        target_w : target width  in pixels
        crop_scale, crop_x, crop_y : crop window params (must match training config)

    Returns:
        (target_h, target_w, 3) uint8 RGB — ready to save
    """
    return crop_and_resize(img_rgb, (target_h, target_w),
                            crop_scale=crop_scale, crop_x=crop_x, crop_y=crop_y)


def extract_images(
    dataset_path,
    output_dir,
    episode=None,
    step=1,
    as_video=False,
    fps=10,
    preprocessed=True,
    image_size=(216, 288),
    crop_scale=1.5,
    crop_x=0.5,
    crop_y=0.5,
):
    import zarr
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening dataset: {dataset_path}")
    z = zarr.open(str(dataset_path), mode='r')

    episode_ends = z['meta/episode_ends'][:]
    n_episodes   = len(episode_ends)
    total_frames = int(episode_ends[-1])
    raw_shape    = z['data/camera_0'].shape[1:]      # (H, W, 3)

    target_h, target_w = image_size
    mode_str = (
        f"preprocessed {target_h}×{target_w} (crop→resize, same as training)"
        if preprocessed else
        f"raw {raw_shape[0]}×{raw_shape[1]}"
    )
    print(f"Dataset  : {n_episodes} episodes, {total_frames} frames")
    print(f"Raw size : {raw_shape[0]}×{raw_shape[1]}")
    print(f"Mode     : {mode_str}")

    episodes = [episode] if episode is not None else list(range(n_episodes))

    for ep_idx in episodes:
        ep_start      = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        ep_end        = int(episode_ends[ep_idx])
        ep_len        = ep_end - ep_start
        frame_indices = list(range(ep_start, ep_end, step))

        print(f"\nEpisode {ep_idx:03d}: frames {ep_start}–{ep_end-1}"
              f" ({ep_len} frames, extracting {len(frame_indices)})")

        if as_video:
            video_path = output_dir / f"episode_{ep_idx:03d}.mp4"
            out_h = target_h if preprocessed else raw_shape[0]
            out_w = target_w if preprocessed else raw_shape[1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (out_w, out_h))

            for i, fi in enumerate(frame_indices):
                img_rgb = z['data/camera_0'][fi]           # uint8 RGB
                if preprocessed:
                    img_rgb = preprocess_image(img_rgb, target_h, target_w, crop_scale, crop_x, crop_y)
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
                img_rgb = z['data/camera_0'][fi]           # uint8 RGB
                if preprocessed:
                    img_rgb = preprocess_image(img_rgb, target_h, target_w, crop_scale, crop_x, crop_y)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_path = ep_dir / f"frame_{i:05d}_global{fi:06d}.jpg"
                cv2.imwrite(str(img_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if (i + 1) % 100 == 0 or i == len(frame_indices) - 1:
                    print(f"  {i+1}/{len(frame_indices)} images saved", end='\r')

            print(f"\n  Saved to: {ep_dir}/")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Extract images from zarr dataset")
    parser.add_argument('--dataset', type=str, default=None,
                        help='Task folder name under data/demo_data/ (e.g. Task0, Task3)')
    parser.add_argument('--output', type=str, default='data/extracted_images',
                        help='Output root directory')
    parser.add_argument('--episode', '-ep', type=int, default=0,
                        help='Extract only this episode index (default: all)')
    parser.add_argument('--step', type=int, default=1,
                        help='Save every N-th frame (1 = all frames)')
    parser.add_argument('--video', action='store_true',
                        help='Save as MP4 video instead of individual images')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for video output (default: 10)')
    parser.add_argument('--preprocessed', default=True, action='store_true',
                        help='Apply training preprocessing: center-crop then resize to image_size')
    parser.add_argument('--image_size', type=int, nargs=2, default=[216, 288],
                        metavar=('H', 'W'),
                        help='Target image size after preprocessing (default: 216 288). '
                             'Must match config image_size used during training.')
    parser.add_argument('--crop_scale', type=float, default=1.5,
                        help='Crop window size as a multiple of image_size (default: 1.5). '
                             'Must match config crop_scale used during training.')
    parser.add_argument('--crop_x', type=float, default=0.5,
                        help='Crop anchor x in [0,1]: 0=left, 0.5=center, 1=right. '
                             'Must match config crop_x used during training.')
    parser.add_argument('--crop_y', type=float, default=0.5,
                        help='Crop anchor y in [0,1]: 0=top, 0.5=center, 1=bottom. '
                             'Must match config crop_y used during training.')
    args = parser.parse_args()
    if args.dataset is None:
        
        data_dir  = Path(__file__).parent.parent.parent / 'data' / 'demo_data'
        dataset   = data_dir / 'dataset.zarr'
        output_dir = Path(args.output) 
    else:
        dataset = args.dataset
        output_dir = Path(args.output)

    extract_images(
        dataset_path=dataset,
        output_dir=output_dir,
        episode=args.episode,
        step=args.step,
        as_video=args.video,
        fps=args.fps,
        preprocessed=args.preprocessed,
        image_size=tuple(args.image_size),
        crop_scale=args.crop_scale,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
    )


if __name__ == '__main__':
    main()
