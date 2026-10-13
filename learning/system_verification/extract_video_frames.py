#!/usr/bin/env python3
"""
Extract frames from a plain video file (e.g. a deployment evaluation
recording) to disk as individual images. No dataset/training coupling --
just reads the video and saves frames as-is.

Usage:
    # All frames
    python system_verification/extract_video_frames.py --video path/to/clip.mp4 --output /tmp/frames

    # Every 10th frame only
    python system_verification/extract_video_frames.py --video path/to/clip.mp4 --output /tmp/frames --step 10

    # Only the frames between 5s and 20s into the video
    python system_verification/extract_video_frames.py --video path/to/clip.mp4 --output /tmp/frames \
        --start_time 5 --end_time 20

    # Save as PNG instead of JPEG
    python system_verification/extract_video_frames.py --video path/to/clip.mp4 --output /tmp/frames --format png
"""

import argparse
from pathlib import Path


def extract_video_frames(
    video_path,
    output_dir,
    step=1,
    start_frame=0,
    end_frame=None,
    start_time=None,
    end_time=None,
    image_format='jpg',
    jpeg_quality=95,
):
    import cv2

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --start_time/--end_time (seconds) override --start_frame/--end_frame if given.
    if start_time is not None:
        start_frame = int(round(start_time * fps)) if fps > 0 else start_frame
    if end_time is not None:
        end_frame = int(round(end_time * fps)) if fps > 0 else end_frame
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    print(f"Video    : {video_path.name}")
    print(f"Frames   : {total_frames} total, {fps:.1f} fps, {raw_w}x{raw_h}")
    print(f"Range    : frames [{start_frame}:{end_frame}) (step={step})")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved = 0
    while frame_idx < end_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"\n  ⚠️  Failed to read frame {frame_idx}, stopping early.")
            break

        if (frame_idx - start_frame) % step == 0:
            img_path = output_dir / f"frame_{frame_idx:06d}.{image_format}"
            if image_format in ('jpg', 'jpeg'):
                cv2.imwrite(str(img_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            else:
                cv2.imwrite(str(img_path), frame_bgr)
            saved += 1
            if saved % 100 == 0:
                print(f"  {saved} frames saved (at source frame {frame_idx})", end='\r')

        frame_idx += 1

    cap.release()
    print(f"\n  Saved {saved} frames to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='data/extracted_frames',
                        help='Output directory for extracted frames')
    parser.add_argument('--step', type=int, default=1,
                        help='Save every N-th frame (1 = all frames)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='First frame index to extract (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='Last frame index to extract (exclusive, default: end of video)')
    parser.add_argument('--start_time', type=float, default=None,
                        help='Start time in seconds (overrides --start_frame)')
    parser.add_argument('--end_time', type=float, default=None,
                        help='End time in seconds (overrides --end_frame)')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'jpeg', 'png'],
                        help='Output image format (default: jpg)')
    parser.add_argument('--jpeg_quality', type=int, default=95,
                        help='JPEG quality 1-100, used when --format is jpg/jpeg (default: 95)')
    args = parser.parse_args()

    extract_video_frames(
        video_path=args.video,
        output_dir=args.output,
        step=args.step,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        start_time=args.start_time,
        end_time=args.end_time,
        image_format=args.format,
        jpeg_quality=args.jpeg_quality,
    )


if __name__ == '__main__':
    main()
