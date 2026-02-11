#!/usr/bin/env python3
"""
Test RealSense Camera for Diffusion Policy Data Collection

Kiểm tra:
1. RGB stream - ảnh màu (bắt buộc cho training)
2. Depth stream - ảnh độ sâu (optional)
3. Camera intrinsics - thông số nội tại
4. FPS thực tế - đảm bảo đủ tốc độ
5. Latency - độ trễ

Usage:
    cd ~/Desktop/my_pickplace
    python scripts/test_camera.py
    python scripts/test_camera.py --no-display  # Không hiển thị GUI
"""

import time
import click
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed. Run: pip install pyrealsense2")
    exit(1)


def get_camera_info():
    """Get connected RealSense camera information"""
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        return None

    cameras = []
    for dev in devices:
        info = {
            'name': dev.get_info(rs.camera_info.name),
            'serial': dev.get_info(rs.camera_info.serial_number),
            'firmware': dev.get_info(rs.camera_info.firmware_version),
            'usb_type': dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else 'Unknown',
        }
        cameras.append(info)

    return cameras


def test_streams(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: int = 10,
    display: bool = True,
    target_width: int = 320,
    target_height: int = 240,
    rgb_only: bool = False,
):
    """
    Test camera RGB and Depth streams

    Args:
        width, height: Capture resolution
        fps: Target FPS
        duration: Test duration in seconds
        display: Show live preview
        target_width, target_height: Resize target for training (from config)
        rgb_only: Only enable RGB stream (no depth)
    """
    print("\n" + "="*60)
    print("         REALSENSE CAMERA STREAM TEST")
    print("="*60)

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable RGB stream
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    # Enable Depth stream (optional)
    if not rgb_only:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    else:
        print("\n*** RGB ONLY MODE (depth disabled) ***")

    print(f"\nRequested config:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Target resize: {target_width}x{target_height}")
    print(f"  Depth enabled: {not rgb_only}")

    # Start pipeline with hardware reset
    try:
        # Reset device first to clear any stuck state
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            print("\nResetting camera...")
            devices[0].hardware_reset()
            time.sleep(2)  # Wait for reset

        profile = pipeline.start(config)
    except Exception as e:
        print(f"\nFailed to start pipeline: {e}")
        print("\nTroubleshooting:")
        print("  1. Unplug and replug USB cable")
        print("  2. Check USB 3.0 connection (blue port)")
        print("  3. Try: python scripts/test_camera.py --rgb-only")
        return False

    # Get device info
    device = profile.get_device()
    print(f"\nCamera: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")

    # Get stream profiles
    color_profile = profile.get_stream(rs.stream.color)

    # Get intrinsics
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    print(f"\n--- RGB Stream Intrinsics ---")
    print(f"  Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
    print(f"  Principal point: ({color_intrinsics.ppx:.2f}, {color_intrinsics.ppy:.2f})")
    print(f"  Focal length: ({color_intrinsics.fx:.2f}, {color_intrinsics.fy:.2f})")
    print(f"  Distortion model: {color_intrinsics.model}")
    print(f"  Distortion coeffs: {color_intrinsics.coeffs}")

    # Get depth info if enabled
    align = None
    if not rgb_only:
        depth_profile = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        print(f"\n--- Depth Stream Intrinsics ---")
        print(f"  Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")

        # Get depth scale
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"  Depth scale: {depth_scale} (multiply by depth value to get meters)")

        # Align depth to color
        align = rs.align(rs.stream.color)

    # FPS tracking
    frame_times = []
    frame_count = 0
    start_time = time.time()

    print(f"\n--- Testing streams for {duration} seconds ---")
    print("Press Ctrl+C to stop early\n")

    # Import cv2 for display if needed
    cv2 = None
    if display:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
            print("OpenCV available - showing live preview")
            print("Press 'q' to quit, 's' to save snapshot\n")
        except ImportError:
            print("OpenCV not available - skipping live preview")
            display = False

    try:
        while time.time() - start_time < duration:
            # Wait for frames with timeout
            t_frame_start = time.time()
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError as e:
                print(f"\nFrame timeout: {e}")
                print("Retrying...")
                continue

            t_frame_received = time.time()

            # Get frames
            if rgb_only:
                color_frame = frames.get_color_frame()
                depth_image = None
                if not color_frame:
                    continue
            else:
                # Align depth to color
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())  # uint16

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())  # RGB

            # Track timing
            frame_times.append(t_frame_received - t_frame_start)
            frame_count += 1

            # Print stats every 30 frames
            if frame_count % 30 == 0:
                avg_latency = np.mean(frame_times[-30:]) * 1000
                current_fps = 30 / (time.time() - start_time) * frame_count / 30 if frame_count > 30 else frame_count / (time.time() - start_time)
                elapsed = time.time() - start_time

                # Check image stats
                if rgb_only:
                    print(f"[{elapsed:5.1f}s] Frames: {frame_count:4d} | "
                          f"FPS: {current_fps:5.1f} | "
                          f"Latency: {avg_latency:5.1f}ms | "
                          f"RGB shape: {color_image.shape}")
                else:
                    print(f"[{elapsed:5.1f}s] Frames: {frame_count:4d} | "
                          f"FPS: {current_fps:5.1f} | "
                          f"Latency: {avg_latency:5.1f}ms | "
                          f"RGB shape: {color_image.shape} | "
                          f"Depth range: [{depth_image.min()}-{depth_image.max()}]")

            # Display if available
            if display and cv2 is not None:
                # Convert RGB to BGR for OpenCV
                color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                # Resize to target size for preview
                color_resized = cv2.resize(color_bgr, (target_width, target_height))

                if rgb_only:
                    # RGB only mode
                    combined = color_resized
                    cv2.putText(combined, f"RGB {target_width}x{target_height}", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # RGB + Depth mode
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET
                    )
                    depth_resized = cv2.resize(depth_colormap, (target_width, target_height))

                    # Stack horizontally
                    combined = np.hstack([color_resized, depth_resized])

                    # Add text
                    cv2.putText(combined, f"RGB {target_width}x{target_height}", (10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined, f"Depth {target_width}x{target_height}", (target_width + 10, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('RealSense Test (Press q to quit, s to save)', combined)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = int(time.time())
                    cv2.imwrite(f'snapshot_rgb_{timestamp}.png', color_bgr)
                    cv2.imwrite(f'snapshot_depth_{timestamp}.png', depth_colormap)
                    print(f"Saved snapshots: snapshot_rgb_{timestamp}.png, snapshot_depth_{timestamp}.png")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        pipeline.stop()
        if display and cv2 is not None:
            cv2.destroyAllWindows()

    # Final stats
    total_time = time.time() - start_time
    actual_fps = frame_count / total_time
    avg_latency = np.mean(frame_times) * 1000 if frame_times else 0

    print("\n" + "="*60)
    print("         TEST RESULTS")
    print("="*60)
    print(f"\nTotal frames: {frame_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Actual FPS: {actual_fps:.2f} (target: {fps})")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Min latency: {min(frame_times)*1000:.2f}ms")
    print(f"Max latency: {max(frame_times)*1000:.2f}ms")

    # Check if FPS is acceptable
    fps_ok = actual_fps >= fps * 0.9  # Allow 10% tolerance
    latency_ok = avg_latency < 100  # Less than 100ms

    print(f"\n--- Data for Diffusion Policy ---")
    print(f"RGB image shape: {color_image.shape} -> resize to (3, {target_height}, {target_width})")
    print(f"Depth available: {'No (rgb_only mode)' if rgb_only else 'Yes (can be used for additional observation)'}")
    print(f"FPS sufficient: {'✅ YES' if fps_ok else '❌ NO - may need to reduce resolution'}")
    print(f"Latency acceptable: {'✅ YES' if latency_ok else '❌ NO - check USB 3.0 connection'}")

    return fps_ok and latency_ok


@click.command()
@click.option('--width', default=1280, help='Capture width')
@click.option('--height', default=720, help='Capture height')
@click.option('--fps', default=30, help='Target FPS')
@click.option('--duration', default=10, help='Test duration in seconds')
@click.option('--display/--no-display', default=True, help='Show live preview')
@click.option('--target-width', default=320, help='Target resize width for training')
@click.option('--target-height', default=240, help='Target resize height for training')
@click.option('--rgb-only', is_flag=True, help='Only test RGB stream (no depth)')
def main(width, height, fps, duration, display, target_width, target_height, rgb_only):
    print("="*60)
    print("         REALSENSE CAMERA TEST")
    print("="*60)

    # Get camera info
    cameras = get_camera_info()

    if cameras is None or len(cameras) == 0:
        print("\n❌ No RealSense camera found!")
        print("\nTroubleshooting:")
        print("  1. Check USB connection (USB 3.0 required for D455)")
        print("  2. Try different USB port")
        print("  3. Run: realsense-viewer to verify camera")
        return

    print(f"\nFound {len(cameras)} camera(s):")
    for i, cam in enumerate(cameras):
        print(f"\n  [{i}] {cam['name']}")
        print(f"      Serial: {cam['serial']}")
        print(f"      Firmware: {cam['firmware']}")
        print(f"      USB Type: {cam['usb_type']}")

        if '2.' in cam['usb_type']:
            print(f"      ⚠️  WARNING: USB 2.0 detected! Use USB 3.0 for best performance")

    # Test streams
    success = test_streams(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        display=display,
        target_width=target_width,
        target_height=target_height,
        rgb_only=rgb_only,
    )

    print("\n" + "="*60)
    if success:
        print("✅ CAMERA TEST PASSED - Ready for data collection!")
    else:
        print("⚠️  CAMERA TEST WARNING - Check issues above")
    print("="*60)


if __name__ == '__main__':
    main()
