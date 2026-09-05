#!/usr/bin/env python3
"""
Test RealSense Camera for Diffusion Policy Data Collection

Usage:
    cd ~/Desktop/flow_contibot_learning/learning
    python system_verification/test_camera.py
    python system_verification/test_camera.py --no-display

    # Dual-camera mode: verify both the global and wrist RealSense cameras
    # can be opened and read from at the same time, using the exact same
    # connection path (hardware/realsense_camera.py's RealSenseCamera,
    # same width/height/fps) that demo_collect.py uses -- run this before
    # trusting demo_collect.py's two-camera recording on new hardware.
    python system_verification/test_camera.py --dual \
        --camera_serial_global <serial> --camera_serial_wrist <serial>
"""

import os
import sys
import time
import click
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed. Run: pip install pyrealsense2")
    exit(1)

SYSVER_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(SYSVER_DIR)
sys.path.insert(0, LEARNING_DIR)


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


def test_dual_streams(
    serial_global: str | None,
    serial_wrist: str | None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    duration: int = 10,
    display: bool = True,
):
    """
    Verify the global and wrist RealSense cameras can both be opened and
    read from at the same time, using the exact connection path
    demo_collect.py uses (hardware/realsense_camera.py's RealSenseCamera,
    same width/height/fps, one get_frames() call per camera per tick) --
    not a reimplementation with raw pyrealsense2 like test_streams() above.

    Catches problems specific to running two RealSense devices in one
    process that a single-camera test can't: USB bandwidth contention
    (starves one or both cameras' FPS), ambiguous serial_number=None
    binding when both cameras are left unspecified, and outright failure
    to open two pipelines concurrently.
    """
    from hardware.realsense_camera import RealSenseCamera

    print("\n" + "="*60)
    print("      DUAL-CAMERA (GLOBAL + WRIST) STREAM TEST")
    print("="*60)
    print(f"\nRequested config: {width}x{height} @ {fps}fps, both cameras")
    if serial_global is None or serial_wrist is None:
        print("⚠️  One or both serials not given -- which physical camera binds to which")
        print("   role is then up to enumeration order, not guaranteed stable across runs.")
        print("   Pass --camera_serial_global/--camera_serial_wrist (see get_camera_info() output above).")

    print("\nConnecting global camera...")
    try:
        cam_global = RealSenseCamera(serial_number=serial_global, width=width, height=height,
                                      fps=fps, enable_depth=False)
    except Exception as e:
        print(f"❌ Global camera failed to start: {e}")
        return False

    print("\nConnecting wrist camera...")
    try:
        cam_wrist = RealSenseCamera(serial_number=serial_wrist, width=width, height=height,
                                     fps=fps, enable_depth=False)
    except Exception as e:
        print(f"❌ Wrist camera failed to start: {e}")
        cam_global.stop()
        return False

    if cam_global.serial == cam_wrist.serial:
        print(f"\n❌ Both pipelines bound to the SAME physical camera (serial {cam_global.serial})! "
              "Pass distinct --camera_serial_global/--camera_serial_wrist.")
        cam_global.stop()
        cam_wrist.stop()
        return False

    cv2 = None
    if display:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
        except ImportError:
            print("OpenCV not available - skipping live preview")
            display = False

    times_global, times_wrist = [], []
    n_global, n_wrist = 0, 0
    frame_global, frame_wrist = None, None
    print(f"\n--- Reading both cameras for {duration}s (sequential get_frames(), matching demo_collect.py) ---")
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            t0 = time.time()
            try:
                frame_global, _ = cam_global.get_frames()
                times_global.append(time.time() - t0)
                n_global += 1
            except Exception as e:
                print(f"\n⚠️  Global camera read error: {e}")

            t1 = time.time()
            try:
                frame_wrist, _ = cam_wrist.get_frames()
                times_wrist.append(time.time() - t1)
                n_wrist += 1
            except Exception as e:
                print(f"\n⚠️  Wrist camera read error: {e}")

            elapsed = time.time() - start_time
            if (n_global + n_wrist) % 20 == 0:
                fps_g = n_global / elapsed if elapsed > 0 else 0
                fps_w = n_wrist / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:5.1f}s] global: {n_global:4d} frames ({fps_g:5.1f} fps) | "
                      f"wrist: {n_wrist:4d} frames ({fps_w:5.1f} fps)", end="\r")

            if display and cv2 is not None and frame_global is not None and frame_wrist is not None:
                g_bgr = cv2.cvtColor(frame_global, cv2.COLOR_RGB2BGR)
                w_bgr = cv2.cvtColor(frame_wrist, cv2.COLOR_RGB2BGR)
                cv2.putText(g_bgr, "GLOBAL", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(w_bgr, "WRIST", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                combined = np.hstack([g_bgr, w_bgr])
                cv2.imshow("Dual Camera Test (press q to quit)", combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cam_global.stop()
        cam_wrist.stop()
        if display and cv2 is not None:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    fps_global = n_global / total_time if total_time > 0 else 0
    fps_wrist = n_wrist / total_time if total_time > 0 else 0
    lat_global = np.mean(times_global) * 1000 if times_global else 0
    lat_wrist = np.mean(times_wrist) * 1000 if times_wrist else 0

    print("\n\n" + "="*60)
    print("         DUAL-CAMERA TEST RESULTS")
    print("="*60)
    print(f"Global camera: {n_global} frames, {fps_global:.1f} fps (target {fps}), "
          f"{lat_global:.1f}ms avg latency")
    print(f"Wrist camera:  {n_wrist} frames, {fps_wrist:.1f} fps (target {fps}), "
          f"{lat_wrist:.1f}ms avg latency")

    fps_ok = fps_global >= fps * 0.8 and fps_wrist >= fps * 0.8  # looser tolerance -- two cameras share USB bandwidth
    print(f"\nBoth cameras sustained close to target FPS while read concurrently: "
          f"{'✅ YES' if fps_ok else '❌ NO'}")
    if not fps_ok:
        print("  If one or both are well under target, check they're on separate USB")
        print("  controllers/hubs -- two color streams at this resolution/fps can exceed")
        print("  a single USB 3.0 controller's bandwidth.")

    return fps_ok


@click.command()
@click.option('--width', default=640, help='Capture width (single-camera mode). '
              'Dual mode (--dual) defaults to 640 instead, matching demo_collect.py, unless overridden.')
@click.option('--height', default=480, help='Capture height (single-camera mode). '
              'Dual mode (--dual) defaults to 480 instead, matching demo_collect.py, unless overridden.')
@click.option('--fps', default=30, help='Target FPS')
@click.option('--duration', default=10, help='Test duration in seconds')
@click.option('--display/--no-display', default=True, help='Show live preview')
@click.option('--target-width', default=320, help='Target resize width for training')
@click.option('--target-height', default=240, help='Target resize height for training')
@click.option('--rgb-only', is_flag=True, help='Only test RGB stream (no depth)')
@click.option('--dual', is_flag=True, help='Test the global + wrist cameras together '
              '(same connection path as demo_collect.py) instead of the default single-camera test.')
@click.option('--camera_serial_global', default='841512070635', help='Serial for the global camera (--dual only).')
@click.option('--camera_serial_wrist', default='827112072398', help='Serial for the wrist camera (--dual only).')
@click.pass_context
def main(ctx, width, height, fps, duration, display, target_width, target_height, rgb_only,
         dual, camera_serial_global, camera_serial_wrist):
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

    if dual:
        if len(cameras) < 2:
            print(f"\n❌ --dual requires 2 connected cameras, found {len(cameras)}.")
            return
        # Match demo_collect.py's own defaults (640x480) unless the user
        # explicitly overrode --width/--height for this run.
        src = click.core.ParameterSource
        if ctx.get_parameter_source('width') == src.DEFAULT:
            width = 640
        if ctx.get_parameter_source('height') == src.DEFAULT:
            height = 480

        success = test_dual_streams(
            serial_global=camera_serial_global,
            serial_wrist=camera_serial_wrist,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            display=display,
        )
        print("\n" + "="*60)
        print("✅ DUAL-CAMERA TEST PASSED" if success else "⚠️  DUAL-CAMERA TEST WARNING - Check issues above")
        print("="*60)
        return

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
