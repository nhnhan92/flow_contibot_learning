#!/usr/bin/env python3
"""
RealSense Camera Interface

Wrapper around pyrealsense2 for easier use.
"""

import numpy as np
import pyrealsense2 as rs
import cv2

class RealSenseCamera:
    """RealSense D455 Camera Interface"""

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        serial_number=None,
        enable_depth=False,
    ):
        """
        Initialize RealSense camera

        Args:
            width: Image width (default 640)
            height: Image height (default 480)
            fps: Frame rate (default 30)
            serial_number: Camera serial number (None = use any available)
            enable_depth: Enable depth stream (default True)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth

        # Create pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable specific device if serial provided
        if serial_number is not None:
            self.config.enable_device(serial_number)

        # Configure streams
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)  #bgr8

        if enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Start streaming
        print(f"Starting RealSense camera ({width}×{height} @ {fps}fps)...")
        try:
            self.profile = self.pipeline.start(self.config)

            # Warm up - skip first few frames
            for _ in range(10):
                self.pipeline.wait_for_frames()
        except Exception as e:
            raise RuntimeError(f"Failed to start camera: {e}")
        # print("  ✅ Camera ready!")

        # Get device info
        device = self.profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)
        print(f"✅ Camera started! Serial: {self.serial}")

    def get_frames(self):
        """
        Get color and depth frames

        Returns:
            color: np.array (H, W, 3) BGR color image
            depth: np.array (H, W) depth image in mm (or None if depth disabled)
        """
        # Wait for frames
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)

        # Get color frame
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to get color frame")

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Get depth frame if enabled
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError("Failed to get depth frame")
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            depth_image = None

        return color_image, depth_image

    def get_color_frame(self):
        """
        Get only color frame (faster if depth not needed)

        Returns:
            np.array: (H, W, 3) BGR color image
        """
        color, _ = self.get_frames()
        return color

    def get_intrinsics(self):
        """
        Get camera intrinsics

        Returns:
            dict with keys: width, height, fx, fy, ppx, ppy, coeffs
        """
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()

        return {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'coeffs': intrinsics.coeffs,
        }

    def stop(self):
        """Stop camera streaming"""
        self.pipeline.stop()
        print("Camera stopped")

# Alias for compatibility
RealSense = RealSenseCamera
