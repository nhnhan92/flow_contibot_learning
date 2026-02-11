#!/usr/bin/env python3
"""
Debug camera view during deployment

Usage:
    python deploy/debug_camera_view.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from custom.realsense_camera import RealSenseCamera
from custom.ur5e_rtde import UR5eRobot

def main():
    print("="*60)
    print("   CAMERA VIEW DEBUG")
    print("="*60)

    # Initialize camera
    print("\nInitializing camera...")
    camera = RealSenseCamera()

    # Initialize robot to get pose
    print("Connecting to robot...")
    robot = UR5eRobot("192.168.11.20")

    # Get current state
    pose = robot.get_tcp_pose()
    print(f"\nCurrent robot pose:")
    print(f"  Position: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}]")
    print(f"  Rotation: [{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}]")

    print("\nPress 'q' to quit, 's' to save image")
    print("Showing camera view...")

    save_counter = 0

    while True:
        # Get frame
        color, depth = camera.get_frames()

        # Resize to model input size for comparison (128x128 to match training)
        color_small = cv2.resize(color, (128, 128))

        # Display both - resize small image to match height
        color_small_display = cv2.resize(color_small, (480, 480), interpolation=cv2.INTER_NEAREST)
        display = np.hstack([color, color_small_display])

        # Add text
        cv2.putText(display, "Full resolution (left) | Model input 128x128 (right)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Pose: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Camera View", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"debug_image_{save_counter:03d}.png"
            cv2.imwrite(filename, color)
            print(f"Saved: {filename}")
            save_counter += 1

    camera.stop()
    robot.disconnect()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
