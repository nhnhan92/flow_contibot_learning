#!/usr/bin/env python3
"""
UR5e Robot Control via RTDE (Real-Time Data Exchange)

Wrapper around rtde_receive and rtde_control for easier use.
"""

import numpy as np
import time
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


class UR5eRobot:
    """UR5e Robot control via RTDE"""

    def __init__(self, robot_ip="192.168.1.102", frequency=500):
        """
        Initialize robot connection

        Args:
            robot_ip: Robot IP address
            frequency: RTDE frequency (Hz), default 500
        """
        self.robot_ip = robot_ip
        self.frequency = frequency

        print(f"Connecting to robot at {robot_ip}...")

        # Initialize RTDE interfaces
        self.rtde_c = RTDEControlInterface(robot_ip, frequency)
        self.rtde_r = RTDEReceiveInterface(robot_ip, frequency)

        print(f"✅ Robot connected!")

    def get_tcp_pose(self):
        """
        Get current TCP (tool center point) pose

        Returns:
            np.array: (6,) [x, y, z, rx, ry, rz] in meters and radians
        """
        pose = self.rtde_r.getActualTCPPose()
        return np.array(pose, dtype=np.float32)

    def get_joint_angles(self):
        """
        Get current joint angles

        Returns:
            np.array: (6,) joint angles in radians
        """
        joints = self.rtde_r.getActualQ()
        return np.array(joints, dtype=np.float32)

    def move_tcp_pose(self, target_pose, velocity=0.5, acceleration=1.0, asynchronous=False):
        """
        Move TCP to target pose

        Args:
            target_pose: (6,) [x, y, z, rx, ry, rz] target pose
            velocity: velocity in m/s (default 0.5)
            acceleration: acceleration in m/s^2 (default 1.0)
            asynchronous: if True, return immediately without waiting
        """
        target_pose = list(target_pose)

        if asynchronous:
            # Start movement without blocking
            self.rtde_c.moveL(target_pose, velocity, acceleration, asynchronous=True)
        else:
            # Block until movement complete
            self.rtde_c.moveL(target_pose, velocity, acceleration)

    def servo_tcp_pose(self, target_pose, dt=0.1, lookahead_time=0.1, gain=300):
        """
        Servo control to target pose (for high-frequency absolute pose control)

        This is the correct method for Diffusion Policy deployment with absolute poses!

        Args:
            target_pose: (6,) [x, y, z, rx, ry, rz] target pose
            dt: time step for servo control (should match control frequency)
            lookahead_time: lookahead time for trajectory smoothing
            gain: servo gain (higher = more responsive, but less smooth)
        """
        target_pose = list(target_pose)
        # servoL signature: servoL(pose, velocity, acceleration, dt, lookahead_time, gain)
        self.rtde_c.servoL(target_pose, 0, 0, dt, lookahead_time, gain)

    def move_joints(self, target_joints, velocity=0.5, acceleration=1.0, asynchronous=False):
        """
        Move to target joint configuration

        Args:
            target_joints: (6,) target joint angles in radians
            velocity: velocity in rad/s (default 0.5)
            acceleration: acceleration in rad/s^2 (default 1.0)
            asynchronous: if True, return immediately without waiting
        """
        target_joints = list(target_joints)

        if asynchronous:
            self.rtde_c.moveJ(target_joints, velocity, acceleration, asynchronous=True)
        else:
            self.rtde_c.moveJ(target_joints, velocity, acceleration)

    def stop(self):
        """Emergency stop - stop all movement immediately"""
        try:
            self.rtde_c.servoStop()  # Stop servo mode first
        except:
            pass
        self.rtde_c.stopScript()

    def get_robot_mode(self):
        """
        Get robot mode

        Returns:
            int: Robot mode
                -1: NO_CONTROLLER
                0: DISCONNECTED
                1: CONFIRM_SAFETY
                2: BOOTING
                3: POWER_OFF
                4: POWER_ON
                5: IDLE
                6: BACKDRIVE
                7: RUNNING
                8: UPDATING_FIRMWARE
        """
        return self.rtde_r.getRobotMode()

    def is_running(self):
        """Check if robot is in running mode"""
        return self.get_robot_mode() == 7

    def is_moving(self):
        """Check if robot is currently moving"""
        # Check if TCP speed is above threshold
        tcp_speed = self.rtde_r.getActualTCPSpeed()
        speed_norm = np.linalg.norm(tcp_speed[:3])  # Linear speed
        return speed_norm > 0.001  # 1mm/s threshold

    def wait_for_motion_complete(self, timeout=10.0):
        """
        Wait until robot stops moving

        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        while self.is_moving():
            if time.time() - start_time > timeout:
                print("⚠️  Motion timeout!")
                break
            time.sleep(0.01)

    def disconnect(self):
        """Disconnect from robot"""
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        print("Robot disconnected")


# Alias for compatibility with deployment script
UR5eRobot = UR5eRobot
