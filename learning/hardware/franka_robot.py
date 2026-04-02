#!/usr/bin/env python3
"""
Franka Panda Robot Control via franky.

Wrapper around franky (https://github.com/TimSchneider42/franky) providing
the same interface as UR5eRobot so demo_collect.py can swap arms with a
single flag.

Install:
    pip install franky-control

Notes:
    - Franka has 7 joints (UR5 has 6); joint_state is (7,).
    - TCP pose is (6,): [x, y, z, rx, ry, rz] in metres / radians (same as UR5).
    - The robot must be unlocked and FCI enabled before connecting.
      (Desk → Activate FCI, or use franka_control_node if using ROS)
    - Requires libfranka >= 0.16.0 and a PREEMPT_RT kernel for real-time control.
"""

import numpy as np
import time

try:
    from franky import Robot, Affine, RobotPose, JointMotion, CartesianMotion, ReferenceType
    _FRANKY_AVAILABLE = True
except ImportError:
    _FRANKY_AVAILABLE = False
    print("[franka] WARNING: franky not installed. Install with: pip install franky-control")


def _robotpose_to_pose(robot_pose) -> np.ndarray:
    """Convert franky RobotPose → (6,) [x, y, z, rx, ry, rz] in m/rad."""
    import scipy.spatial.transform as st
    aff = robot_pose.end_effector_pose          # franky Affine (SE3)
    pos = np.array(aff.translation(), dtype=np.float32)
    rot = st.Rotation.from_matrix(
        np.array(aff.rotation_matrix())
    ).as_rotvec()
    return np.concatenate([pos, rot]).astype(np.float32)


def _pose_to_robotpose(pose: np.ndarray) -> "RobotPose":
    """Convert (6,) [x, y, z, rx, ry, rz] → franky RobotPose."""
    import scipy.spatial.transform as st
    translation = pose[:3].tolist()
    # franky Affine: Affine(translation, quaternion [x, y, z, w])
    quat = st.Rotation.from_rotvec(pose[3:]).as_quat().tolist()  # [x,y,z,w]
    aff = Affine(translation, quat)
    return RobotPose(aff)


class FrankaRobot:
    """
    Franka Panda control via franky.

    Provides the same interface as UR5eRobot:
        get_tcp_pose()       → np.ndarray (6,)  [x,y,z,rx,ry,rz] m/rad
        get_joint_angles()   → np.ndarray (7,)  joint angles rad
        servo_tcp_pose()     → real-time Cartesian servo
        move_tcp_pose()      → blocking linear move
        move_joints()        → blocking joint move
        disconnect()         → cleanup
    """

    def __init__(self, robot_ip: str = "172.16.0.2", frequency: float = 10.0,
                 dynamic_rel: float = 0.2):
        """
        Parameters
        ----------
        robot_ip    : Franka FCI IP (default 172.16.0.2)
        frequency   : Control loop frequency (Hz); used to compute dt.
        dynamic_rel : Relative dynamics scaling (0–1). Lower = slower/safer.
        """
        if not _FRANKY_AVAILABLE:
            raise ImportError("franky is required. Install with: pip install franky-control")

        self.robot_ip    = robot_ip
        self.frequency   = frequency
        self.dt          = 1.0 / frequency
        self.dynamic_rel = dynamic_rel

        print(f"Connecting to Franka at {robot_ip} ...")
        self._robot = Robot(robot_ip)
        self._robot.recover_from_errors()
        # franky exposes dynamics scaling as a property (replaces set_dynamic_rel)
        self._robot.relative_dynamics_factor = dynamic_rel
        print("Franka connected!")

    # ── State ─────────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """Return current TCP pose as (6,) [x, y, z, rx, ry, rz] m/rad."""
        robot_pose = self._robot.current_cartesian_state.pose
        return _robotpose_to_pose(robot_pose)

    def get_joint_angles(self) -> np.ndarray:
        """Return current joint angles as (7,) rad."""
        return np.array(self._robot.current_joint_state.position, dtype=np.float32)

    # ── Motion ────────────────────────────────────────────────────────────────

    def move_tcp_pose(self, target_pose, velocity: float = 0.1,
                      acceleration: float = 0.1, asynchronous: bool = False):
        """
        Blocking (or async) linear move to target TCP pose.

        Parameters
        ----------
        target_pose  : (6,) [x, y, z, rx, ry, rz] m/rad
        velocity     : Scales relative_dynamics_factor (0–1 range; higher = faster).
        acceleration : Scales relative_dynamics_factor (0–1 range; higher = faster).
        asynchronous : If True return immediately.
        """
        target_pose  = np.asarray(target_pose, dtype=float)
        robot_pose   = _pose_to_robotpose(target_pose)
        motion       = CartesianMotion(robot_pose, ReferenceType.Absolute)

        # Temporarily scale dynamics; restore afterward
        prev_dyn = self._robot.relative_dynamics_factor
        self._robot.relative_dynamics_factor = float(np.clip(
            min(velocity, acceleration), 0.01, 1.0
        ))
        try:
            if asynchronous:
                self._robot.move_async(motion)
            else:
                self._robot.move(motion)
        finally:
            self._robot.relative_dynamics_factor = prev_dyn

    def servo_tcp_pose(self, target_pose, velocity: float = 0.1,
                       acceleration: float = 0.1, dt: float = None):
        """
        High-frequency Cartesian servo step (mirrors UR5e servo_tcp_pose API).

        Sends an incremental relative motion toward target_pose.
        Called every control tick.

        Parameters
        ----------
        target_pose  : (6,) [x, y, z, rx, ry, rz] m/rad  (absolute target)
        velocity     : Max translation step per tick (m/s × dt).
        acceleration : Max rotation step per tick (rad/s × dt).
        dt           : Time step (s); defaults to self.dt.
        """
        if dt is None:
            dt = self.dt

        target_pose  = np.asarray(target_pose, dtype=float)
        current_pose = self.get_tcp_pose()

        # Clip positional displacement to max step size
        delta_pos = target_pose[:3] - current_pose[:3]
        delta_pos_clipped = np.clip(delta_pos, -velocity * dt, velocity * dt)

        import scipy.spatial.transform as st
        r_cur    = st.Rotation.from_rotvec(current_pose[3:])
        r_tgt    = st.Rotation.from_rotvec(target_pose[3:])
        r_rel    = r_tgt * r_cur.inv()
        rotvec   = r_rel.as_rotvec()
        rotvec_clipped = np.clip(rotvec, -acceleration * dt, acceleration * dt)

        delta_quat = st.Rotation.from_rotvec(rotvec_clipped).as_quat().tolist()
        delta_aff  = Affine(delta_pos_clipped.tolist(), delta_quat)
        delta_pose = RobotPose(delta_aff)

        try:
            self._robot.move(CartesianMotion(delta_pose, ReferenceType.Relative))
        except Exception:
            self._robot.recover_from_errors()

    def move_joints(self, target_joints, velocity: float = 0.5,
                    acceleration: float = 1.0, asynchronous: bool = False):
        """
        Move to target joint configuration.

        Parameters
        ----------
        target_joints : (7,) target joint angles in rad
        """
        target_joints = list(np.asarray(target_joints, dtype=float))
        motion = JointMotion(target_joints)
        if asynchronous:
            self._robot.move_async(motion)
        else:
            self._robot.move(motion)

    def stop(self):
        """Stop any ongoing motion."""
        try:
            self._robot.stop()
        except Exception:
            pass

    def recover(self):
        """Recover from robot errors (e.g. after a collision)."""
        self._robot.recover_from_errors()
        print("[franka] Recovered from errors.")

    def disconnect(self):
        """Disconnect from Franka."""
        try:
            self.stop()
        except Exception:
            pass
        print("Franka disconnected.")
