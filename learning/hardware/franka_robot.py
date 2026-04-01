#!/usr/bin/env python3
"""
Franka Panda Robot Control via frankx.

Wrapper around frankx (https://github.com/pantor/frankx) providing the same
interface as UR5eRobot so demo_collect.py can swap arms with a single flag.

Install:
    pip install frankx

Notes:
    - Franka has 7 joints (UR5 has 6); joint_state is (7,).
    - TCP pose is (6,): [x, y, z, rx, ry, rz] in metres / radians (same as UR5).
    - The robot must be unlocked and FCI enabled before connecting.
      (Desk → Activate FCI, or use franka_control_node if using ROS)
"""

import numpy as np
import time

try:
    import frankx
    from frankx import Robot, Affine, JointMotion, LinearRelativeMotion, ImpedanceMotion
    _FRANKX_AVAILABLE = True
except ImportError:
    _FRANKX_AVAILABLE = False
    print("[franka] WARNING: frankx not installed. Install with: pip install frankx")


def _affine_to_pose(aff) -> np.ndarray:
    """Convert frankx Affine → (6,) [x, y, z, rx, ry, rz] in m/rad."""
    import scipy.spatial.transform as st
    pos = np.array([aff.x, aff.y, aff.z], dtype=np.float32)
    rot = st.Rotation.from_matrix(np.array(aff.rotation_matrix())).as_rotvec()
    return np.concatenate([pos, rot]).astype(np.float32)


def _pose_to_affine(pose: np.ndarray):
    """Convert (6,) [x, y, z, rx, ry, rz] → frankx Affine."""
    import scipy.spatial.transform as st
    pos = pose[:3].tolist()
    rot_mat = st.Rotation.from_rotvec(pose[3:]).as_matrix().tolist()
    return Affine(*pos, rotation=rot_mat)


class FrankaRobot:
    """
    Franka Panda control via frankx.

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
        robot_ip     : Franka FCI IP (default 172.16.0.2)
        frequency    : Control loop frequency (Hz); used to compute dt.
        dynamic_rel  : Relative dynamics scaling (0–1). Lower = slower/safer.
        """
        if not _FRANKX_AVAILABLE:
            raise ImportError("frankx is required. Install with: pip install frankx")

        self.robot_ip    = robot_ip
        self.frequency   = frequency
        self.dt          = 1.0 / frequency
        self.dynamic_rel = dynamic_rel

        print(f"Connecting to Franka at {robot_ip} ...")
        self._robot = Robot(robot_ip)
        self._robot.set_default_behavior()
        self._robot.recover_from_errors()
        self._robot.set_dynamic_rel(dynamic_rel)

        # frankx uses a motion generator; for real-time servo we keep one active
        self._servo_motion = None
        print("Franka connected!")

    # ── State ─────────────────────────────────────────────────────────────────

    def get_tcp_pose(self) -> np.ndarray:
        """Return current TCP pose as (6,) [x, y, z, rx, ry, rz] m/rad."""
        aff = self._robot.current_pose()
        return _affine_to_pose(aff)

    def get_joint_angles(self) -> np.ndarray:
        """Return current joint angles as (7,) rad."""
        state = self._robot.read_once()
        return np.array(state.q, dtype=np.float32)

    # ── Motion ────────────────────────────────────────────────────────────────

    def move_tcp_pose(self, target_pose, velocity: float = 0.1,
                      acceleration: float = 0.1, asynchronous: bool = False):
        """
        Blocking (or async) linear move to target TCP pose.

        Parameters
        ----------
        target_pose  : (6,) [x, y, z, rx, ry, rz] m/rad
        velocity     : Cartesian velocity (m/s)
        acceleration : Cartesian acceleration (m/s²)
        asynchronous : If True return immediately (frankx async motion).
        """
        target_pose = np.asarray(target_pose, dtype=float)
        target_aff  = _pose_to_affine(target_pose)

        motion = frankx.LinearMotion(target_aff, elbow=-1)
        if asynchronous:
            self._robot.move_async(motion)
        else:
            self._robot.move(motion)

    def servo_tcp_pose(self, target_pose, velocity: float = 0.1,
                       acceleration: float = 0.1, dt: float = None,
                       lookahead_time: float = 0.1, gain: float = 300):
        """
        High-frequency Cartesian servo step (mirrors UR5e servo_tcp_pose API).

        Sends an incremental linear motion toward target_pose using frankx's
        LinearRelativeMotion. Called every control tick.

        Parameters
        ----------
        target_pose   : (6,) [x, y, z, rx, ry, rz] m/rad  (absolute target)
        dt            : Time step (s); defaults to self.dt
        """
        if dt is None:
            dt = self.dt

        target_pose  = np.asarray(target_pose, dtype=float)
        current_pose = self.get_tcp_pose()

        # Compute incremental displacement
        delta_pos = target_pose[:3] - current_pose[:3]
        delta_pos_clipped = np.clip(delta_pos, -velocity * dt, velocity * dt)

        import scipy.spatial.transform as st
        r_cur  = st.Rotation.from_rotvec(current_pose[3:])
        r_tgt  = st.Rotation.from_rotvec(target_pose[3:])
        r_rel  = r_tgt * r_cur.inv()
        rotvec = r_rel.as_rotvec()
        rotvec_clipped = np.clip(rotvec, -acceleration * dt, acceleration * dt)

        delta_aff = Affine(
            delta_pos_clipped[0],
            delta_pos_clipped[1],
            delta_pos_clipped[2],
        )
        try:
            self._robot.move(LinearRelativeMotion(delta_aff))
        except frankx.InvalidOperationException:
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
        """Stop any ongoing motion (no-op if already stopped)."""
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
