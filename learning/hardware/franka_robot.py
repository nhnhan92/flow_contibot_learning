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
    from franky import (
        Robot, Affine, RobotPose, JointMotion, CartesianMotion, ReferenceType,
        CartesianState, Twist,
    )
    _FRANKY_AVAILABLE = True
except ImportError:
    _FRANKY_AVAILABLE = False
    print("[franka] WARNING: franky not installed. Install with: pip install franky-control")


def _robotpose_to_pose(robot_pose) -> np.ndarray:
    """Convert franky RobotPose → (6,) [x, y, z, rx, ry, rz] in m/rad."""
    import scipy.spatial.transform as st
    aff = robot_pose.end_effector_pose          # franky Affine (SE3)
    T = np.asarray(aff.matrix, dtype=np.float64)  # (4,4) homogeneous transform (property, not a method)
    pos = T[:3, 3].astype(np.float32)
    rot = st.Rotation.from_matrix(T[:3, :3]).as_rotvec()
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

    Plus Franka-specific extras (no UR5e equivalent):
        get_ee_wrench()             → np.ndarray (6,) estimated EE force/torque
        get_joint_external_torques() → np.ndarray (7,) estimated joint torques
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

    def get_ee_wrench(self, frame: str = "base") -> np.ndarray:
        """
        Return Franka's built-in estimated external wrench at the EE.

        This comes from the arm's joint-torque sensing (no external F/T
        sensor needed) -- libfranka's O_F_ext_hat_K / K_F_ext_hat_K.

        Parameters
        ----------
        frame : "base" -> wrench expressed in the robot base frame (O_F_ext_hat_K)
                "ee"   -> wrench expressed in the stiffness/EE frame (K_F_ext_hat_K)

        Returns
        -------
        np.ndarray (6,) [Fx, Fy, Fz, Tx, Ty, Tz] in N / N*m.
        """
        state = self._robot.state
        field = state.O_F_ext_hat_K if frame == "base" else state.K_F_ext_hat_K
        return np.asarray(field, dtype=np.float32).reshape(6)

    def get_joint_external_torques(self) -> np.ndarray:
        """Return filtered estimated external joint torques (7,) in N*m."""
        return np.asarray(self._robot.state.tau_ext_hat_filtered, dtype=np.float32).reshape(7)

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
                       acceleration: float = 0.1, dt: float = None,
                       lookahead_time: float = None, gain: float = None):
        """
        High-frequency Cartesian servo step (mirrors UR5e servo_tcp_pose API).

        franky has no raw per-cycle streaming primitive like RTDE's servoL --
        every move() plans a fresh trajectory, which by default targets zero
        velocity at its end (see franky docs: "the robot will stop at each
        waypoint unless you specify a target velocity"). Calling that once
        per control tick made the arm brake to a full stop every tick, which
        is what showed up as jerky/pulsing motion.

        To avoid that, this tick's (clipped, safety-bounded) step is issued
        as an absolute CartesianState carrying a non-zero feed-forward
        velocity toward target_pose, with return_when_finished=False so this
        call does not block waiting for that open-ended motion to finish --
        the next servo_tcp_pose() call is expected to supersede it before it
        would naturally stop.

        NOT verified on hardware -- start with small velocity/acceleration
        and watch closely; franky raises a discontinuity error if successive
        motions' dynamics are too aggressive to hand off smoothly.

        Parameters
        ----------
        target_pose  : (6,) [x, y, z, rx, ry, rz] m/rad  (absolute target)
        velocity     : Max translation step per tick (m/s × dt); also the
                       feed-forward linear speed given to this tick's motion.
        acceleration : Max rotation step per tick (rad/s × dt); also the
                       feed-forward angular speed (reused loosely, same
                       convention as elsewhere in this class).
        dt           : Time step (s); defaults to self.dt.
        lookahead_time, gain : Accepted for signature compatibility with
                      UR5eRobot.servo_tcp_pose (RTDE trajectory-smoothing
                      params); unused here, franky has no equivalent.
        """
        if dt is None:
            dt = self.dt

        target_pose  = np.asarray(target_pose, dtype=float)
        current_pose = self.get_tcp_pose()

        import scipy.spatial.transform as st

        # Clip this tick's step the same way as before (bounds how far a
        # single call can move the arm, e.g. against a noisy/jumpy caller).
        delta_pos = target_pose[:3] - current_pose[:3]
        delta_pos_clipped = np.clip(delta_pos, -velocity * dt, velocity * dt)
        step_pos = current_pose[:3] + delta_pos_clipped

        r_cur  = st.Rotation.from_rotvec(current_pose[3:])
        r_tgt  = st.Rotation.from_rotvec(target_pose[3:])
        rotvec = (r_tgt * r_cur.inv()).as_rotvec()
        rotvec_clipped = np.clip(rotvec, -acceleration * dt, acceleration * dt)
        step_rot = st.Rotation.from_rotvec(rotvec_clipped) * r_cur

        # Feed-forward velocity toward the (unclipped) target so this tick's
        # motion doesn't target zero velocity -- that's what caused the stop
        # every tick.
        dist = float(np.linalg.norm(delta_pos))
        lin_vel = (delta_pos / dist * velocity) if dist > 1e-9 else np.zeros(3)
        angle = float(np.linalg.norm(rotvec))
        ang_vel = (rotvec / angle * acceleration) if angle > 1e-9 else np.zeros(3)

        step_affine = Affine(step_pos.tolist(), step_rot.as_quat().tolist())
        step_state = CartesianState(RobotPose(step_affine), Twist(lin_vel.tolist(), ang_vel.tolist()))

        # No relative_dynamics_factor here: it would multiply with
        # self._robot.relative_dynamics_factor (set once in __init__), and
        # that stacking made the effective velocity ceiling far smaller than
        # the feed-forward `velocity`/`acceleration` targets above -- Ruckig
        # rejected every tick as infeasible (error -100 / ErrorInvalidInput).
        # The physical step is already bounded by the explicit dt-based clip
        # above, so a second dynamics-scaling layer here is both redundant
        # and was the actual bug.
        motion = CartesianMotion(
            step_state,
            ReferenceType.Absolute,
            return_when_finished=False,
        )

        try:
            self._robot.move(motion)
        except Exception as e:
            print(f"[franka] servo_tcp_pose motion error: {e}")
            self._robot.recover_from_errors()

    def move_joints(self, target_joints, velocity: float = 0.5,
                    acceleration: float = 1.0, asynchronous: bool = False):
        """
        Move to target joint configuration.

        Parameters
        ----------
        target_joints : (7,) target joint angles in rad
        velocity      : Scales relative_dynamics_factor (0-1 range; higher = faster).
        acceleration  : Scales relative_dynamics_factor (0-1 range; higher = faster).
        """
        target_joints = list(np.asarray(target_joints, dtype=float))
        dyn_factor = float(np.clip(min(velocity, acceleration), 0.01, 1.0))
        motion = JointMotion(target_joints, relative_dynamics_factor=dyn_factor)
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
