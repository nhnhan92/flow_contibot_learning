#!/usr/bin/env python3
"""
Franka Panda control via franky (https://github.com/TimSchneider42/franky).

Alternative backend to FrankaController (learning/hardware/franka_control.py,
pylibfranka-based) -- same public API, so demo_collect.py can pick either
with a flag while both get evaluated on hardware.

Why this exists alongside FrankaController: FrankaController's set_ee_velocity
runs its own hand-rolled background thread that has to service libfranka's
realtime FCI loop on a strict cadence; under GIL contention with other
synchronous work in the same process (e.g. flowbot's IK + serial I/O +
matplotlib redraw) that thread can miss its deadline and trip a
"communication_constraints_violation" reflex. franky's Ruckig-based online
trajectory generation instead replans reactively on each move() call, with no
separate background thread of our own to starve -- validated in
franky_flowbot/combined_teleop/tasks/franka_task.py, which drives Franka via
spacemouse using exactly the pattern ported below:
    - set_ee_velocity() issues a fresh CartesianVelocityMotion(Twist(...))
      every call (asynchronous=True); franky/Ruckig smoothly retargets
      between calls, no manual accel-ramp loop needed.
    - stop() issues a CartesianVelocityStopMotion() (blocking), a properly
      planned deceleration, rather than the raw Robot.stop() e-stop.

Install:
    pip install franky-control

Notes:
    - Franka has 7 joints (UR5 has 6); joint_state is (7,).
    - TCP pose is (6,): [x, y, z, rx, ry, rz] in metres / radians (same as
      UR5eRobot / FrankaController).
    - The robot must be unlocked and FCI enabled before connecting.
    - Requires the same realtime-kernel setup as pylibfranka.
"""

import numpy as np
import scipy.spatial.transform as st

try:
    from franky import (
        Robot, Gripper, Affine, RobotPose, RelativeDynamicsFactor,
        CartesianMotion, CartesianVelocityMotion, CartesianVelocityStopMotion,
        JointMotion, Twist, ReferenceType,
    )
    _FRANKY_AVAILABLE = True
except ImportError:
    _FRANKY_AVAILABLE = False
    print("[FrankaRobot] WARNING: franky not installed. Install with: pip install franky-control")


def _pose6_to_affine(pose6) -> "Affine":
    """Convert (6,) [x, y, z, rx, ry, rz] rotation-vector pose to a franky Affine."""
    pose6 = np.asarray(pose6, dtype=float).reshape(6)
    quat = st.Rotation.from_rotvec(pose6[3:]).as_quat()  # [x, y, z, w]
    return Affine(pose6[:3].tolist(), quat.tolist())


def _dyn_factor(velocity: float, acceleration: float, floor: float = 0.01) -> "RelativeDynamicsFactor":
    """
    Build a franky RelativeDynamicsFactor with velocity and acceleration
    scaled independently, jerk capped to the acceleration factor.

    Collapsing to a single min(velocity, acceleration) factor (an earlier
    approach here) let a small `acceleration` value -- physical-units-style
    callers commonly pass something like 0.01 as if it were a literal
    m/s^2, not a 0-1 dynamics fraction -- silently throttle *velocity* too,
    making the whole move look stalled rather than just slow. Scaling
    velocity/acceleration separately fixes that.

    Jerk was then left fully unscaled (1.0) -- but that's its own failure
    mode: with velocity/acceleration both small (e.g. 0.02/0.1) and jerk at
    100%, Ruckig can plan a deceleration phase so abrupt the real robot's
    tracking can't keep up by the exact final control cycle, tripping
    libfranka's cartesian_motion_generator_*_discontinuity reflexes
    ("Motion finished commanded, but the robot is still moving!").
    Capping jerk to the acceleration factor keeps the rate-of-change of
    acceleration proportioned to the acceleration ceiling itself, without
    reintroducing the velocity-throttling bug above (jerk is tied to
    acceleration, not to velocity).
    """
    vel_f = float(np.clip(velocity, floor, 1.0))
    acc_f = float(np.clip(acceleration, floor, 1.0))
    return RelativeDynamicsFactor(vel_f, acc_f, acc_f)


def _affine_to_pose6(affine) -> np.ndarray:
    """Convert a franky Affine to (6,) [x, y, z, rx, ry, rz] rotation-vector pose."""
    T = np.asarray(affine.matrix, dtype=np.float64)
    pos = T[:3, 3]
    rotvec = st.Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec]).astype(np.float32)


class FrankaRobot:
    """
    Franka Panda control via franky.

    Public API mirrors FrankaController (learning/hardware/franka_control.py,
    pylibfranka-based) and UR5eRobot (learning/hardware/ur5e_rtde.py):
        get_tcp_pose()       -> np.ndarray (6,)  [x,y,z,rx,ry,rz] m/rad (rotvec)
        get_joint_angles()   -> np.ndarray (7,)  joint angles rad
        move_tcp_pose()      -> blocking (or async) point-to-point Cartesian move
        move_joints()        -> blocking (or async) point-to-point joint move
        set_ee_velocity()    -> real-time Cartesian velocity control (e.g. spacemouse teleop)
        stop()               -> stop motion, keep the connection alive
        recover()            -> automatic error recovery
        disconnect()         -> stop and release

    Plus Franka-specific extras (matches FrankaController):
        get_ee_wrench()             -> np.ndarray (6,) estimated EE force/torque
        get_joint_external_torques() -> np.ndarray (7,) estimated joint torques
        get_gripper() / gripper_move() / gripper_grasp()
    """

    def __init__(
        self,
        robot_ip: str = "172.16.0.2",
        frequency: float = 10.0,
        use_gripper: bool = False,
        do_gripper_homing: bool = False,
        dynamics_factor: float = 0.1,
        max_cart_vel: float = 0.2,
        max_ang_vel: float = 0.5,
    ):
        """
        Parameters
        ----------
        robot_ip        : Franka FCI IP (default 172.16.0.2).
        frequency       : Advisory only, for signature parity with
                           UR5eRobot/FrankaController; franky has no
                           per-cycle dt of its own to configure here.
        dynamics_factor : Robot.relative_dynamics_factor (0-1), the ceiling
                           franky/Ruckig scales its velocity/acceleration/
                           jerk limits by. Kept conservative and constant
                           through a teleop session (matches franka_task.py);
                           move_tcp_pose() additionally overrides it
                           temporarily per-call (see below).
        max_cart_vel, max_ang_vel : Default set_ee_velocity() speed caps
                           (m/s, rad/s) when a call doesn't pass its own.
        """
        if not _FRANKY_AVAILABLE:
            raise ImportError("franky is required. Install with: pip install franky-control")

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.dynamics_factor = dynamics_factor
        self.max_cart_vel = max_cart_vel
        self.max_ang_vel = max_ang_vel

        print(f"[FrankaRobot] Connecting to {robot_ip} ...")
        self._robot = Robot(robot_ip)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = dynamics_factor
        print(f"[FrankaRobot] Connected. relative_dynamics_factor={dynamics_factor}")

        self.gripper = None
        if use_gripper:
            try:
                self.gripper = Gripper(robot_ip)
                print("[FrankaRobot] Gripper connected")
                if do_gripper_homing:
                    print("[FrankaRobot] Homing gripper...")
                    ok = self.gripper.homing()
                    print(f"[FrankaRobot] Gripper homing result: {ok}")
                else:
                    print("[FrankaRobot] Gripper homing skipped")
            except Exception as e:
                self.gripper = None
                print(f"[FrankaRobot] No gripper or gripper connection failed: {e}")

    # -------------------------
    # State
    # -------------------------

    def get_tcp_pose(self) -> np.ndarray:
        """Return current TCP pose as (6,) [x, y, z, rx, ry, rz] m/rad (rotation vector)."""
        pose = self._robot.current_cartesian_state.pose.end_effector_pose
        return _affine_to_pose6(pose)

    def get_joint_angles(self) -> np.ndarray:
        """Return current joint angles as (7,) rad. Franka has 7 joints (UR5e has 6)."""
        return np.asarray(self._robot.current_joint_state.position, dtype=np.float32).reshape(7)

    def get_ee_wrench(self, frame: str = "base") -> np.ndarray:
        """
        Return Franka's built-in estimated external wrench at the EE.

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

    # -------------------------
    # Gripper
    # -------------------------

    def get_gripper(self):
        if self.gripper is None:
            return None
        return self.gripper.state

    def gripper_move(self, width: float = 0.08, speed: float = 0.05) -> bool:
        if self.gripper is None:
            raise RuntimeError("Gripper is not available")
        return self.gripper.move(width, speed)

    def gripper_grasp(
        self,
        width: float = 0.02,
        speed: float = 0.03,
        force: float = 30.0,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ) -> bool:
        if self.gripper is None:
            raise RuntimeError("Gripper is not available")
        return self.gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    # -------------------------
    # Motion
    # -------------------------

    def move_tcp_pose(
        self,
        target_pose,
        velocity: float = 0.1,
        acceleration: float = 0.1,
        asynchronous: bool = False,
    ):
        """
        Blocking (or async) point-to-point move to a target TCP pose.

        Parameters
        ----------
        target_pose  : (6,) [x, y, z, rx, ry, rz] m/rad (rotation vector)
        velocity, acceleration : Unlike FrankaController (physical m/s,
                       m/s^2), franky has no direct per-call speed argument
                       -- only Robot.relative_dynamics_factor (0-1 each,
                       scaling Franka's own velocity/acceleration limits
                       independently; see _dyn_factor). Not literal m/s
                       values -- this only matches FrankaController's call
                       sites in this codebase (move_2_init_pos etc.)
                       because they already pass small 0-1-range numbers
                       (0.1, 0.15, ...).
        asynchronous : If True return immediately.
        """
        target_pose = np.asarray(target_pose, dtype=float)
        # Preserve the arm's current elbow (redundant joint-3) configuration
        # instead of leaving it to the planner to pick freely. This is a
        # 7-DOF arm: an unconstrained elbow choice for the target pose can
        # require a large, discontinuous null-space swing to reach even a
        # modest Cartesian target, tripping joint_velocity_discontinuity
        # (bundled with the cartesian_motion_generator_*_discontinuity
        # reflexes) regardless of how conservatively translation/rotation
        # dynamics are scaled.
        current_elbow = self._robot.current_pose.elbow_state
        target_robot_pose = RobotPose(_pose6_to_affine(target_pose), current_elbow)
        motion = CartesianMotion(target_robot_pose, ReferenceType.Absolute)

        prev_dyn = self._robot.relative_dynamics_factor
        self._robot.relative_dynamics_factor = _dyn_factor(velocity, acceleration)
        try:
            self._robot.move(motion, asynchronous=asynchronous)
        except Exception as e:
            # A reflex (e.g. cartesian_motion_generator_*_discontinuity)
            # leaves the robot in a fault state that blocks all further
            # motion until recovered -- match set_ee_velocity()'s pattern
            # instead of leaving the caller to figure that out from a bare
            # exception.
            print(f"[FrankaRobot] move_tcp_pose motion error: {e}")
            self._robot.recover_from_errors()
            raise
        finally:
            self._robot.relative_dynamics_factor = prev_dyn

    def set_ee_velocity(
        self,
        linear_velocity,
        angular_velocity=None,
        max_vel=None,
        max_ang_vel=None,
    ):
        """
        Command a Cartesian EE velocity, base frame (e.g. from a spacemouse).

        Unlike FrankaController.set_ee_velocity (which opens a persistent
        background servo thread that ramps toward the target itself), this
        issues a fresh CartesianVelocityMotion every call and lets franky's
        Ruckig planner handle the accel-limited retargeting -- no session
        bookkeeping, no separate thread that can starve under GIL
        contention. Safe to call every control tick.

        Parameters
        ----------
        linear_velocity  : (3,) m/s, base frame.
        angular_velocity : (3,) rad/s, base frame. Defaults to zero.
        max_vel, max_ang_vel : Speed caps applied every call (unlike
                       FrankaController, where they only take effect on the
                       first call that opens the session).
        """
        lin = np.asarray(linear_velocity, dtype=float).reshape(3)
        ang = np.zeros(3) if angular_velocity is None else np.asarray(angular_velocity, dtype=float).reshape(3)
        if not np.all(np.isfinite(lin)) or not np.all(np.isfinite(ang)):
            raise ValueError("Velocity command contains NaN or Inf")

        cap_lin = self.max_cart_vel if max_vel is None else max_vel
        cap_ang = self.max_ang_vel if max_ang_vel is None else max_ang_vel

        lin_speed = float(np.linalg.norm(lin))
        if lin_speed > cap_lin and lin_speed > 1e-9:
            lin = lin / lin_speed * cap_lin
        ang_speed = float(np.linalg.norm(ang))
        if ang_speed > cap_ang and ang_speed > 1e-9:
            ang = ang / ang_speed * cap_ang

        motion = CartesianVelocityMotion(Twist(lin.tolist(), ang.tolist()))
        try:
            self._robot.move(motion, asynchronous=True)
        except Exception as e:
            print(f"[FrankaRobot] set_ee_velocity motion error: {e}")
            self._robot.recover_from_errors()
            raise

    def move_joints(
        self,
        target_joints,
        velocity: float = 0.5,
        acceleration: float = 1.0,
        asynchronous: bool = False,
    ):
        """
        Blocking (or async) point-to-point move to a target joint configuration.

        Parameters
        ----------
        target_joints : (7,) target joint angles in rad.
        velocity, acceleration : See move_tcp_pose / _dyn_factor.
        """
        target_joints = list(np.asarray(target_joints, dtype=float).reshape(7))
        motion = JointMotion(target_joints, relative_dynamics_factor=_dyn_factor(velocity, acceleration))
        try:
            self._robot.move(motion, asynchronous=asynchronous)
        except Exception as e:
            print(f"[FrankaRobot] move_joints motion error: {e}")
            self._robot.recover_from_errors()
            raise

    def stop(self):
        """
        Stop any active motion. Mirrors UR5e's / FrankaController's stop():
        halts motion, keeps the connection alive.

        Issues a CartesianVelocityStopMotion (blocking) -- a properly
        Ruckig-planned deceleration to zero -- rather than the raw
        Robot.stop() e-stop, which is an abrupt discontinuity and the kind
        of thing that trips a reflex. Safe to call when already idle (the
        motion is already at zero, so it finishes immediately) and safe to
        call every control tick.
        """
        try:
            self._robot.move(CartesianVelocityStopMotion(), asynchronous=False)
        except Exception as e:
            print(f"[FrankaRobot] stop() failed: {e}")
            try:
                self._robot.recover_from_errors()
            except Exception:
                pass

    def recover(self):
        """Recover from robot errors (e.g. after a collision)."""
        self._robot.recover_from_errors()
        print("[FrankaRobot] Recovered from errors.")

    def disconnect(self):
        """Stop any motion and release. Best-effort; mirrors FrankaController.disconnect()."""
        try:
            self.stop()
        except Exception:
            pass
        print(f"[FrankaRobot] Disconnected: {self.robot_ip}")
