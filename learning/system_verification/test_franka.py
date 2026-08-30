#!/usr/bin/env python3
"""
Check if the Franka robot is connected and responsive via FrankaRobot interface.

Usage:
    python check_franka_connection.py [--ip 172.16.0.2]

    # Also exercise move_tcp_pose / servo_tcp_pose / move_joints (moves the
    # real arm a small amount and back). Off by default -- opt in explicitly:
    python check_franka_connection.py --move
"""

import argparse
import sys
import time
import traceback

import numpy as np


def check_franky_import():
    print("[1/6] Checking franky installation ...")
    try:
        import franky
        print(f"      franky found: {franky.__file__}")
        return True
    except ImportError as e:
        print(f"      FAIL: {e}")
        print("      Install with: pip install franky-control")
        return False


def check_connection(robot_ip: str):
    print(f"[2/6] Connecting to Franka at {robot_ip} ...")
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "hardware"))
        from franka_robot import FrankaRobot
        robot = FrankaRobot(robot_ip=robot_ip)
        print("      Connection OK")
        return robot
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return None


def check_state(robot):
    print("[3/6] Reading robot state ...")
    try:
        joints = robot.get_joint_angles()
        assert joints.shape == (7,), f"Expected (7,) joint angles, got {joints.shape}"
        print(f"      Joint angles (rad): {np.round(joints, 4)}")

        tcp = robot.get_tcp_pose()
        assert tcp.shape == (6,), f"Expected (6,) TCP pose, got {tcp.shape}"
        print(f"      TCP pose [x,y,z,rx,ry,rz]: {np.round(tcp, 4)}")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_ft_sensor(robot):
    print("[4/6] Reading built-in force/torque estimate ...")
    try:
        wrench_base = robot.get_ee_wrench(frame="base")
        assert wrench_base.shape == (6,), f"Expected (6,) wrench, got {wrench_base.shape}"
        print(f"      EE wrench, base frame  [Fx,Fy,Fz,Tx,Ty,Tz]: {np.round(wrench_base, 3)}")

        wrench_ee = robot.get_ee_wrench(frame="ee")
        print(f"      EE wrench, EE frame    [Fx,Fy,Fz,Tx,Ty,Tz]: {np.round(wrench_ee, 3)}")

        tau_ext = robot.get_joint_external_torques()
        assert tau_ext.shape == (7,), f"Expected (7,) joint torques, got {tau_ext.shape}"
        print(f"      Joint ext. torques (Nm): {np.round(tau_ext, 3)}")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_move_tcp_pose(robot, delta: float, velocity: float, acceleration: float):
    """Point-to-point Cartesian move (blocking) up by `delta` metres in Z, then back."""
    print("[5a/6] Testing move_tcp_pose (point-to-point) ...")
    try:
        start = robot.get_tcp_pose()
        target = start.copy()
        target[2] += delta  # +Z, straight up

        robot.move_tcp_pose(target, velocity=velocity, acceleration=acceleration)
        reached = robot.get_tcp_pose()
        print(f"      Target Z {target[2]:.4f}  ->  reached Z {reached[2]:.4f}")

        robot.move_tcp_pose(start, velocity=velocity, acceleration=acceleration)
        back = robot.get_tcp_pose()
        print(f"      Returned to Z {back[2]:.4f} (start was {start[2]:.4f})")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_set_ee_velocity(robot, speed: float, duration: float, hold_dt: float,
                           max_vel: float, max_ang_vel: float):
    """Hold a constant +Z EE velocity for `duration`s, then the reverse, then stop.

    Mirrors test_franka_control.py's check_set_ee_velocity (FrankaController /
    pylibfranka) so the two backends can be compared directly on hardware.
    """
    print("[5b/6] Testing set_ee_velocity (real-time velocity control) ...")
    try:
        start = robot.get_tcp_pose()
        steps = max(1, int(duration / hold_dt))

        for _ in range(steps):
            robot.set_ee_velocity([0.0, 0.0, speed], max_vel=max_vel, max_ang_vel=max_ang_vel)
            time.sleep(hold_dt)
        mid = robot.get_tcp_pose()
        print(f"      Held +Z velocity {speed:.3f} m/s for {duration:.2f}s: Z {start[2]:.4f} -> {mid[2]:.4f}")

        for _ in range(steps):
            robot.set_ee_velocity([0.0, 0.0, -speed], max_vel=max_vel, max_ang_vel=max_ang_vel)
            time.sleep(hold_dt)
        back = robot.get_tcp_pose()
        print(f"      Held -Z velocity for {duration:.2f}s: Z {mid[2]:.4f} -> {back[2]:.4f} (start was {start[2]:.4f})")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def check_move_joints(robot, delta: float, velocity: float, acceleration: float):
    """Small move of the wrist joint (joint 7 -- safest to nudge), then back."""
    print("[5c/6] Testing move_joints ...")
    try:
        start = robot.get_joint_angles()
        target = start.copy()
        target[-1] += delta

        robot.move_joints(target, velocity=velocity, acceleration=acceleration)
        reached = robot.get_joint_angles()
        print(f"      Target joint7 {target[-1]:.4f}  ->  reached {reached[-1]:.4f}")

        robot.move_joints(start, velocity=velocity, acceleration=acceleration)
        back = robot.get_joint_angles()
        print(f"      Returned to joint7 {back[-1]:.4f} (start was {start[-1]:.4f})")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        traceback.print_exc()
        return False


def check_error_recovery(robot):
    print("[6/6] Testing error recovery call ...")
    try:
        robot.recover()
        print("      recover_from_errors() OK")
        return True
    except Exception as e:
        print(f"      FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check Franka robot connection.")
    parser.add_argument("--ip", default="172.16.0.2", help="Franka FCI IP address")
    parser.add_argument("--move", action="store_true",
                         help="Also exercise move_tcp_pose/servo_tcp_pose/move_joints "
                              "(moves the real arm a small amount and back). "
                              "Requires interactive confirmation unless --yes is given.")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation before moving (use with care).")
    parser.add_argument("--delta", type=float, default=0.04,
                         help="Cartesian test displacement in metres (default 0.02 = 2cm).")
    parser.add_argument("--joint_delta", type=float, default=0.5,
                         help="Joint test displacement in radians (default 0.1).")
    parser.add_argument("--velocity", type=float, default=0.05, help="Test move velocity.")
    parser.add_argument("--acceleration", type=float, default=0.05, help="Test move acceleration.")
    parser.add_argument("--ee_velocity", type=float, default=0.01,
                         help="set_ee_velocity test speed in m/s.")
    parser.add_argument("--ee_velocity_duration", type=float, default=1.0,
                         help="How long to hold each velocity direction (s).")
    parser.add_argument("--ee_velocity_dt", type=float, default=0.05,
                         help="Interval between set_ee_velocity calls (s).")
    args = parser.parse_args()

    print("=" * 50)
    print(" Franka Connection Check")
    print("=" * 50)

    results = {}

    results["franky"] = check_franky_import()
    if not results["franky"]:
        print("\nAborting: franky not available.")
        sys.exit(1)

    robot = check_connection(args.ip)
    results["connection"] = robot is not None
    if robot is None:
        print("\nAborting: could not connect to robot.")
        sys.exit(1)

    results["state"] = check_state(robot)
    results["ft_sensor"] = check_ft_sensor(robot)

    if args.move:
        if not args.yes:
            print(f"\nAbout to move the real Franka arm at {args.ip}:")
            print(f"  - Cartesian +{args.delta*1000:.0f}mm in Z and back (move_tcp_pose)")
            print(f"  - Hold {args.ee_velocity:.3f} m/s in +Z then -Z for "
                  f"{args.ee_velocity_duration:.2f}s each (set_ee_velocity)")
            print(f"  - Joint 7 +{args.joint_delta:.3f} rad and back (move_joints)")
            confirm = input("Proceed? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted by user.")
                robot.disconnect()
                sys.exit(1)

        results["move_tcp_pose"] = check_move_tcp_pose(
            robot, args.delta, args.velocity, args.acceleration)
        results["set_ee_velocity"] = check_set_ee_velocity(
            robot, args.ee_velocity, args.ee_velocity_duration, args.ee_velocity_dt,
            args.velocity, args.acceleration)
        results["move_joints"] = check_move_joints(
            robot, args.joint_delta, args.velocity, args.acceleration)
    else:
        print("\n[5/6] Skipping motion tests (pass --move to exercise "
              "move_tcp_pose/set_ee_velocity/move_joints).")

    results["recovery"] = check_error_recovery(robot)

    robot.disconnect()

    print("\n" + "=" * 50)
    all_ok = all(results.values())
    for check, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {check:<16} : {status}")
    print("=" * 50)
    print("Overall:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
