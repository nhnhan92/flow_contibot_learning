#!/usr/bin/env python3
"""
Verify the spacemouse -> velocity mapping that demo_collect.py's
_servo_toward() will send to FrankaController.set_ee_velocity(), before
trusting it in the full teleop loop.

Uses the actual production spacemouse module (hardware/spacemouse.py --
same one demo_collect.py imports), not a reimplementation.

Two modes:
    --dry_run (default): connects the spacemouse only, prints the linear
        velocity that WOULD be commanded each tick. No robot connection,
        completely safe -- use this first to sanity-check the axis mapping
        (does "push right" show +X? does "lift up" show +Z? etc. -- check
        against how your Franka is actually mounted relative to you, which
        may differ from the UR5e workcell this mapping was tuned for).
    --live: also connects to FrankaController and actually calls
        set_ee_velocity() with the computed velocity -- moves the real arm.
        Only use once the dry-run mapping looks correct, and start with a
        small --max_pos_speed.

Usage:
    python system_verification/test_spacemouse_velocity.py
    python system_verification/test_spacemouse_velocity.py --live --ip 172.16.0.2 --max_pos_speed 0.02
"""

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hardware"))
from spacemouse import _build_spacemouse


def main():
    parser = argparse.ArgumentParser(description="Verify spacemouse -> Franka velocity mapping.")
    parser.add_argument("--live", action="store_true",
                         help="Actually command the real Franka arm via set_ee_velocity(). "
                              "Default is dry-run (print only, no robot connection).")
    parser.add_argument("--ip", default="172.16.0.2", help="Franka FCI IP address (--live only)")
    parser.add_argument("--max_pos_speed", type=float, default=0.07,
                         help="m/s -- matches demo_collect.py's --max_pos_speed default. "
                              "This is the speed at full spacemouse deflection.")
    parser.add_argument("--deadzone", type=float, default=0.2,
                         help="Matches demo_collect.py's --deadzone default.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to run.")
    args = parser.parse_args()

    os_name = platform.system().lower()

    print("=" * 60)
    print(" SpaceMouse -> velocity mapping check")
    print("=" * 60)
    print(f"  Mode: {'LIVE (will move the real Franka arm)' if args.live else 'DRY-RUN (print only, no robot)'}")
    print(f"  max_pos_speed: {args.max_pos_speed} m/s, deadzone: {args.deadzone}")
    print()
    print("  Per hardware/spacemouse.py's documented mapping:")
    print("    push spacemouse right/left  -> expect +/-X")
    print("    push spacemouse away/toward -> expect +/-Y")
    print("    lift/lower spacemouse       -> expect +/-Z")
    print("  Check these signs against how YOUR Franka is physically")
    print("  mounted relative to you -- it may not match the UR5e")
    print("  workcell this mapping was originally tuned for.")
    print("=" * 60)

    print("\nConnecting SpaceMouse...")
    sm = _build_spacemouse(os_name=os_name, deadzone=args.deadzone)
    print("SpaceMouse connected.")

    robot = None
    if args.live:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hardware"))
        from franka_control import FrankaController
        print(f"\nConnecting to Franka at {args.ip} ...")
        robot = FrankaController(robot_ip=args.ip, use_gripper=False)
        print("Connected.")
        confirm = input(f"\nAbout to command the real arm at up to {args.max_pos_speed} m/s "
                         f"from spacemouse input. Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            sm.stop()
            robot.disconnect()
            sys.exit(1)

    print(f"\nMove the spacemouse -- Ctrl-C to stop ({args.duration:.0f}s max).\n")
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            xyz = sm.get_latest_xyz()            # (3,) normalized [-1,1], translation only
            lin_vel = xyz * args.max_pos_speed    # m/s -- same scaling _servo_toward reconstructs
            print(f"spacemouse xyz={np.round(xyz, 3)}  ->  "
                  f"linear_velocity (m/s)={np.round(lin_vel, 4)}", end="   \r")

            if robot is not None:
                robot.set_ee_velocity(lin_vel, angular_velocity=np.zeros(3),
                                       max_vel=args.max_pos_speed, max_ang_vel=0.1)

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        print("\nCleaning up...")
        sm.stop()
        if robot is not None:
            robot.stop()
            robot.disconnect()


if __name__ == "__main__":
    main()
