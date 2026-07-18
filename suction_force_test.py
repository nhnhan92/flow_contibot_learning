#!/usr/bin/env python3
"""
Suction force pull-off test.

Drives the Franka arm (via FrankaController / pylibfranka) and Flowbot
together through a fixed test procedure, and records the arm's built-in
F/T estimate against EE displacement during the pull-off phase.

Procedure
---------
0. Move the arm to the safety pose.
1. Apply the test PWM signal [pwm1, pwm2, pwm3] to Flowbot.
2. Move the arm from the safety pose to the suction pose at a constant
   speed/acceleration (the "suctioning" / approach phase).
3. Move the arm back from the suction pose to the safety pose at a lower
   speed (the actual pull-off test), using servo_tcp_pose so each tick can
   be sampled.
4. During step 3, sample get_ee_wrench(frame="base") synchronized with EE
   displacement (distance travelled from the suction pose).
5. Live-plot one wrench component against displacement while step 3 runs;
   record the plot to a video file via VideoRecorder.
6. All 6 wrench components are tared (zeroed against a baseline reading
   taken right before step 3 starts) before being plotted or saved.
7. Once back at the safety pose, stop and save the recorded data to CSV.

Usage
-----
    python suction_force_test.py \\
        --ip 172.16.0.2 --flowbot_port /dev/ttyACM0 \\
        --safety_pose  0.40 0.00 0.40 3.14 0.0 0.0 \\
        --suction_pose 0.40 0.00 0.20 3.14 0.0 0.0 \\
        --pwm 15 15 15 \\
        --record

There are no built-in defaults for --safety_pose/--suction_pose/--pwm --
they depend on your workcell and must be supplied explicitly.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning"))
from hardware.franka_control import FrankaController
from hardware.flowbot import flowbot

try:
    from flowbot.video_recorder import VideoRecorder
except ImportError:
    VideoRecorder = None

FORCE_COMPONENTS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def parse_args():
    parser = argparse.ArgumentParser(description="Suction force pull-off test.")
    parser.add_argument("--ip", default="172.16.0.2", help="Franka FCI IP address")
    parser.add_argument("--flowbot_port", default="/dev/ttyACM0", help="Flowbot serial port")
    parser.add_argument("--flowbot_baud", type=int, default=115200)
    parser.add_argument("--pwm_min", type=int, default=0)
    parser.add_argument("--pwm_max", type=int, default=26)

    parser.add_argument("--safety_pose", type=float, nargs=6, required=True,
                         metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                         help="Safety / initial TCP pose (m, rad rotvec).")
    parser.add_argument("--suction_pose", type=float, nargs=6, required=True,
                         metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                         help="Predefined suctioning/contact TCP pose (m, rad rotvec).")
    parser.add_argument("--pwm", type=int, nargs=3, required=True,
                         metavar=("PWM1", "PWM2", "PWM3"),
                         help="PWM signal to apply for the suction test.")

    parser.add_argument("--approach_velocity", type=float, default=0.05,
                         help="Constant speed for the safety->suction approach move (m/s).")
    parser.add_argument("--approach_acceleration", type=float, default=0.1,
                         help="Acceleration for the safety->suction approach move (m/s^2).")
    parser.add_argument("--test_velocity", type=float, default=0.01,
                         help="Lower speed for the suction->safety pull-off test (m/s).")
    parser.add_argument("--test_acceleration", type=float, default=0.02,
                         help="Acceleration for the pull-off test (m/s^2).")
    parser.add_argument("--dt", type=float, default=0.05,
                         help="Sample/servo tick period during the pull-off test (s).")
    parser.add_argument("--converge_tol", type=float, default=0.003,
                         help="Stop the pull-off test once within this distance (m) of the safety pose.")
    parser.add_argument("--phase3_timeout", type=float, default=60.0,
                         help="Abort the pull-off test if it hasn't converged within this many seconds.")
    parser.add_argument("--tare_samples", type=int, default=10,
                         help="Number of wrench samples to average for the tare baseline.")

    parser.add_argument("--force_component", choices=FORCE_COMPONENTS, default="Fz",
                         help="Which tared wrench component to live-plot (default Fz).")

    parser.add_argument("--record", action="store_true",
                         help="Record the live plot to a video file.")
    parser.add_argument("--record_fps", type=float, default=15.0)

    parser.add_argument("--output", default=None,
                         help="Output folder (default: data/suction_test/<timestamp>).")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation before moving (use with care).")
    return parser.parse_args()


def confirm_or_exit(args):
    print("\nAbout to run a suction force test:")
    print(f"  Safety pose  : {np.round(args.safety_pose, 4)}")
    print(f"  Suction pose : {np.round(args.suction_pose, 4)}")
    print(f"  PWM          : {args.pwm}")
    print(f"  Approach     : v={args.approach_velocity} m/s, a={args.approach_acceleration} m/s^2")
    print(f"  Pull-off test: v={args.test_velocity} m/s, a={args.test_acceleration} m/s^2")
    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            sys.exit(1)


def run_test(robot, fb, args):
    safety_pose = np.asarray(args.safety_pose, dtype=float)
    suction_pose = np.asarray(args.suction_pose, dtype=float)
    pwm = np.asarray(args.pwm, dtype=int)
    force_idx = FORCE_COMPONENTS.index(args.force_component)

    # ── Step 0: return to safety position ───────────────────────────────────
    print("\n[0/6] Moving to safety pose ...")
    robot.move_tcp_pose(safety_pose, velocity=args.approach_velocity,
                         acceleration=args.approach_acceleration)
    print(f"      At: {np.round(robot.get_tcp_pose(), 4)}")

    # ── Step 1: apply the test PWM signal ───────────────────────────────────
    print(f"\n[1/6] Applying PWM {pwm.tolist()} ...")
    fb.serial_sending(pwm, wait_ack=True, ack_timeout=1.0)
    fb.last_pwm = pwm

    # ── Step 2: move to the suction pose (constant speed, the approach) ────
    print("\n[2/6] Moving to suction pose (approach) ...")
    robot.move_tcp_pose(suction_pose, velocity=args.approach_velocity,
                         acceleration=args.approach_acceleration)
    print(f"      At: {np.round(robot.get_tcp_pose(), 4)}")

    # ── Tare: baseline wrench right before the pull-off test starts ────────
    print(f"\n[3/6] Taring wrench over {args.tare_samples} samples ...")
    tare_samples = []
    for _ in range(args.tare_samples):
        tare_samples.append(robot.get_ee_wrench(frame="base"))
        time.sleep(args.dt)
    wrench_offset = np.mean(tare_samples, axis=0)
    print(f"      Offset: {np.round(wrench_offset, 3)}")

    start_xyz = robot.get_tcp_pose()[:3].copy()

    # ── Live plot + video recorder setup ────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.canvas.manager.set_window_title("Suction pull-off test  (close window to abort)")
    line, = ax.plot([], [], marker="o", markersize=2)
    ax.set_xlabel("EE displacement from suction pose [m]")
    ax.set_ylabel(f"{args.force_component} [{'N' if force_idx < 3 else 'N*m'}]")
    ax.set_title("Suction pull-off test")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.show(block=False)

    recorder = None
    video_path = None
    if args.record:
        if VideoRecorder is None:
            print("[video] VideoRecorder not available -- recording skipped.")
        else:
            video_path = os.path.join(args.output, "pulloff_test.mp4")
            recorder = VideoRecorder(video_path, fps=args.record_fps, fig=fig)

    # ── Step 3+4: pull-off test -- one continuous move_tcp_pose back to the
    # safety pose (asynchronous=True runs it in a background thread with its
    # own smooth trapezoidal profile), while the main thread polls state at
    # ~1/dt Hz. 
    print("\n[4/6] Running pull-off test (continuous move back to safety pose) ...")
    log = {"t": [], "displacement": [], "wrench": [], "pose": []}
    t0 = time.time()
    aborted = False

    move_thread = robot.move_tcp_pose(safety_pose, velocity=args.test_velocity,
                                       acceleration=args.test_acceleration, asynchronous=True)
    try:
        while move_thread.is_alive():
            pose = robot.get_tcp_pose()
            wrench = robot.get_ee_wrench(frame="base").astype(float) - wrench_offset
            displacement = float(np.linalg.norm(pose[:3] - start_xyz))
            t = time.time() - t0

            log["t"].append(t)
            log["displacement"].append(displacement)
            log["wrench"].append(wrench.copy())
            log["pose"].append(pose.copy())

            line.set_data(log["displacement"], [w[force_idx] for w in log["wrench"]])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            if recorder is not None:
                recorder.capture()

            if not plt.fignum_exists(fig.number):
                print("      Plot window closed -- aborting test.")
                aborted = True
                robot.stop()   # lets move_tcp_pose's loop finish gracefully
                break
            if t > args.phase3_timeout:
                print(f"      WARNING: phase3_timeout ({args.phase3_timeout}s) exceeded -- stopping.")
                aborted = True
                robot.stop()
                break

            time.sleep(args.dt)

        move_thread.join(timeout=5.0)
    except Exception as e:
        # Keep whatever was already logged -- a mid-test fault shouldn't
        # discard real (and hard to reproduce) pull-off data.
        print(f"      ERROR during pull-off test: {e}")
        print(f"      Continuing with {len(log['t'])} samples already recorded.")
    finally:
        if recorder is not None:
            recorder.close()
            print(f"[video] Saved pull-off video -> {video_path}")

    final_pose = robot.get_tcp_pose()
    dist_to_safety = float(np.linalg.norm(final_pose[:3] - safety_pose[:3]))
    if not aborted and dist_to_safety >= args.converge_tol:
        # move_tcp_pose's background thread can fail silently (exceptions in
        # a raw Thread don't propagate to the caller) -- check where we
        # actually ended up rather than trusting "thread finished" alone.
        print(f"      WARNING: move may not have completed -- {dist_to_safety:.4f} m "
              f"from safety pose (tolerance {args.converge_tol} m).")

    # ── Step 5/6: stop and report ────────────────────────────────────────────
    print(f"\n[5/6] At: {np.round(final_pose, 4)}")
    print("[6/6] Pull-off test finished.")

    return log, wrench_offset, fig


def save_csv(log, wrench_offset, output_path):
    with open(output_path, "w", newline="") as f:
        f.write(f"# tare wrench offset (already subtracted below): {wrench_offset.tolist()}\n")
        writer = csv.writer(f)
        writer.writerow(
            ["t", "displacement", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "x", "y", "z", "rx", "ry", "rz"]
        )
        for t, disp, wrench, pose in zip(log["t"], log["displacement"], log["wrench"], log["pose"]):
            writer.writerow([t, disp, *wrench.tolist(), *pose.tolist()])
    print(f"[csv] Saved {len(log['t'])} samples -> {output_path}")


def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        args.output = os.path.join("data", "suction_test")
    os.makedirs(args.output, exist_ok=True)

    confirm_or_exit(args)

    print(f"\nConnecting to Franka at {args.ip} ...")
    robot = FrankaController(robot_ip=args.ip, use_gripper=False)

    print(f"Connecting to Flowbot on {args.flowbot_port} ...")
    fb = flowbot(serial_port=args.flowbot_port, baud=args.flowbot_baud,
                 pwm_min=args.pwm_min, pwm_max=args.pwm_max, enable_plot=False)
    fb.start()
    time.sleep(2.0)  # Arduino reset delay

    try:
        log, wrench_offset, fig = run_test(robot, fb, args)
    finally:
        try:
            fb.reset()  # sends PWM [0,0,0]; avoids flowbot.stop()'s enable_plot=False bug
        except Exception as e:
            print(f"Flowbot reset error: {e}")
        try:
            robot.disconnect()
        except Exception as e:
            print(f"Robot disconnect error: {e}")

    csv_path = os.path.join(args.output, f"suction_pwm{args.pwm}.csv")
    save_csv(log, wrench_offset, csv_path)

    fig_path = os.path.join(args.output, f"suction_pwm{args.pwm}.eps")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved figure -> {fig_path}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
