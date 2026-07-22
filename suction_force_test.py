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
3. Pull-off test: the suction_pose -> safety_pose trajectory is divided into
   waypoints spaced --step_displacement apart. At each waypoint the arm
   makes one point-to-point move (move_tcp_pose, blocking) and then holds
   still: after --settle_time (residual dynamics die out), get_ee_wrench()
   is sampled --step_samples times and averaged. Measuring at rest -- not
   during continuous motion -- means the reading is a genuine static force,
   free of the inertial/dynamic-model-mismatch noise a moving arm produces.
4. Each waypoint's averaged wrench is logged against EE displacement
   (distance travelled from the suction pose).
5. Live-plot one wrench component against displacement, one point per
   waypoint as it completes; record the plot to a video file via
   VideoRecorder.
6. All 6 wrench components are tared (zeroed against a baseline reading
   taken right before the pull-off test starts) before being plotted/saved.
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
import threading
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    parser.add_argument("--load_mass", type=float, default=0.0,
                         help="Mass of the tool/gripper attached to the flange, in kg. "
                              "Franka's get_ee_wrench() is a model-based residual (measured "
                              "joint torque minus predicted), so an uncalibrated tool mass "
                              "shows up as spurious motion-correlated force -- set this to "
                              "your actual gripper mass to cancel that out. Default 0.0 "
                              "leaves the robot's default (no-tool) assumption unchanged.")
    parser.add_argument("--load_com", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                         metavar=("X", "Y", "Z"),
                         help="Load center of mass, in the flange frame (m). Default (0,0,0).")
    parser.add_argument("--load_inertia", type=float, nargs=9, default=None,
                         metavar=tuple(f"I{i}" for i in range(1, 10)),
                         help="Load inertia tensor about its COM, 3x3 column-major (kg*m^2), "
                              "9 values. If omitted, a small solid-sphere estimate is computed "
                              "from --load_mass and --load_radius -- libfranka's setLoad "
                              "rejects an all-zero inertia tensor as an invalid (physically "
                              "unrealizable) matrix, so this must be nonzero.")
    parser.add_argument("--load_radius", type=float, default=0.1,
                         help="Characteristic radius (m) used to estimate --load_inertia as a "
                              "solid sphere when --load_inertia isn't given explicitly. "
                              "Default 0.05 (5cm) -- adjust to roughly your gripper's size.")

    parser.add_argument("--approach_velocity", type=float, default=0.02,
                         help="Constant speed for the safety->suction approach move (m/s).")
    parser.add_argument("--approach_acceleration", type=float, default=0.2,
                         help="Acceleration for the safety->suction approach move (m/s^2).")
    parser.add_argument("--test_velocity", type=float, default=0.005,
                         help="Speed for each pull-off waypoint move (m/s).")
    parser.add_argument("--test_acceleration", type=float, default=0.01,
                         help="Acceleration for each pull-off waypoint move (m/s^2).")
    parser.add_argument("--step_displacement", type=float, default=0.0005,
                         help="Distance (m) between successive pull-off waypoints -- the "
                              "suction_pose->safety_pose trajectory is divided into steps of "
                              "this size. Default 0.005 (5mm); pick your own value, this is "
                              "just a starting point.")
    parser.add_argument("--settle_time", type=float, default=1.2,
                         help="Seconds to hold still at each waypoint before sampling, so "
                              "residual dynamics from the move die out (default 0.5).")
    parser.add_argument("--step_samples", type=int, default=5,
                         help="Number of wrench samples to average at each waypoint (default 5).")
    parser.add_argument("--step_retries", type=int, default=1,
                         help="If a waypoint's move fails (e.g. an intermittent libfranka "
                              "reflex trip), recover() and retry it this many times before "
                              "giving up on that waypoint and moving on (default 1).")
    parser.add_argument("--dt", type=float, default=0.05,
                         help="Interval between repeated samples (tare and per-waypoint) (s).")
    parser.add_argument("--converge_tol", type=float, default=0.003,
                         help="Sanity-check tolerance (m): warn if the final waypoint isn't "
                              "actually this close to the safety pose.")
    parser.add_argument("--phase3_timeout", type=float, default=120.0,
                         help="Abort the pull-off test if it's still running after this many "
                              "seconds (default 120 -- stepping is slower than one continuous "
                              "move, raise this if you have many steps).")
    parser.add_argument("--tare_samples", type=int, default=10,
                         help="Number of wrench samples to average for the tare baseline.")

    parser.add_argument("--force_component", choices=FORCE_COMPONENTS, default="Fz",
                         help="Which tared wrench component to live-plot (default Fz).")

    parser.add_argument("--record", action="store_true",
                         help="Record the live plot to a video file.")
    parser.add_argument("--record_fps", type=float, default=10.0)

    parser.add_argument("--output", default=None,
                         help="Output folder (default: data/suction_test/<timestamp>).")
    parser.add_argument("--yes", action="store_true",
                         help="Skip the interactive confirmation before moving (use with care).")
    return parser.parse_args()


def _num_pulloff_steps(safety_pose, suction_pose, step_displacement):
    total_distance = float(np.linalg.norm(np.asarray(safety_pose[:3]) - np.asarray(suction_pose[:3])))
    return max(1, int(np.ceil(total_distance / step_displacement))), total_distance


def confirm_or_exit(args):
    n_steps, total_distance = _num_pulloff_steps(args.safety_pose, args.suction_pose, args.step_displacement)
    est_seconds = n_steps * (args.settle_time + args.step_samples * args.dt)

    print("\nAbout to run a suction force test:")
    print(f"  Safety pose  : {np.round(args.safety_pose, 4)}")
    print(f"  Suction pose : {np.round(args.suction_pose, 4)}")
    print(f"  PWM          : {args.pwm}")
    print(f"  Load         : mass={args.load_mass} kg, com={args.load_com}")
    print(f"  Approach     : v={args.approach_velocity} m/s, a={args.approach_acceleration} m/s^2")
    print(f"  Pull-off test: {n_steps} steps of {args.step_displacement*1000:.1f}mm "
          f"({total_distance*1000:.1f}mm total), v={args.test_velocity} m/s, "
          f"a={args.test_acceleration} m/s^2")
    print(f"                 ~{est_seconds:.0f}s of settle+sample time alone "
          f"(plus per-step move time)")
    if not args.yes:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            sys.exit(1)


def _check_abort(stop_flag, abort_info):
    """If an abort has been requested, mark it and tell the caller to bail out."""
    if stop_flag["stop"]:
        abort_info["aborted"] = True
        return True
    return False


def _run_live_plot(fig, ax, line, log, force_idx, recorder, stop_flag, on_ready, on_abort=None):
    """
    Live plot on the main thread via FuncAnimation, redrawing at a fixed
    ~10 Hz independent of _sample_loop's sampling rate. Blocks (plt.show())
    until stop_flag["stop"] is set (by the test procedure finishing, or by
    the user closing the window / pressing q here).

    on_ready() is called only once the window is actually visible on screen
    -- it's meant to kick off the whole test procedure (not just the
    pull-off phase), so the plot is showing well before any data exists
    instead of racing the window's own render startup.

    on_abort(), if given, is called on window-close/q -- e.g. robot.stop(),
    so closing the window can interrupt a blocking move mid-flight, not just
    the pull-off phase's sampling loop.
    """
    def _on_key(event):
        if event.key == "q":
            print("\n[plot] Q pressed -- stopping.")
            stop_flag["stop"] = True
            if on_abort is not None:
                on_abort()

    def _on_close(_event):
        stop_flag["stop"] = True
        if on_abort is not None:
            on_abort()

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", _on_close)

    # Show the window now (non-blocking) and force it to actually render
    # before starting motion/sampling.
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    on_ready()

    def _update(_):
        if len(log["t"]) >= 1:
            line.set_data(log["displacement"], [w[force_idx] for w in log["wrench"]])
            ax.relim()
            ax.autoscale_view()
        if recorder is not None:
            recorder.capture()
        if stop_flag["stop"]:
            plt.close("all")   # closes all windows -> plt.show() returns
        return line,

    _ani = animation.FuncAnimation(fig, _update, interval=100, blit=False, cache_frame_data=False)  # noqa: F841 -- must be held to prevent GC

    plt.show()   # blocks until the window is closed


def _test_procedure(robot, fb, args, log, stop_flag, abort_info, result):
    """
    Runs the entire step 0-6 procedure in a background thread, so the main
    thread can own the live plot (_run_live_plot) from before step 0 all the
    way through the pull-off phase. That gives the window the whole
    safety-move + approach-move + tare duration to actually render before
    any data exists, and keeps it responsive throughout since its
    FuncAnimation loop runs continuously on the main thread for the entire
    test, not just the pull-off phase. Checks stop_flag between every step
    so a window-close/q during steps 0-3 aborts cleanly instead of
    continuing on to the next step regardless.
    """
    try:
        safety_pose = np.asarray(args.safety_pose, dtype=float)
        suction_pose = np.asarray(args.suction_pose, dtype=float)
        pwm = np.asarray(args.pwm, dtype=int)

        print("\n[0/6] Moving to safety pose ...")
        robot.move_tcp_pose(safety_pose, velocity=args.approach_velocity,
                             acceleration=args.approach_acceleration)
        print(f"      At: {np.round(robot.get_tcp_pose(), 4)}")
        if _check_abort(stop_flag, abort_info):
            return
        
        print(f"\n[1/6] Applying PWM {pwm.tolist()} ...")
        fb.serial_sending(pwm, wait_ack=False, ack_timeout=2.0)
        fb.last_pwm = pwm
        time.sleep(2.5)   # let the suctioning effect settle a bit
        if _check_abort(stop_flag, abort_info):
            return
        
        print("\n[2/6] Moving to suction pose (approach) ...")
        robot.move_tcp_pose(suction_pose, velocity=args.test_velocity,
                             acceleration=args.test_acceleration)
        print(f"      At: {np.round(robot.get_tcp_pose(), 4)}")
        if _check_abort(stop_flag, abort_info):
            return
        
        

        print(f"\n[3/6] Taring wrench over {args.tare_samples} samples ...")
        tare_samples = []
        for _ in range(args.tare_samples):
            if _check_abort(stop_flag, abort_info):
                return
            tare_samples.append(robot.get_ee_wrench(frame="base"))
            time.sleep(args.dt)
        wrench_offset = np.mean(tare_samples, axis=0)
        print(f"      Offset: {np.round(wrench_offset, 3)}")
        result["wrench_offset"] = wrench_offset

        start_xyz = robot.get_tcp_pose()[:3].copy()
        if _check_abort(stop_flag, abort_info):
            return

        # ── Step 4: pull-off test -- step-and-hold, not continuous motion.
        # The suction_pose->safety_pose line is divided into waypoints
        # step_displacement apart; at each one, move (blocking point-to-
        # point), hold for settle_time so residual dynamics die out, then
        # average step_samples wrench readings. Measuring while stationary
        # means the reading is a genuine static force, not contaminated by
        # the inertial/dynamic-model-mismatch effects a moving arm produces.
        time.sleep(1.5)   # let the approach move's residual dynamics die out
        n_steps, total_distance = _num_pulloff_steps(safety_pose, suction_pose, args.step_displacement)
        print(f"\n[4/6] Running stepped pull-off test: {n_steps} steps over "
              f"{total_distance*1000:.1f}mm ...")
        t0 = time.time()

        for i in range(1, n_steps + 1):
            if _check_abort(stop_flag, abort_info):
                return

            frac = min(1.0, i / n_steps)
            waypoint = suction_pose + frac * (safety_pose - suction_pose)

            # Opening a brand-new realtime control session on nearly every
            # step (instead of once for the whole trajectory) occasionally
            # trips libfranka's reflex (acceleration_discontinuity) even
            # when the commanded profile itself is smooth -- intermittent,
            # not reliably reproduced. Rather than losing the rest of a
            # multi-minute stepped test to one transient trip, recover and
            # retry this step a bounded number of times before giving up on
            # it specifically and moving on.
            moved = False
            for attempt in range(args.step_retries + 1):
                try:
                    robot.move_tcp_pose(waypoint, velocity=args.test_velocity,
                                         acceleration=args.test_acceleration)
                    moved = True
                    break
                except Exception as e:
                    print(f"      Step {i}/{n_steps} move error ({e}); "
                          f"recovering (attempt {attempt + 1}/{args.step_retries + 1}) ...")
                    try:
                        robot.recover()
                    except Exception:
                        pass
                    time.sleep(0.5)
            if _check_abort(stop_flag, abort_info):
                return
            if not moved:
                print(f"      Step {i}/{n_steps}: giving up after {args.step_retries} "
                      f"retries -- skipping this waypoint.")
                continue

            time.sleep(args.settle_time)   # let residual dynamics die out

            step_samples = []
            for _ in range(args.step_samples):
                if _check_abort(stop_flag, abort_info):
                    return
                step_samples.append(robot.get_ee_wrench(frame="base"))
                time.sleep(args.dt)
            wrench = np.mean(step_samples, axis=0).astype(float) - wrench_offset

            pose = robot.get_tcp_pose()
            displacement = float(np.linalg.norm(pose[:3] - start_xyz))

            log["t"].append(time.time() - t0)
            log["displacement"].append(displacement)
            log["wrench"].append(wrench.copy())
            log["pose"].append(pose.copy())

            force_idx = FORCE_COMPONENTS.index(args.force_component)
            print(f"      Step {i}/{n_steps}: displacement={displacement*1000:.1f}mm  "
                  f"{args.force_component}={wrench[force_idx]:.3f}")

            if time.time() - t0 > args.phase3_timeout:
                print(f"      WARNING: phase3_timeout ({args.phase3_timeout}s) exceeded -- "
                      f"stopping early ({i}/{n_steps} steps done).")
                abort_info["aborted"] = True
                return

        final_pose = robot.get_tcp_pose()
        result["final_pose"] = final_pose
        dist_to_safety = float(np.linalg.norm(final_pose[:3] - safety_pose[:3]))
        if dist_to_safety >= args.converge_tol:
            print(f"      WARNING: final position {dist_to_safety:.4f} m from safety pose "
                  f"(tolerance {args.converge_tol} m).")

        print(f"\n[5/6] At: {np.round(final_pose, 4)}")
        print("[6/6] Pull-off test finished.")
        fb.serial_sending(np.zeros(3, dtype=int), wait_ack=True, ack_timeout=2.0)   # stop suctioning
        fb.last_pwm = np.zeros(3, dtype=int)
        fb.serial_sending("r")

    except Exception as e:
        # Keep whatever was already logged -- a mid-test fault shouldn't
        # discard real (and hard to reproduce) pull-off data.
        print(f"      ERROR during test procedure: {e}")
        abort_info["aborted"] = True
    finally:
        stop_flag["stop"] = True   # tell the plot loop we're done, however we exited


def run_test(robot, fb, args):
    force_idx = FORCE_COMPONENTS.index(args.force_component)

    # ── Live plot + video recorder setup -- created and shown before any
    # robot motion starts, so the window has the entire safety-move +
    # approach-move + tare duration to actually render, instead of only
    # appearing partway through the pull-off phase once samples exist.
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.canvas.manager.set_window_title("Suction pull-off test  (close window to abort)")
    line, = ax.plot([], [], marker="o", markersize=2)
    ax.set_xlabel("EE displacement from suction pose [m]")
    ax.set_ylabel(f"{args.force_component} [{'N' if force_idx < 3 else 'N*m'}]")
    ax.set_title("Suction pull-off test")

    recorder = None
    video_path = None
    if args.record:
        if VideoRecorder is None:
            print("[video] VideoRecorder not available -- recording skipped.")
        else:
            video_path = os.path.join(args.output, f"pulloff_{args.pwm}.mp4")
            recorder = VideoRecorder(video_path, fps=args.record_fps, fig=fig)

    log = {"t": [], "displacement": [], "wrench": [], "pose": []}
    stop_flag = {"stop": False}
    abort_info = {"aborted": False}
    result = {}

    test_thread = threading.Thread(
        target=_test_procedure,
        args=(robot, fb, args, log, stop_flag, abort_info, result),
        daemon=True,
    )

    _run_live_plot(fig, ax, line, log, force_idx, recorder, stop_flag,
                    on_ready=test_thread.start, on_abort=robot.stop)   # blocks on the main thread

    stop_flag["stop"] = True   # in case the window closed before _test_procedure did
    test_thread.join(timeout=10.0)

    if recorder is not None:
        recorder.close()
        print(f"[video] Saved pull-off video -> {video_path}")

    wrench_offset = result.get("wrench_offset", np.zeros(6))
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

    if args.load_mass > 0.0:
        load_inertia = args.load_inertia
        if load_inertia is None:
            # Solid-sphere approximation (I = 2/5 * m * r^2 on each diagonal,
            # zero off-diagonal) -- just needs to be a physically valid
            # (positive-definite) matrix, not an accurate one; libfranka
            # rejects an all-zero tensor outright.
            i_diag = 0.4 * args.load_mass * args.load_radius ** 2
            load_inertia = [i_diag, 0.0, 0.0, 0.0, i_diag, 0.0, 0.0, 0.0, i_diag]
        print(f"Setting load: mass={args.load_mass} kg, com={args.load_com}, "
              f"inertia={load_inertia} ...")
        robot.set_load(args.load_mass, args.load_com, load_inertia)

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


if __name__ == "__main__":
    main()
