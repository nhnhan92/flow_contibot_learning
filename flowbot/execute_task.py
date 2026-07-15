"""
execute_task.py  –  Execute predefined motion tasks on the flowbot with data logging.

Each task file must define:
    get_waypoints(robot) -> list[tuple[np.ndarray(3,), float]]
    where each entry is (pc_target_mm, hold_time_s).

Optional in task file:
    TASK_NAME = "my_task"   # used in the output filename

Usage:
    python execute_task.py --task tasks/circle_xy.py
    python execute_task.py --task tasks/step_response.py --pressure-model linear -opt
    python execute_task.py --task tasks/sine_z.py --repeat 3 --output data/logs/sine_z
    python execute_task.py --task tasks/circle_xy.py --record          # saves .mp4 alongside CSV
    python execute_task.py --task tasks/circle_xy.py --record --record-fps 20
"""
from __future__ import annotations

import os, sys, csv, time, importlib.util, argparse, threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
TASK_DIR   = os.path.join(FILE_DIR, "tasks")
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)


# ──────────────────────────────────────────────────────────────
from flowbot.video_recorder import VideoRecorder


# ──────────────────────────────────────────────────────────────
# CSV Logger
# ──────────────────────────────────────────────────────────────
class TaskLogger:
    """Logs timestamped rows: t, pwm, pc (commanded), optitrack in manipulator frame (mm)."""

    HEADER = [
        "t_s",
        "pwm_1", "pwm_2", "pwm_3",
        "cmd_pc_x", "cmd_pc_y", "cmd_pc_z",
        "opti_mm_x", "opti_mm_y", "opti_mm_z",  # manipulator frame mm
    ]

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._t0 = time.perf_counter()
        print(f"[logger] Writing to {path}")

    def log(self, pwm, pc, opti_mm=None):
        t = time.perf_counter() - self._t0
        pwm = np.asarray(pwm, dtype=int).reshape(3,)
        pc  = np.asarray(pc,  dtype=float).reshape(3,)
        if opti_mm is not None:
            ox, oy, oz = float(opti_mm[0]), float(opti_mm[1]), float(opti_mm[2])
        else:
            ox = oy = oz = float("nan")
        self._w.writerow([
            f"{t:.4f}",
            int(pwm[0]), int(pwm[1]), int(pwm[2]),
            f"{pc[0]:.4f}", f"{pc[1]:.4f}", f"{pc[2]:.4f}",
            f"{ox:.3f}", f"{oy:.3f}", f"{oz:.3f}",
        ])

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.flush()
        self._f.close()


# ──────────────────────────────────────────────────────────────
# Task loader
# ──────────────────────────────────────────────────────────────
def load_task_module(task: str):
    """Load a task file by path and return its module."""
    task_path = Path(TASK_DIR, task).resolve()
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    spec = importlib.util.spec_from_file_location("_task_module", task_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_waypoints"):
        raise AttributeError(f"Task file must define get_waypoints(robot): {task_path}")
    return mod


# ──────────────────────────────────────────────────────────────
# Motion helpers
# ──────────────────────────────────────────────────────────────
ARRIVAL_THRESHOLD_MM = 1.0   # mm — close enough to declare waypoint reached


def move_to_waypoint(fb, target_pc, hold_s, logger, opti,
                     plot_handles=None, opti_trail_buf=None,
                     opti_origin_m=None, optitrack_init_ref=None,
                     stop_event: threading.Event = None,
                     recorder=None,
                     robot_trail_buf=None, robot_trail_handles=None,
                     log_data: bool = True,
                     opti_signs=(1.0, 1.0, 1.0)):
    """
    Drive fb toward target_pc using step(), then hold for hold_s seconds.
    Logs every control tick when log_data=True.
    Stops early if stop_event is set.
    Compensation (if any) is handled transparently inside fb.step().
    """
    def _stopped():
        return stop_event is not None and stop_event.is_set()


    # ── Phase 1: move toward target ──────────────────────────
    max_iters = 5000
    import random
    for _ in range(max_iters):
        if _stopped():
            return

        dist = float(np.linalg.norm(target_pc - fb.pc))
        if dist < ARRIVAL_THRESHOLD_MM:
            break

        d         = target_pc - fb.pc
        direction = d / (np.linalg.norm(d) + 1e-12)
        pwm       = fb.step(direction)

        opti_sample = opti.get_latest() if opti is not None else None
        if log_data:
            opti_mm = opti.transform_to_manip_mm(opti_sample, opti_origin_m, opti_signs) if opti is not None else None
            logger.log(pwm, fb.pc, opti_mm=opti_mm)

        if plot_handles is not None:
            _update_plot(fb, opti, opti_sample, opti_trail_buf,
                         opti_origin_m, optitrack_init_ref, recorder, opti_signs)

    # ── Phase 2: hold at target ───────────────────────────────────
    t_hold_end = time.perf_counter() + hold_s
    while time.perf_counter() < t_hold_end:
        if _stopped():
            return

        pwm = fb.step(np.zeros(3))
        opti_sample = opti.get_latest() if opti is not None else None
        if log_data:
            opti_mm = opti.transform_to_manip_mm(opti_sample, opti_origin_m, opti_signs) if opti is not None else None
            logger.log(pwm, fb.pc, opti_mm=opti_mm)

        if plot_handles is not None:
            _update_plot(fb, opti, opti_sample, opti_trail_buf,
                         opti_origin_m, optitrack_init_ref, recorder, opti_signs)

    # Record the OptiTrack position at end of hold as a waypoint dot
    if (robot_trail_buf is not None and robot_trail_handles is not None
            and opti is not None and optitrack_init_ref is not None
            and not optitrack_init_ref[0]):
        hold_sample = opti.get_latest()
        pt = opti.transform_to_manip_mm(hold_sample, opti_origin_m, opti_signs)
        if pt is not None:
            robot_trail_buf.append(pt.copy())
            pts = np.vstack(robot_trail_buf)
            robot_trail_handles["xy"].set_data(pts[:, 0], pts[:, 1])
            robot_trail_handles["xz"].set_data(pts[:, 0], pts[:, 2])
            robot_trail_handles["yz"].set_data(pts[:, 1], pts[:, 2])


def _update_plot(fb, opti, opti_sample, opti_trail_buf,
                 opti_origin_m, optitrack_init_ref, recorder=None,
                 opti_signs=(1.0, 1.0, 1.0)):
    """Update the 2D projection plot (pc + optitrack trail), then capture a frame."""
    OPTITRACK_TRAIL_LEN = 15

    if opti is not None and opti_sample is not None:
        if optitrack_init_ref[0]:
            opti_origin_m[:] = np.array(opti_sample.pos_xyz, dtype=float)
            opti_origin_m[1] += (fb.flowbot.l0 + fb.flowbot.lu) / 1000.0
            optitrack_init_ref[0] = False

        if not optitrack_init_ref[0]:
            transformed = opti.transform_to_manip_mm(opti_sample, opti_origin_m, opti_signs)
            fb.pl.update_opti_handle(fb.opti_handles, transformed)

            opti_trail_buf.append(transformed.copy())
            if len(opti_trail_buf) > OPTITRACK_TRAIL_LEN:
                opti_trail_buf[:] = opti_trail_buf[-OPTITRACK_TRAIL_LEN:]
            if len(opti_trail_buf) > 1:
                fb.pl.update_trail_handle(fb.trail_handles, np.vstack(opti_trail_buf))

    fb.update_plot()

    if recorder is not None:
        recorder.capture()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    from learning.hardware import flowbot as flowbot_module
    from online_optitrack import MotiveNatNetReader
    import platform

    ap = argparse.ArgumentParser(description="Execute a motion task on the flowbot with logging.")
    ap.add_argument("--task",           required=True,  help="Path to task file (defines get_waypoints).")
    ap.add_argument("--output", "-o",   default=None,   help="Output CSV path (auto-generated if omitted).")
    ap.add_argument("--repeat", "-n",   type=int, default=1, help="Repeat the task N times (default 1).")
    ap.add_argument("--baud",           type=int, default=115200)
    ap.add_argument("--pwm-min",        type=int, default=0)
    ap.add_argument("--pwm-max",        type=int, default=26)
    ap.add_argument("--pressure-model", choices=["learned", "linear"], default="linear",
                    help="'learned' (pkl) or 'linear' (a*pwm+b).")
    ap.add_argument("--no-plot",        action="store_true")
    ap.add_argument("--optitrack", "-opt", action="store_true", default=True)
    ap.add_argument("--max-pos-speed",  type=float, default=30.0,
                    help="Max task-space speed in mm/s (default 50).")
    ap.add_argument("--record",         action="store_true",
                    help="Record the plot window to an MP4 file (requires imageio[ffmpeg]).")
    ap.add_argument("--record-fps",     type=float, default=15.0,
                    help="Frame rate for recorded video (default 15).")
    ap.add_argument("--record-output",  default=None,
                    help="Video output path (auto-generated alongside CSV if omitted).")
    ap.add_argument("--inf",           type=str, default=None,
                    help="information about the task (randome seed, radius, etc.)")
    ap.add_argument("--seed",          type=int, default=8,
                    help="Random seed for reproducible tasks (default 8).")
    ap.add_argument("--home-every",     type=int, default=None,
                    help="Return to home and rest after every N waypoints (disabled if omitted).")
    ap.add_argument("--home-rest",      type=float, default=20.0,
                    help="Hold time at home during periodic rest (default 20.0 s).")
    ap.add_argument("--reverse",     type=bool, default=False,
                    help="For tasks with forward+reverse waypoints (e.g. lemniscate), "
                         "only do the reverse (inner→outer) half.")
    # ── Frame alignment ──────────────────────────────────────────
    ap.add_argument("--opti_signs", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    metavar=("SX", "SY", "SZ"),
                    help="Sign per axis for optitrack display and logging (default: 1 1 1).")
    # ── Compensation model ───────────────────────────────────────
    ap.add_argument("--compensate",     action="store_true", default=False,
                    help="Enable error compensator (ResGRU).")
    ap.add_argument("--compensate-ckpt", default="flowbot/residual_model/checkpoints",
                    help="Path to compensator checkpoint directory.")
    ap.add_argument("--compensate-method", choices=["simple", "mpc"], default="simple",
                    help="'simple': subtract position correction; 'mpc': optimise deltaU.")
    ap.add_argument("--compensate-alpha",   type=float, default=0.8,
                    help="[simple] Correction gain (default 0.5).")
    ap.add_argument("--compensate-dead-zone", type=float, default=0.2,
                    help="Minimum predicted error (mm) to trigger correction.")
    ap.add_argument("--compensate-min-disp",  type=float, default=3.0,
                    help="Minimum displacement (mm) since last correction.")
    ap.add_argument("--compensate-mpc-Q",   type=float, default=1.0,
                    help="[mpc] Position tracking weight.")
    ap.add_argument("--compensate-mpc-R",   type=float, default=0.1,
                    help="[mpc] Control effort weight.")
    ap.add_argument("--compensate-mpc-iter", type=int,  default=20,
                    help="[mpc] Number of Adam steps per tick.")
    args = ap.parse_args()

    # ── Load task ───────────────────────────────────────────────
    task_mod  = load_task_module(args.task)
    task_name = getattr(task_mod, "TASK_NAME",
                        Path(args.task).stem)

    # ── Output paths ─────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    inf_tag = f"_{args.inf}" if args.inf is not None else ""
    if args.output is None:
        
        out_path = str(Path(FILE_DIR).parent / "data" / "task_logs"
                       / f"{task_name}{inf_tag}_FF_{ts}.csv")
    else:
        out_path = str(Path(args.output) / f"{task_name}{inf_tag}_FF_{ts}.csv")

    # ── Serial port ─────────────────────────────────────────────
    os_name = platform.system().lower()
    serial_port = "/dev/ttyACM0" if "linux" in os_name else "COM9"

    # ── Init flowbot ────────────────────────────────────────────
    fb = flowbot_module.flowbot(
        serial_port    = serial_port,
        baud           = args.baud,
        pwm_min        = args.pwm_min,
        pwm_max        = args.pwm_max,
        enable_plot    = not args.no_plot,
        frequency      = 30.0,
        max_pos_speed  = args.max_pos_speed,
        pressure_model = args.pressure_model,
    )
    fb.start()

    # ── Init OptiTrack ──────────────────────────────────────────
    opti = None
    opti_origin_m       = np.zeros(3, dtype=float)
    optitrack_init_ref  = [True]   # mutable so helpers can update it
    opti_trail_buf      = []

    if args.optitrack:
        opti = MotiveNatNetReader(
            server_ip="150.65.146.84",
            local_ip="150.65.146.84",
            use_multicast=False,
            rigid_body_id=1,
        )
        opti.start()
        time.sleep(1.0)
        s = opti.get_latest()
        if s is not None:
            opti_origin_m[:] = np.array(s.pos_xyz, dtype=float)
            opti_origin_m[1] += (fb.flowbot.l0 + fb.flowbot.lu) / 1000.0
            optitrack_init_ref[0] = False
            print(f"[optitrack] Origin set: {np.round(opti_origin_m, 4)}")
        else:
            print("[optitrack] WARNING: no sample yet — origin stays [0,0,0]")

    # ── Get waypoints ────────────────────────────────────────────
    import inspect
    _gw_sig   = inspect.signature(task_mod.get_waypoints)
    _gw_kwargs = {}
    if "seed"    in _gw_sig.parameters:
        _gw_kwargs["seed"]    = _gw_sig.parameters["seed"].default if args.seed is None else args.seed
    if "reverse" in _gw_sig.parameters:
        _gw_kwargs["reverse"] = args.reverse
    waypoints = task_mod.get_waypoints(fb, **_gw_kwargs)
    print(f"[task] '{task_name}'  {len(waypoints)} waypoints  x{args.repeat} repeats")
    print(f"[task] Pressure model : {args.pressure_model}")
    print(f"[task] Output CSV     : {out_path}")

    # ── Compensator (optional) ───────────────────────────────────
    compensator = None
    if args.compensate:
        from flowbot.residual_model.compensator import ErrorCompensator
        compensator = ErrorCompensator.from_checkpoint(
            ckpt_dir            = args.compensate_ckpt,
            method              = args.compensate_method,
            alpha               = args.compensate_alpha,
            dead_zone_mm        = args.compensate_dead_zone,
            min_displacement_mm = args.compensate_min_disp,
            mpc_Q               = args.compensate_mpc_Q,
            mpc_R               = args.compensate_mpc_R,
            mpc_n_iter          = args.compensate_mpc_iter,
        )
        compensator.reset()
        print(f"[task] Compensator   : {args.compensate_method}  "
              f"ckpt={args.compensate_ckpt}")

    # ── Logger ───────────────────────────────────────────────────
    logger = TaskLogger(out_path)

    # ── Video recorder (optional) ────────────────────────────────
    recorder = None
    if args.record and fb.fig is not None:
        if args.record_output is not None:
            vid_path = args.record_output if args.record_output.endswith(".mp4") \
                       else args.record_output + ".mp4"
        else:
            vid_path = out_path.replace(".csv", ".mp4")
        recorder = VideoRecorder(vid_path, fps=args.record_fps, fig=fb.fig)
    elif args.record and fb.fig is None:
        print("[video] --record ignored: plot is disabled (--no-plot).")

    # ── Stop event (shared across threads) ───────────────────────
    stop_event = threading.Event()

    # ── Draw task reference on plot + wire stop keys ─────────────
    plot_handles = fb.pl if not args.no_plot else None
    if fb.fig is not None:
        # 'q' or Escape → stop
        def _on_key(event):
            if event.key in ('q', 'escape'):
                print("\n[stop] 'q' pressed — stopping after current waypoint.")
                stop_event.set()
        # closing the window → stop
        def _on_close(_event):
            print("\n[stop] Plot window closed — stopping.")
            stop_event.set()

        fb.fig.canvas.mpl_connect('key_press_event', _on_key)
        fb.fig.canvas.mpl_connect('close_event',     _on_close)

        # Draw optional reference geometry from the task module
        if hasattr(task_mod, "draw_reference"):
            task_mod.draw_reference(fb.axes, fb)

        # Waypoint dot markers — OptiTrack position at each arrival
        kw_dot = dict(color="tab:blue", marker="o", markersize=5,
                      linestyle="-", linewidth=1.0, alpha=0.9, label="measured")
        robot_trail_handles = {
            "xy": fb.axes["xy"].plot([], [], **kw_dot)[0],
            "xz": fb.axes["xz"].plot([], [], **kw_dot)[0],
            "yz": fb.axes["yz"].plot([], [], **kw_dot)[0],
        }
        fb.fig.canvas.draw_idle()
        fb.fig.canvas.flush_events()
    else:
        robot_trail_handles = None

    robot_trail_buf = [] if robot_trail_handles is not None else None

    # ── Attach compensator to flowbot (handles it inside step()) ─
    if compensator is not None:
        fb.set_compensator(compensator)

    # ── Run ──────────────────────────────────────────────────────
    _move_kwargs = dict(
        plot_handles=plot_handles,
        opti_trail_buf=opti_trail_buf,
        opti_origin_m=opti_origin_m,
        optitrack_init_ref=optitrack_init_ref,
        stop_event=stop_event,
        recorder=recorder,
        robot_trail_buf=robot_trail_buf,
        robot_trail_handles=robot_trail_handles,
        opti_signs=tuple(args.opti_signs),
    )

    home_pc        = np.asarray(fb.pc_init, dtype=float).reshape(3,)
    home_every     = args.home_every    # None = disabled
    home_rest_s    = args.home_rest
    waypoint_count = 0                  # counts executed non-home waypoints

    try:
        fb.step(np.array([0.0, 0.0, 15.0])) 
        input("   Press Enter when ready to begin... ")
        for rep in range(args.repeat):
            if stop_event.is_set():
                break
            print(f"\n-- Repeat {rep+1}/{args.repeat} --")
            for idx, (pc_target, hold_s) in enumerate(waypoints):
                if stop_event.is_set():
                    break
                pc_target = np.asarray(pc_target, dtype=float).reshape(3,)
                print(f"  Waypoint {idx+1}/{len(waypoints)}: "
                      f"target={np.round(pc_target,2)}  hold={hold_s}s")
                move_to_waypoint(fb, pc_target, hold_s, logger, opti, **_move_kwargs)
                logger.flush()

                # Periodic home return
                if home_every is not None:
                    waypoint_count += 1
                    if waypoint_count % home_every == 0:
                        print(f"  [home-every] Returning to home for {home_rest_s}s rest "
                              f"(after {waypoint_count} waypoints)")
                        move_to_waypoint(fb, home_pc, home_rest_s, logger, opti,
                                         log_data=False, **_move_kwargs)

        if not stop_event.is_set():
            print("\nTask complete. Returning to home.")
            move_to_waypoint(fb, fb.pc_init, hold_s=1.0, logger=logger, opti=opti,
                             **_move_kwargs)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        logger.close()
        if recorder is not None:
            recorder.close()
        if fb.fig is not None:
            plot_path = out_path.replace(".csv", ".eps")
            fb.fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {plot_path}")
        fb.stop()
        if opti is not None:
            opti.stop()
        print(f"Data saved to: {out_path}")


if __name__ == "__main__":
    main()
