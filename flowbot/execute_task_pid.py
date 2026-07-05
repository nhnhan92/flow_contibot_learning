"""
execute_task_pid.py  –  Execute predefined tasks with proprioception-model PID feedback.

After the IK drives the robot near each waypoint (MOVE phase), a task-space PID
controller takes over during the hold phase to eliminate steady-state position
error using the trained flow-based proprioception model as the position sensor.

Control law (hold phase only):
    pred_pos       = propri_model(flow, pwm)        # measured tip (mm)
    error          = target − pred_pos              # task-space error (mm)
    u              = Kp·e + Ki·∫e + Kd·ė           # PID correction (mm)
    virtual_target = target + u                     # biased IK goal
    fb.step(direction toward virtual_target)

Each task file must define:
    get_waypoints(robot) -> list[tuple[np.ndarray(3,), float]]
    where each entry is (pc_target_mm, hold_time_s).

Usage:
    python flowbot/execute_task_pid.py --task tasks/circle_xy.py \\
        --propri-ckpt flowbot/proprioception_model/checkpoints/free_human/freeload \\
        --pid-kp 0.4 --pid-ki 0.01
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

from flowbot.video_recorder import VideoRecorder


# ──────────────────────────────────────────────────────────────
# CSV Logger  (extends base header with proprioception predictions)
# ──────────────────────────────────────────────────────────────
class TaskLogger:
    HEADER = [
        "t_s",
        "pwm_1", "pwm_2", "pwm_3",
        "cmd_pc_x", "cmd_pc_y", "cmd_pc_z",
        "pred_x", "pred_y", "pred_z",          # proprioception estimate
        "pid_err_x", "pid_err_y", "pid_err_z", # target − pred
        "opti_x", "opti_y", "opti_z",
        "opti_qx", "opti_qy", "opti_qz", "opti_qw",
    ]

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f  = open(path, "w", newline="", encoding="utf-8")
        self._w  = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._t0 = time.perf_counter()
        print(f"[logger] Writing to {path}")

    def log(self, pwm, pc, target_pc, pred_pos=None, opti_sample=None):
        t   = time.perf_counter() - self._t0
        pwm = np.asarray(pwm,      dtype=int).reshape(3,)
        pc  = np.asarray(pc,       dtype=float).reshape(3,)
        tgt = np.asarray(target_pc, dtype=float).reshape(3,)

        if pred_pos is not None:
            px, py, pz = float(pred_pos[0]), float(pred_pos[1]), float(pred_pos[2])
            ex, ey, ez = float(tgt[0]-px),  float(tgt[1]-py),  float(tgt[2]-pz)
        else:
            px = py = pz = ex = ey = ez = float("nan")

        if opti_sample is not None:
            ox, oy, oz = opti_sample.pos_xyz
            qx, qy, qz, qw = opti_sample.quat_xyzw
        else:
            ox = oy = oz = qx = qy = qz = qw = float("nan")

        self._w.writerow([
            f"{t:.4f}",
            int(pwm[0]), int(pwm[1]), int(pwm[2]),
            f"{pc[0]:.4f}", f"{pc[1]:.4f}", f"{pc[2]:.4f}",
            f"{px:.4f}", f"{py:.4f}", f"{pz:.4f}",
            f"{ex:.4f}", f"{ey:.4f}", f"{ez:.4f}",
            f"{ox:.6f}", f"{oy:.6f}", f"{oz:.6f}",
            f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}",
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
ARRIVAL_THRESHOLD_MM = 1.0


def _opti_transform(opti, opti_sample, opti_origin_m):
    if opti is None or opti_sample is None:
        return None
    t = opti.opti_to_manip(np.array(opti_sample.pos_xyz, dtype=float), opti_origin_m)
    t[0] = -t[0]
    t[1] = -t[1]
    return t


def move_to_waypoint(fb, target_pc, hold_s, logger, opti, pid_ctrl,
                     plot_handles=None, opti_trail_buf=None,
                     opti_origin_m=None, optitrack_init_ref=None,
                     stop_event: threading.Event = None,
                     recorder=None,
                     robot_trail_buf=None, robot_trail_handles=None,
                     log_data: bool = True):
    """
    Drive fb toward target_pc (open-loop IK), then hold with PID correction.

    Phase 1 — MOVE : fb.step(direction) until dist < ARRIVAL_THRESHOLD_MM
    Phase 2 — HOLD : pid_ctrl.correct(fb, target_pc) for hold_s seconds
                     PID closes the loop using the proprioception model.
    """
    def _stopped():
        return stop_event is not None and stop_event.is_set()

    # ── Phase 1: open-loop transit ────────────────────────────────
    for _ in range(5000):
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
            logger.log(pwm, fb.pc, target_pc, opti_sample=opti_sample)

        if plot_handles is not None:
            _update_plot(fb, opti, opti_sample, opti_trail_buf,
                         opti_origin_m, optitrack_init_ref, recorder)

    # ── Phase 2: PID-corrected hold ───────────────────────────────
    pid_ctrl.reset()
    t_hold_end = time.perf_counter() + hold_s

    while time.perf_counter() < t_hold_end:
        if _stopped():
            return

        pwm      = pid_ctrl.correct(fb, target_pc)
        pred_pos = pid_ctrl.predict_pos(fb)         # for logging

        opti_sample = opti.get_latest() if opti is not None else None
        if log_data:
            logger.log(pwm, fb.pc, target_pc, pred_pos=pred_pos, opti_sample=opti_sample)

        if plot_handles is not None:
            _update_plot(fb, opti, opti_sample, opti_trail_buf,
                         opti_origin_m, optitrack_init_ref, recorder)

    # Record OptiTrack arrival dot
    if (robot_trail_buf is not None and robot_trail_handles is not None
            and opti is not None and optitrack_init_ref is not None
            and not optitrack_init_ref[0]):
        hold_sample = opti.get_latest()
        pt = _opti_transform(opti, hold_sample, opti_origin_m)
        if pt is not None:
            robot_trail_buf.append(pt.copy())
            pts = np.vstack(robot_trail_buf)
            robot_trail_handles["xy"].set_data(pts[:, 0], pts[:, 1])
            robot_trail_handles["xz"].set_data(pts[:, 0], pts[:, 2])
            robot_trail_handles["yz"].set_data(pts[:, 1], pts[:, 2])


def _update_plot(fb, opti, opti_sample, opti_trail_buf,
                 opti_origin_m, optitrack_init_ref, recorder=None):
    OPTITRACK_TRAIL_LEN = 15

    if opti is not None and opti_sample is not None:
        if optitrack_init_ref[0]:
            opti_origin_m[:] = np.array(opti_sample.pos_xyz, dtype=float)
            opti_origin_m[1] += (fb.flowbot.l0 + fb.flowbot.lu) / 1000.0
            optitrack_init_ref[0] = False

        if not optitrack_init_ref[0]:
            transformed = _opti_transform(opti, opti_sample, opti_origin_m)
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

    ap = argparse.ArgumentParser(
        description="Execute a motion task with proprioception-model PID hold correction."
    )
    # ── Task / output ─────────────────────────────────────────────
    ap.add_argument("--task",          required=True,
                    help="Path to task file (defines get_waypoints).")
    ap.add_argument("--output", "-o",  default=None,
                    help="Output CSV path (auto-generated if omitted).")
    ap.add_argument("--repeat", "-n",  type=int, default=1,
                    help="Repeat the task N times (default 1).")
    # ── Hardware ──────────────────────────────────────────────────
    ap.add_argument("--baud",          type=int, default=115200)
    ap.add_argument("--pwm-min",       type=int, default=0)
    ap.add_argument("--pwm-max",       type=int, default=26)
    ap.add_argument("--pressure-model", choices=["learned", "linear"], default="linear")
    ap.add_argument("--max-pos-speed", type=float, default=30.0,
                    help="Max task-space speed in mm/s (default 30).")
    # ── OptiTrack ─────────────────────────────────────────────────
    ap.add_argument("--optitrack", "-opt", action="store_true", default=True)
    # ── Recording ─────────────────────────────────────────────────
    ap.add_argument("--no-plot",       action="store_true")
    ap.add_argument("--record",        action="store_true")
    ap.add_argument("--record-fps",    type=float, default=15.0)
    ap.add_argument("--record-output", default=None)
    # ── Task options ──────────────────────────────────────────────
    ap.add_argument("--seed",          type=int, default=None)
    ap.add_argument("--home-every",    type=int, default=None,
                    help="Return home after every N waypoints.")
    ap.add_argument("--home-rest",     type=float, default=20.0)
    ap.add_argument("--reverse",       action="store_true", default=False)
    # ── Proprioception PID ────────────────────────────────────────
    ap.add_argument("--propri-ckpt",
                    default="flowbot/proprioception_model/checkpoints/free_human/freeload",
                    help="Checkpoint directory for the proprioception model.")
    ap.add_argument("--pid-kp",    type=float, default=0.4,
                    help="Proportional gain (mm → mm).  Start here; tune before adding Ki.")
    ap.add_argument("--pid-ki",    type=float, default=0.01,
                    help="Integral gain.  Eliminate residual offset.  Keep small.")
    ap.add_argument("--pid-kd",    type=float, default=0.0,
                    help="Derivative gain.  Usually 0; flow noise amplifies this term.")
    ap.add_argument("--pid-iclamp", type=float, default=5.0,
                    help="Anti-windup clamp on the integral term (mm, default 5).")
    args = ap.parse_args()

    # ── Load task ─────────────────────────────────────────────────
    task_mod  = load_task_module(args.task)
    task_name = getattr(task_mod, "TASK_NAME", Path(args.task).stem)

    # ── Output paths ──────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        seed_tag = f"_s{args.seed}" if args.seed is not None else ""
        out_path = str(Path(FILE_DIR).parent / "data" / "task_logs"
                       / f"{task_name}{seed_tag}_pid_{ts}.csv")
    else:
        out_path = args.output if args.output.endswith(".csv") else args.output + ".csv"

    # ── Serial port ───────────────────────────────────────────────
    os_name = platform.system().lower()
    serial_port = "/dev/ttyACM0" if "linux" in os_name else "COM9"

    # ── Init flowbot ──────────────────────────────────────────────
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

    # ── Proprioception PID setup ──────────────────────────────────
    from flowbot.proprioception_model.propri_pid import (
        ProprioceptionPIDController, SerialReader,
    )
    fb.stop_flag["stop"] = True   # yield serial port to our reader
    time.sleep(0.15)
    fb.ser.reset_input_buffer()
    serial_reader = SerialReader(fb.ser)
    pid_ctrl = ProprioceptionPIDController(
        ckpt_dir       = args.propri_ckpt,
        reader         = serial_reader,
        Kp             = args.pid_kp,
        Ki             = args.pid_ki,
        Kd             = args.pid_kd,
        integral_limit = args.pid_iclamp,
    )
    print(f"[task] Propri-PID    : Kp={args.pid_kp}  Ki={args.pid_ki}  "
          f"Kd={args.pid_kd}  ckpt={args.propri_ckpt}")

    # ── Init OptiTrack ────────────────────────────────────────────
    opti = None
    opti_origin_m      = np.zeros(3, dtype=float)
    optitrack_init_ref = [True]
    opti_trail_buf     = []

    if args.optitrack:
        opti = MotiveNatNetReader(
            server_ip="150.65.146.84",
            local_ip="150.65.146.84",
            use_multicast=False,
            rigid_body_id=1,
        )
        opti.start()

    # ── Get waypoints ─────────────────────────────────────────────
    import inspect
    _gw_sig    = inspect.signature(task_mod.get_waypoints)
    _gw_kwargs = {}
    if "seed"    in _gw_sig.parameters and args.seed    is not None:
        _gw_kwargs["seed"]    = args.seed
    if "reverse" in _gw_sig.parameters:
        _gw_kwargs["reverse"] = args.reverse
    waypoints = task_mod.get_waypoints(fb, **_gw_kwargs)
    print(f"[task] '{task_name}'  {len(waypoints)} waypoints  x{args.repeat} repeats")
    print(f"[task] Pressure model : {args.pressure_model}")
    print(f"[task] Output CSV     : {out_path}")

    # ── Logger ────────────────────────────────────────────────────
    logger = TaskLogger(out_path)

    # ── Video recorder (optional) ─────────────────────────────────
    recorder = None
    if args.record and fb.fig is not None:
        vid_path = (args.record_output or out_path.replace(".csv", ".mp4"))
        if not vid_path.endswith(".mp4"):
            vid_path += ".mp4"
        recorder = VideoRecorder(vid_path, fps=args.record_fps, fig=fb.fig)
    elif args.record:
        print("[video] --record ignored: plot is disabled (--no-plot).")

    # ── Stop event ────────────────────────────────────────────────
    stop_event = threading.Event()

    # ── Plot wiring ───────────────────────────────────────────────
    plot_handles = fb.pl if not args.no_plot else None
    if fb.fig is not None:
        def _on_key(event):
            if event.key in ('q', 'escape'):
                print("\n[stop] 'q' pressed — stopping after current waypoint.")
                stop_event.set()
        def _on_close(_event):
            print("\n[stop] Plot window closed — stopping.")
            stop_event.set()
        fb.fig.canvas.mpl_connect('key_press_event', _on_key)
        fb.fig.canvas.mpl_connect('close_event',     _on_close)
        if hasattr(task_mod, "draw_reference"):
            task_mod.draw_reference(fb.axes, fb)

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

    # ── Run ───────────────────────────────────────────────────────
    _move_kwargs = dict(
        pid_ctrl=pid_ctrl,
        plot_handles=plot_handles,
        opti_trail_buf=opti_trail_buf,
        opti_origin_m=opti_origin_m,
        optitrack_init_ref=optitrack_init_ref,
        stop_event=stop_event,
        recorder=recorder,
        robot_trail_buf=robot_trail_buf,
        robot_trail_handles=robot_trail_handles,
    )

    home_pc        = np.asarray(fb.pc_init, dtype=float).reshape(3,)
    home_every     = args.home_every
    home_rest_s    = args.home_rest
    waypoint_count = 0

    try:
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

                if home_every is not None:
                    waypoint_count += 1
                    if waypoint_count % home_every == 0:
                        print(f"  [home-every] Returning home for {home_rest_s}s "
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
        serial_reader.stop()
        fb.stop()
        if opti is not None:
            opti.stop()
        print(f"Data saved to: {out_path}")


if __name__ == "__main__":
    main()
