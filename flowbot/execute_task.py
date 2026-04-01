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
# Video Recorder
# ──────────────────────────────────────────────────────────────
class VideoRecorder:
    """
    Captures matplotlib figure frames and writes them to a video/GIF file.

    Priority:
      1. MP4 via imageio+ffmpeg  (pip install imageio imageio-ffmpeg)
      2. Animated GIF via Pillow (already installed — no extra deps)

    Output path should end in .mp4 or .gif; .gif is used automatically
    if imageio-ffmpeg is not available.
    """

    def __init__(self, path: str, fps: float = 15.0, fig=None):
        self._path           = path
        self._fps            = fps
        self._fig            = fig
        self._frames         = []   # PIL Image list (GIF mode)
        self._writer         = None
        self._frame_idx      = 0
        self._mode           = None   # "imageio" | "gif"
        self._frame_interval  = 1.0 / fps   # min seconds between captured frames
        self._last_capture_t  = -1e9       # force capture on first call
        self._capture_times   = []         # wall-clock time of each captured frame

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # ── Try MP4 via imageio ──────────────────────────────────
        try:
            import imageio
            # imageio v3 uses imageio.v3; v2 uses imageio directly — try both
            try:
                writer = imageio.get_writer(
                    path, fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                    macro_block_size=1,
                )
            except TypeError:
                writer = imageio.get_writer(path, fps=fps)
            self._writer = writer
            self._mode   = "imageio"
            print(f"[video] Recording MP4 → {path}  ({fps:.0f} fps)")
            return
        except Exception as e:
            print(f"[video] imageio MP4 unavailable ({e}). Falling back to GIF.")

        # ── Fallback: animated GIF via Pillow ────────────────────
        try:
            if importlib.util.find_spec("PIL") is None:
                raise ImportError("Pillow not found")
            gif_path = Path(path).with_suffix(".gif")
            self._path = str(gif_path)
            self._mode = "gif"
            print(f"[video] Recording animated GIF → {self._path}  ({fps:.0f} fps)")
            print(f"[video] Tip: install imageio-ffmpeg for MP4:  "
                  f"pip install imageio imageio-ffmpeg")
        except ImportError:
            print("[video] Neither imageio nor Pillow available — recording disabled.")

    def _grab_frame(self):
        """Return current figure canvas as an RGB numpy array (even dimensions for H.264)."""
        # Do NOT call canvas.draw() here — fb.update_plot() already rendered the frame.
        # Calling draw() again would slow the control loop and cause fps mismatch.
        rgba_buf = self._fig.canvas.buffer_rgba()
        # physical=True accounts for Windows DPI scaling (e.g. 125% → 1.25× buffer)
        canvas_w, canvas_h = self._fig.canvas.get_width_height(physical=True)
        frame = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(canvas_h, canvas_w, 4)
        rgb = frame[..., :3]   # drop alpha → RGB
        # H.264 requires even dimensions — crop 1px if odd
        h, w = rgb.shape[:2]
        return rgb[:h & ~1, :w & ~1]

    def capture(self):
        """Grab one frame from the figure canvas (rate-limited to self._fps)."""
        if self._fig is None or self._mode is None:
            return
        now = time.perf_counter()
        if now - self._last_capture_t < self._frame_interval:
            return   # too soon — skip this tick
        self._last_capture_t = now
        try:
            frame = self._grab_frame()
            if self._mode == "imageio":
                self._writer.append_data(frame)
            elif self._mode == "gif":
                from PIL import Image
                self._frames.append(Image.fromarray(frame))
            self._capture_times.append(now)
            self._frame_idx += 1
        except Exception as exc:
            # Print first failure only to avoid log spam
            if self._frame_idx == 0:
                print(f"[video] Frame capture error: {exc}")

    def _actual_fps_str(self):
        if len(self._capture_times) >= 2:
            actual = (len(self._capture_times) - 1) / \
                     (self._capture_times[-1] - self._capture_times[0])
            return f"{actual:.1f} fps actual"
        return ""

    def _correct_mp4_speed(self, actual_fps: float):
        """Rewrite the MP4 with setpts so playback matches real wall-clock time."""
        ratio = self._fps / actual_fps   # > 1 → slow down; < 1 → speed up
        if abs(ratio - 1.0) <= 0.05:
            return   # within 5% — no correction needed
        try:
            import imageio_ffmpeg, subprocess, os
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            tmp = self._path + ".tmp.mp4"
            cmd = [
                ffmpeg_exe, "-y", "-i", self._path,
                "-vf", f"setpts={ratio:.6f}*PTS",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                tmp,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                os.replace(tmp, self._path)
                print(f"[video] Speed-corrected: {actual_fps:.1f} fps actual → "
                      f"video plays at real-time  (stretched {ratio:.2f}×)")
            else:
                print(f"[video] Speed correction failed; video plays at "
                      f"{actual_fps:.1f}/{self._fps:.0f} = "
                      f"{actual_fps/self._fps:.2f}× real speed")
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as exc:
            print(f"[video] Speed correction skipped ({exc}); "
                  f"video plays {self._fps/actual_fps:.2f}× real speed")

    def close(self):
        if self._mode == "imageio" and self._writer is not None:
            self._writer.close()
            if len(self._capture_times) >= 2:
                actual_fps = (len(self._capture_times) - 1) / \
                             (self._capture_times[-1] - self._capture_times[0])
                self._correct_mp4_speed(actual_fps)
            print(f"[video] Saved {self._frame_idx} frames → {self._path}"
                  f"  ({self._actual_fps_str()})")
        elif self._mode == "gif" and self._frames:
            # Use actual inter-frame intervals for correct real-time playback.
            # GIF stores delay in centiseconds (10 ms steps); minimum is 20 ms.
            if len(self._capture_times) >= 2:
                durations = []
                for i in range(1, len(self._capture_times)):
                    dt_ms = (self._capture_times[i] - self._capture_times[i - 1]) * 1000
                    # round to nearest 10 ms (centisecond resolution), floor at 20 ms
                    durations.append(max(20, round(dt_ms / 10) * 10))
                durations.append(durations[-1])   # repeat duration for last frame
            else:
                durations = int(1000 / self._fps)
            self._frames[0].save(
                self._path,
                save_all=True,
                append_images=self._frames[1:],
                duration=durations,
                loop=0,
            )
            print(f"[video] Saved {self._frame_idx} frames → {self._path}"
                  f"  ({self._actual_fps_str()})")


# ──────────────────────────────────────────────────────────────
# CSV Logger
# ──────────────────────────────────────────────────────────────
class TaskLogger:
    """Logs timestamped rows: t, pwm, pc (commanded), optitrack pos + quat."""

    HEADER = [
        "t_s",
        "pwm_1", "pwm_2", "pwm_3",
        "cmd_pc_x", "cmd_pc_y", "cmd_pc_z",
        "opti_x", "opti_y", "opti_z",
        "opti_qx", "opti_qy", "opti_qz", "opti_qw",
    ]

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)
        self._t0 = time.perf_counter()
        print(f"[logger] Writing to {path}")

    def log(self, pwm, pc, opti_sample=None):
        t = time.perf_counter() - self._t0
        pwm = np.asarray(pwm, dtype=int).reshape(3,)
        pc  = np.asarray(pc,  dtype=float).reshape(3,)
        if opti_sample is not None:
            ox, oy, oz = opti_sample.pos_xyz
            qx, qy, qz, qw = opti_sample.quat_xyzw
        else:
            ox = oy = oz = qx = qy = qz = qw = float("nan")
        self._w.writerow([
            f"{t:.4f}",
            int(pwm[0]), int(pwm[1]), int(pwm[2]),
            f"{pc[0]:.4f}", f"{pc[1]:.4f}", f"{pc[2]:.4f}",
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


def _opti_transform(opti, opti_sample, opti_origin_m):
    """Apply opti_to_manip and flip X/Y axes. Returns transformed 3-vector or None."""
    if opti is None or opti_sample is None:
        return None
    t = opti.opti_to_manip(np.array(opti_sample.pos_xyz, dtype=float), opti_origin_m)
    t[0] = -t[0]
    t[1] = -t[1]
    return t

def move_to_waypoint(fb, target_pc, hold_s, logger, opti,
                     plot_handles=None, opti_trail_buf=None,
                     opti_origin_m=None, optitrack_init_ref=None,
                     stop_event: threading.Event = None,
                     recorder=None,
                     robot_trail_buf=None, robot_trail_handles=None,
                     log_data: bool = True):
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
            logger.log(pwm, fb.pc, opti_sample)

        if plot_handles is not None:
            _update_plot(fb, opti, opti_sample, opti_trail_buf,
                         opti_origin_m, optitrack_init_ref, recorder)

    # ── Phase 2: hold at target ───────────────────────────────
    t_hold_end = time.perf_counter() + hold_s
    while time.perf_counter() < t_hold_end:
        if _stopped():
            return

        pwm = fb.step(np.zeros(3))
        opti_sample = opti.get_latest() if opti is not None else None
        if log_data:
            logger.log(pwm, fb.pc, opti_sample)

        if plot_handles is not None:
            _update_plot(fb, opti, opti_sample, opti_trail_buf,
                         opti_origin_m, optitrack_init_ref, recorder)

    # Record the OptiTrack position at end of hold as a waypoint dot
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
    """Update the 2D projection plot (pc + optitrack trail), then capture a frame."""
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
    ap.add_argument("--seed",           type=int, default=None,
                    help="Random seed passed to get_waypoints (useful for random task).")
    ap.add_argument("--home-every",     type=int, default=None,
                    help="Return to home and rest after every N waypoints (disabled if omitted).")
    ap.add_argument("--home-rest",      type=float, default=20.0,
                    help="Hold time at home during periodic rest (default 20.0 s).")
    ap.add_argument("--reverse",        action="store_true", default=False,
                    help="For tasks with forward+reverse waypoints (e.g. lemniscate), "
                         "only do the reverse (inner→outer) half.")
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
    if args.output is None:
        seed_tag = f"_s{args.seed}" if args.seed is not None else ""
        out_path = str(Path(FILE_DIR).parent / "data" / "task_logs"
                       / f"{task_name}{seed_tag}_{ts}.csv")
    else:
        out_path = args.output if args.output.endswith(".csv") else args.output + ".csv"

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

    # ── Get waypoints ────────────────────────────────────────────
    import inspect
    _gw_sig   = inspect.signature(task_mod.get_waypoints)
    _gw_kwargs = {}
    if "seed"    in _gw_sig.parameters and args.seed    is not None:
        _gw_kwargs["seed"]    = args.seed
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
    )

    home_pc        = np.asarray(fb.pc_init, dtype=float).reshape(3,)
    home_every     = args.home_every    # None = disabled
    home_rest_s    = args.home_rest
    waypoint_count = 0                  # counts executed non-home waypoints

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
        fb.stop()
        if opti is not None:
            opti.stop()
        print(f"Data saved to: {out_path}")


if __name__ == "__main__":
    main()
