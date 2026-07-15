#!/usr/bin/env python3
"""
collect_free_human.py — Automated data collection for flow proprioception model (free_human only).

Identical to collect_flow_tip.py but skips the human-interaction phase entirely.
State machine per waypoint:
  MOVE      → fb.step() loop toward random target (not logged)
  SETTLE    → settling_s pause after arrival
  EQUIL_LOG → log equil_s rows as state="free_human"
  → next waypoint  (fully automated, no Enter press needed)

CSV columns:
  current_time, t_s, state,
  pwm1_cmd, pwm2_cmd, pwm3_cmd,
  proc_flow1, proc_flow2, proc_flow3,
  opti_x_mm, opti_y_mm, opti_z_mm
"""

import argparse
import csv
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import serial
import serial.tools.list_ports

# ── sys.path: allow running directly from this file's directory ───────────────
_prog_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _prog_dir not in sys.path:
    sys.path.insert(0, _prog_dir)

from flowbot.online_optitrack import MotiveNatNetReader
from flowbot.video_recorder import CameraRecorder

BAUD_RATE  = 115200
ARRIVAL_MM = 1.0   # mm — close enough to declare waypoint reached


# ── Serial helpers ────────────────────────────────────────────────────────────

def _default_port() -> str:
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ("arduino", "ch340", "cp210", "usb serial")):
            return p.device
    if ports:
        return ports[0].device
    return "COM9" if sys.platform == "win32" else "/dev/ttyACM0"


class SerialReader:
    """
    Background thread that reads Arduino sensor CSV lines from fb.ser.
    Skips ACK lines (start with "ACK") and non-data lines.

    Arduino CSV column order:
      t_ms, rawFlow1, proc_flow1, rawFlow2, proc_flow2,
      rawFlow3, proc_flow3, rawPress, pressMPa, pwm1, pwm2, pwm3
    """

    def __init__(self, ser: serial.Serial, maxlen: int = 400):
        self._ser = ser
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                break
            if not line or not line[0].isdigit():
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            try:
                row = {
                    "proc_flow1": float(parts[2]),
                    "proc_flow2": float(parts[4]),
                    "proc_flow3": float(parts[6]),
                }
            except ValueError:
                continue
            with self._lock:
                self._buf.append((time.perf_counter(), row))

    def drain(self):
        with self._lock:
            self._buf.clear()

    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._buf[-1][1] if self._buf else None

    def stop(self):
        self._stop.set()


# ── OptiTrack helper ──────────────────────────────────────────────────────────

def _opti_pos(opti: Optional[MotiveNatNetReader],
              origin: np.ndarray) -> Tuple[float, float, float]:
    if opti is None:
        return float("nan"), float("nan"), float("nan")
    sample = opti.get_latest()
    if sample is None:
        return float("nan"), float("nan"), float("nan")
    mm = opti.opti_to_manip(sample.pos_xyz, origin)
    return float(mm[0]), float(mm[1]), float(mm[2])


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_timed(writer, f_csv, reader: SerialReader,
              opti: Optional[MotiveNatNetReader], origin: np.ndarray,
              pwm_cmd: np.ndarray, state: str, duration_s: float, t0: float) -> int:
    """Log at ~20 Hz for duration_s seconds with a fixed PWM command. Returns row count."""
    reader.drain()
    deadline = time.perf_counter() + duration_s
    logged = 0
    while time.perf_counter() < deadline:
        row = reader.latest()
        if row is not None:
            ox, oy, oz = _opti_pos(opti, origin)
            writer.writerow([
                datetime.now().strftime("%H:%M:%S.%f"),
                f"{time.perf_counter() - t0:.4f}",
                state,
                int(pwm_cmd[0]), int(pwm_cmd[1]), int(pwm_cmd[2]),
                f"{row['proc_flow1']:.4f}",
                f"{row['proc_flow2']:.4f}",
                f"{row['proc_flow3']:.4f}",
                f"{ox:.3f}", f"{oy:.3f}", f"{oz:.3f}",
            ])
            logged += 1
        time.sleep(0.05)
    f_csv.flush()
    return logged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect free_human flow + OptiTrack tip data (automated, no human interaction)."
    )
    parser.add_argument("--port", default=_default_port(),
                        help="Arduino serial port")
    parser.add_argument("--n_waypoints", type=int, default=100,
                        help="Number of random waypoints (default 60)")
    parser.add_argument("--equil_s", type=float, default=0.1,
                        help="Equilibrium logging duration after each step (default 0.2 s)")
    parser.add_argument("--settling_s", type=float, default=1.2,
                        help="Delay after each step before logging (default 2.0 s)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default data/flow_tip_free/free_<timestamp>.csv)")
    parser.add_argument("--opti_ip", default="150.65.146.84",
                        help="OptiTrack server IP")
    parser.add_argument("--local_ip", default="150.65.146.84",
                        help="Local IP for NatNet multicast")
    parser.add_argument("--opti_id", type=int, default=1,
                        help="Rigid body ID in Motive (default 1)")
    parser.add_argument("--opti_alpha", type=float, default=-30.0,
                        help="OptiTrack Z-rotation offset in degrees (default -30)")
    parser.add_argument("--no_optitrack", action="store_true",
                        help="Skip OptiTrack; log nan for tip position")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--pwm_min", type=int, default=0,
                        help="Minimum PWM value sent to each actuator (default 0)")
    parser.add_argument("--pwm_max", type=int, default=26,
                        help="Maximum PWM value sent to each actuator (default 26)")
    parser.add_argument("--max_pos_speed", type=float, default=20.0,
                        help="Max task-space speed in mm/s (default 50)")
    parser.add_argument("--frequency", type=float, default=10.0,
                        help="Control loop frequency in Hz (default 20)")
    parser.add_argument("--pressure_model", choices=["learned", "linear"], default="linear",
                        help="IK pressure model: 'learned' (pkl) or 'linear' (default linear)")
    parser.add_argument("--camera", action="store_true",
                        help="Record USB camera video alongside the CSV")
    parser.add_argument("--camera_id", type=int, default=1,
                        help="USB camera device index (default 1)")
    parser.add_argument("--camera_fps", type=float, default=15.0,
                        help="Camera recording fps (default 15)")
    parser.add_argument("--load", type=int, default=None,
                        help="Load at the End effector")
    parser.add_argument("--min_xy_dist", type=float, default=10.0,
                        help="Min XY radial distance from workspace centre (mm). "
                             "Biases sampling toward large bending angles. Default 0 (no constraint)")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Output CSV path ───────────────────────────────────────────────────────
    if args.output is None:
        folder = "data/flow_tip_free_200g"
        os.makedirs(folder, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.load is None:
            args.output = f"{folder}/free_seed{args.seed}_{ts}.csv"
        else:
            args.output = f"{folder}/free_load{args.load}_seed{args.seed}_{ts}.csv"
    else:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ── Init flowbot (IK model + workspace) ───────────────────────────────────
    print(f"[flowbot] Initializing on {args.port} (pressure_model={args.pressure_model}) ...")
    from learning.hardware import flowbot as flowbot_module
    fb = flowbot_module.flowbot(
        serial_port    = args.port,
        baud           = BAUD_RATE,
        pwm_min        = args.pwm_min,
        pwm_max        = args.pwm_max,
        enable_plot    = False,
        frequency      = args.frequency,
        max_pos_speed  = args.max_pos_speed,
        pressure_model = args.pressure_model,
    )
    fb.start()

    fb.stop_flag["stop"] = True
    time.sleep(0.25)
    print("[flowbot] Internal reader stopped; sensor reader starting.")

    fb.ser.reset_input_buffer()
    reader = SerialReader(fb.ser)

    # ── OptiTrack ─────────────────────────────────────────────────────────────
    opti: Optional[MotiveNatNetReader] = None
    opti_origin = np.zeros(3)
    if not args.no_optitrack:
        print(f"[optitrack] Connecting to {args.opti_ip} ...")
        opti = MotiveNatNetReader(
            server_ip=args.opti_ip,
            local_ip=args.local_ip,
            use_multicast=False,
            rigid_body_id=args.opti_id,
            alpha=np.deg2rad(args.opti_alpha),
        )
        opti.start()
        time.sleep(1.0)
        sample = opti.get_latest()
        if sample is not None:
            opti_origin = np.array(sample.pos_xyz)
            print(f"[optitrack] Origin set to {opti_origin}")
        else:
            print("[optitrack] WARNING: no sample yet — origin stays [0,0,0]")
    else:
        print("[optitrack] Skipped (--no_optitrack).")

    # ── CSV writer ────────────────────────────────────────────────────────────
    f_csv = open(args.output, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow([
        "current_time", "t_s", "state",
        "pwm1_cmd", "pwm2_cmd", "pwm3_cmd",
        "proc_flow1", "proc_flow2", "proc_flow3",
        "opti_x_mm", "opti_y_mm", "opti_z_mm",
    ])

    # ── Camera ────────────────────────────────────────────────────────────────
    camera_rec: Optional[CameraRecorder] = None
    if args.camera:
        cam_path = args.output.replace(".csv", "_camera.mp4")
        try:
            camera_rec = CameraRecorder(cam_path, camera_id=args.camera_id,
                                        fps=args.camera_fps)
            camera_rec.start()
        except ImportError as e:
            print(f"[camera] {e} — camera recording skipped.")
            camera_rec = None

    # ── Workspace ─────────────────────────────────────────────────────────────
    bbox_lo, bbox_hi = fb.bbox
    print(f"\n[collect] Workspace bbox:")
    print(f"           lo = {np.round(bbox_lo, 1)} mm")
    print(f"           hi = {np.round(bbox_hi, 1)} mm")

    t0 = time.perf_counter()
    print(f"[collect] Starting {args.n_waypoints} waypoints (free_human only, automated).")
    print(f"[collect] Output  : {args.output}")
    print("[collect] Ctrl+C to stop cleanly.\n")

    
    try:
        fb.step(np.array([0.0, 0.0, 10.0])) 
        input("   Press Enter when ready to begin... ")
        print(" Warmup step done. Starting main loop...")
        for wp_idx in range(args.n_waypoints):
            # Sample a random target inside the workspace convex hull
            for _ in range(10000):
                target = rng.uniform(bbox_lo, bbox_hi)
                if (fb.ws.is_inside_workspace(target, fb.tri)
                        and target[2] > fb.pc_init[2] + 5.0
                        and target[2] < bbox_hi[2] - 1.0
                        and target[0]**2 + target[1]**2 > args.min_xy_dist**2):
                    break
            else:
                print("[collect] WARNING: workspace rejection sampler failed — skipping waypoint")
                continue

            print(f"\n── Waypoint {wp_idx + 1}/{args.n_waypoints}  "
                  f"target={np.round(target, 1)} mm ──")

            # ── MOVE: drive toward target (no logging) ────────────────────────
            max_iters = 20
            step_count = 0
            for _ in range(max_iters):
                dist = float(np.linalg.norm(target - fb.pc))
                print(f"\r   Moving...  dist={dist:.2f} mm  ", end="")
                if dist < ARRIVAL_MM:
                    break
                d         = target - fb.pc
                direction = d / (np.linalg.norm(d) + 1e-12)
                fb.step(direction)
                # fb.step(np.array([0.0, 0.0, 0.0]))
                step_count += 1

            dist_final = float(np.linalg.norm(target - fb.pc))
            print(f"   Arrived: {step_count} steps  dist={dist_final:.2f} mm")

            # ── SETTLE ────────────────────────────────────────────────────────
            print(f"   Settling {args.settling_s:.1f} s...")
            if ((wp_idx - 1) % 25 == 0 and wp_idx > 0) or wp_idx == 0:
                print("   (First waypoint — allowing extra settling time)")
                time.sleep(args.settling_s + 1.5)
            else:
                time.sleep(args.settling_s)

            # ── EQUIL_LOG: log equilibrium as "free_human" ────────────────────
            print(f"   Logging equilibrium ({args.equil_s:.1f} s)...")
            n_free = log_timed(writer, f_csv, reader, opti, opti_origin,
                               fb.last_pwm, "free_human", args.equil_s, t0)
            print(f"   → {n_free} rows (free_human)")

            if wp_idx % 25 == 0 and wp_idx > 0:
                print(f"\n[collect] Reached waypoint {wp_idx}/{args.n_waypoints} "
                      f"— pausing briefly to reset robot.")
                try:
                    fb.reset()
                    time.sleep(20.0)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n[collect] Interrupted.")
    finally:
        if camera_rec is not None:
            camera_rec.stop()
        print("[collect] Resetting robot (PWM → 0 0 0)...")
        try:
            fb.reset()
            time.sleep(0.5)
        except Exception:
            pass
        reader.stop()
        f_csv.flush()
        f_csv.close()
        try:
            fb.ser.close()
        except Exception:
            pass
        if opti is not None:
            opti.stop()
        print(f"[collect] Done. Data saved to: {args.output}")


if __name__ == "__main__":
    main()
