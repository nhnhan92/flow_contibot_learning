"""
demo_live.py — Real-time proprioception model inference demo.

Layout:
  Left  : fixed-bounds 3D scatter  blue=pred  green=GT (OptiTrack)
  Right : rolling-window panels — X/Y/Z prediction error + flow sensors

Usage
-----
    python flowbot/proprioception_model/demo_live.py \
        --ckpt_dir flowbot/proprioception_model/checkpoints/free_human/freeload

    # Disable OptiTrack:
    python ... --no_optitrack

    # Record to MP4:
    python ... --record

    # Replay from a saved CSV (no hardware):
    python ... --csv_run data/flow_tip/seed1.csv

Terminal commands:
    r           random PWM
    5 10 20     set PWM
    q           quit

Coordinate frame
----------------
Model predicts in opti_to_manip frame (same as opti_x_mm training labels).
OptiTrack GT is also read in opti_to_manip frame.  No extra flip applied.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

_PROG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROG_DIR not in sys.path:
    sys.path.insert(0, _PROG_DIR)

from flowbot.proprioception_model.dataset import (
    StandardScaler,
    _compute_features,
    parse_feature_groups,
    FEATURE_NAMES,
)
from flowbot.proprioception_model.model import PropMLP, PlainMLP

BAUD_RATE = 115200
N_WINDOW  = 100


class SerialReader:
    """Background thread reading proc_flow1/2/3 + current PWM from Arduino CSV."""

    def __init__(self, ser, maxlen: int = 200):
        self._ser    = ser
        self._buf: deque = deque(maxlen=maxlen)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
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
            if len(parts) < 7:
                continue
            try:
                row = {
                    "proc_flow1": float(parts[2]),
                    "proc_flow2": float(parts[4]),
                    "proc_flow3": float(parts[6]),
                    # pwm_cur NOT read — it includes Arduino base offset (149+),
                    # but the model expects 0–26 command values.
                }
            except (ValueError, IndexError):
                continue
            with self._lock:
                self._buf.append(row)

    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def stop(self):
        self._stop.set()


def _apply_gt_signs(xyz: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Multiply each axis by its sign (+1 or -1) to align live GT with training frame."""
    return xyz.astype(float) * signs


def _default_port() -> str:
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(k in (p.description or "").lower()
               for k in ("arduino", "ch340", "cp210", "usb serial")):
            return p.device
    return ports[0].device if ports else ("COM9" if sys.platform == "win32" else "/dev/ttyACM0")



def main():
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",    required=True)
    parser.add_argument("--port",        default=None)
    parser.add_argument("--pwm_min",     type=int, default=0)
    parser.add_argument("--pwm_max",     type=int, default=26)
    parser.add_argument("--x_lim",      type=float, default=30.0)
    parser.add_argument("--z_lo",       type=float, default=-5.0)
    parser.add_argument("--z_hi",       type=float, default=25.0)
    parser.add_argument("--no_optitrack", action="store_true")
    parser.add_argument("--opti_ip",    default="150.65.146.84")
    parser.add_argument("--local_ip",   default="150.65.146.84")
    parser.add_argument("--opti_id",    type=int, default=1)
    parser.add_argument("--opti_alpha", type=float, default=-30.0)
    parser.add_argument("--record",     action="store_true")
    parser.add_argument("--csv_run",    default=None)
    parser.add_argument("--csv_delay",  type=float, default=0.05)
    parser.add_argument("--device",     default="cpu")
    # GT frame alignment: set sign per axis to match model training frame.
    # Default -1 -1 1 because current optitrack calibration has X,Y flipped vs training.
    # If you recalibrate and pred/GT go opposite again, try --gt_signs 1 1 1 first,
    # then -1 1 1 or 1 -1 1 until pred and GT move together.
    parser.add_argument("--gt_signs", type=float, nargs=3, default=[-1, -1, 1],
                        metavar=("SX", "SY", "SZ"),
                        help="Sign correction for live GT axes (default: -1 -1 1)")
    args = parser.parse_args()
    gt_signs = np.array(args.gt_signs, dtype=float)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = Path(args.ckpt_dir)
    with open(ckpt / "train_config.yaml") as f:
        cfg = yaml.safe_load(f)

    feature_indices = cfg.get("feature_indices") or parse_feature_groups(
        cfg.get("features", "flow,K,diff"))
    input_size = len(feature_indices)
    print(f"[demo] Features ({input_size}): {[FEATURE_NAMES[i] for i in feature_indices]}")

    arch   = cfg.get("arch",       "resmlp")
    hidden = cfg.get("hidden",     128)
    blocks = cfg.get("num_blocks", 3)

    if arch == "plainmlp":
        model = PlainMLP(input_size=input_size, hidden_size=hidden,
                         num_layers=blocks, dropout=0.0, output_size=3)
    else:
        model = PropMLP(input_size=input_size, hidden_size=hidden,
                        num_blocks=blocks, dropout=0.0, output_size=3)

    model.load_state_dict(torch.load(ckpt / "best_model.pt", map_location=args.device))
    model.eval()
    x_scaler = StandardScaler.load(ckpt / "scaler_x.pkl")
    y_scaler = StandardScaler.load(ckpt / "scaler_y.pkl")
    print(f"[demo] Checkpoint: {args.ckpt_dir}")

    @torch.no_grad()
    def _infer(pwm: np.ndarray, flow: np.ndarray) -> np.ndarray:
        X   = _compute_features(pwm.astype(np.float32),
                                 flow.astype(np.float32))[feature_indices]
        X_s = x_scaler.transform(X[None])
        y_s = model(torch.tensor(X_s, dtype=torch.float32)).numpy()
        return y_scaler.inverse_transform(y_s)[0]   # opti_to_manip frame (same as training)

    # ── Shared state ──────────────────────────────────────────────────────────
    _pwm  = np.zeros(3, dtype=np.float32)
    _flow: list[Optional[np.ndarray]] = [None]   # None until first sample

    _pred_buf = deque(maxlen=N_WINDOW)
    _gt_buf   = deque(maxlen=N_WINDOW)
    _flow_buf = deque(maxlen=N_WINDOW)
    _quit     = threading.Event()

    # ── Serial / CSV setup ────────────────────────────────────────────────────
    _ser        = None
    _ser_reader = None
    csv_rows    = None

    if args.csv_run is not None:
        import pandas as pd
        csv_rows = pd.read_csv(args.csv_run).to_dict("records")
        print(f"[demo] CSV replay: {len(csv_rows)} rows")
    else:
        import serial
        port        = args.port or _default_port()
        _ser        = serial.Serial(port, BAUD_RATE, timeout=1.0)
        time.sleep(0.5)
        _ser.reset_input_buffer()
        _ser_reader = SerialReader(_ser)
        print(f"[demo] Serial: {port}")

    # ── OptiTrack ─────────────────────────────────────────────────────────────
    opti        = None
    opti_origin = np.zeros(3)
    if not args.no_optitrack:
        from flowbot.online_optitrack import MotiveNatNetReader
        opti = MotiveNatNetReader(
            server_ip=args.opti_ip, local_ip=args.local_ip,
            use_multicast=False, rigid_body_id=args.opti_id,
            alpha=np.deg2rad(args.opti_alpha),
        )
        opti.start()
        time.sleep(1.0)
        s = opti.get_latest()
        if s is not None:
            opti_origin = np.array(s.pos_xyz)
            print(f"[demo] OptiTrack origin: {opti_origin}")
        else:
            print("[demo] OptiTrack: no sample yet")

    # ── PWM command ───────────────────────────────────────────────────────────
    def _send_pwm(pwm: np.ndarray):
        nonlocal _pwm
        _pwm = pwm.astype(np.float32)
        if _ser is not None:
            try:
                _ser.write(f"{int(pwm[0])} {int(pwm[1])} {int(pwm[2])}\n".encode())
            except Exception as e:
                print(f"  [serial] {e}")

    # ── Terminal input ────────────────────────────────────────────────────────
    def _terminal_loop():
        rng = np.random.default_rng()
        print("\n[demo] Commands: 'r' random | '5 10 20' set PWM | 'q' quit\n")
        while not _quit.is_set():
            try:
                cmd = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                _quit.set(); break
            if cmd in ("q", "quit"):
                _quit.set(); break
            elif cmd == "r":
                pwm = rng.integers(args.pwm_min, args.pwm_max + 1, size=3)
                _send_pwm(pwm); print(f"  → PWM {pwm}")
            else:
                parts = cmd.split()
                if len(parts) == 3:
                    try:
                        pwm = np.clip([int(p) for p in parts], args.pwm_min, args.pwm_max)
                        _send_pwm(np.array(pwm)); print(f"  → PWM {pwm}")
                    except ValueError:
                        print("  Use '5 10 20' or 'r'")
                elif cmd:
                    print("  Unknown command.")

    threading.Thread(target=_terminal_loop, daemon=True).start()

    # ── CSV replay thread ─────────────────────────────────────────────────────
    _csv_gt: list[Optional[np.ndarray]] = [None]

    if csv_rows is not None:
        _csv_has_gt = "opti_x_mm" in csv_rows[0]

        def _csv_loop():
            nonlocal _pwm
            for row in csv_rows:
                if _quit.is_set():
                    break
                try:
                    _pwm      = np.array([row["pwm1_cmd"],   row["pwm2_cmd"],   row["pwm3_cmd"]],
                                          dtype=np.float32)
                    _flow[0]  = np.array([row["proc_flow1"], row["proc_flow2"], row["proc_flow3"]],
                                          dtype=np.float32)
                except KeyError:
                    continue
                if _csv_has_gt:
                    try:
                        _csv_gt[0] = np.array([row["opti_x_mm"], row["opti_y_mm"], row["opti_z_mm"]],
                                               dtype=float)
                    except (KeyError, ValueError):
                        _csv_gt[0] = None
                time.sleep(args.csv_delay)
            _quit.set()

        threading.Thread(target=_csv_loop, daemon=True).start()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title("Proprioception — Live Demo")
    status_text = fig.text(
        0.5, 0.97,
        "Waiting for data…",
        ha="center", va="top",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#222222", alpha=0.75),
        color="white",
    )
    gs   = gridspec.GridSpec(4, 5, figure=fig, hspace=0.55, wspace=0.45,
                             top=0.91)
    ax3d = fig.add_subplot(gs[:, :2], projection="3d")
    ax_xe = fig.add_subplot(gs[0, 2:])
    ax_ye = fig.add_subplot(gs[1, 2:])
    ax_ze = fig.add_subplot(gs[2, 2:])
    ax_fl = fig.add_subplot(gs[3, 2:])

    lim = args.x_lim
    ax3d.set_xlim(-lim, lim);            ax3d.set_xlabel("X (mm)")
    ax3d.set_ylim(-lim, lim);            ax3d.set_ylabel("Y (mm)")
    ax3d.set_zlim(args.z_lo, args.z_hi); ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("Tip Position  blue=pred  green=GT", fontsize=9)
    pred_trace, = ax3d.plot([], [], [], color="tab:blue",  lw=1.0, alpha=0.5)
    gt_trace,   = ax3d.plot([], [], [], color="tab:green", lw=1.0, alpha=0.5)
    pred_dot,   = ax3d.plot([], [], [], "o", color="tab:blue",  ms=9,  label="Predicted")
    gt_dot,     = ax3d.plot([], [], [], "o", color="tab:green", ms=9,  label="GT (OptiTrack)")
    ax3d.legend(loc="upper left", fontsize=8)

    err_colors  = ["tab:red", "tab:orange", "tab:purple"]
    err_labels  = ["X error (mm)", "Y error (mm)", "Z error (mm)"]
    err_axes    = [ax_xe, ax_ye, ax_ze]
    err_titles  = ["X error", "Y error", "Z error"]
    err_lines   = []
    for ax, color, ylabel in zip(err_axes, err_colors, err_labels):
        line, = ax.plot([], [], color=color, lw=1.5)
        ax.axhline(0, color="k", lw=1.2, ls="--")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        err_lines.append(line)
    for ax, tb in zip(err_axes, err_titles):
        ax.set_title(f"{tb}  MAE=— mm", fontsize=8)
    ax_ze.set_xlabel("Sample", fontsize=8)

    fl_lines = [ax_fl.plot([], [], color=c, lw=1.2, label=f"flow{i+1}")[0]
                for i, c in enumerate(["tab:blue", "tab:orange", "tab:green"])]
    ax_fl.set_title("Flow sensors", fontsize=8)
    ax_fl.set_ylabel("Flow (L/min)", fontsize=8)
    ax_fl.set_xlabel("Sample", fontsize=8)
    ax_fl.tick_params(labelsize=7)
    ax_fl.legend(fontsize=7, loc="upper right")

    fig.canvas.mpl_connect("close_event", lambda _: _quit.set())
    fig.canvas.mpl_connect("key_press_event",
                            lambda e: _quit.set() if e.key in ("q", "Q", "escape") else None)

    # ── Video recorder ────────────────────────────────────────────────────────
    recorder = None
    if args.record:
        from flowbot.video_recorder import VideoRecorder
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        vid_path = str(ckpt / f"demo_{ts}.mp4")
        recorder = VideoRecorder(vid_path, fps=15.0, fig=fig)
        print(f"[demo] Recording: {vid_path}")

    # ── Animation update ──────────────────────────────────────────────────────
    def _update(_):
        nonlocal _pwm

        if _quit.is_set():
            plt.close(fig)
            return

        # Pull from serial reader (live mode)
        # _pwm is NOT updated here — it tracks the 0–26 command sent from Python.
        # pwm1_cur from Arduino includes the base offset (149+) and would corrupt K features.
        if _ser_reader is not None:
            row = _ser_reader.latest()
            if row is not None:
                _flow[0] = np.array([row["proc_flow1"], row["proc_flow2"], row["proc_flow3"]],
                                     dtype=np.float32)

        if _flow[0] is None:
            return

        # Predict (opti_to_manip frame — same as training labels)
        pred = _infer(_pwm, _flow[0])

        # Ground truth: live optitrack has X,Y flipped vs training frame → negate X,Y.
        # CSV GT from old training files is already in training frame → no flip.
        gt = None
        if opti is not None:
            s = opti.get_latest()
            if s is not None:
                gt = _apply_gt_signs(opti.opti_to_manip(np.array(s.pos_xyz), opti_origin), gt_signs)
        elif _csv_gt[0] is not None:
            gt = _csv_gt[0].copy()

        _pred_buf.append(pred.copy())
        _gt_buf.append(gt)
        _flow_buf.append(_flow[0].copy())

        # Status bar
        p, f = _pwm, _flow[0]
        gt_str = (f"GT=({gt[0]:.1f}, {gt[1]:.1f}, {gt[2]:.1f}) mm"
                  if gt is not None else "GT=—")
        status_text.set_text(
            f"PWM [{int(p[0]):3d} {int(p[1]):3d} {int(p[2]):3d}]"
            f"  Flow [{f[0]:.2f} {f[1]:.2f} {f[2]:.2f}] L/min"
            f"  pred=({pred[0]:.1f}, {pred[1]:.1f}, {pred[2]:.1f}) mm"
            f"  {gt_str}"
        )

        # 3-D traces + dots
        pred_arr = np.array(list(_pred_buf))
        pred_trace.set_data(pred_arr[:, 0], pred_arr[:, 1])
        pred_trace.set_3d_properties(pred_arr[:, 2])
        pred_dot.set_data([pred[0]], [pred[1]])
        pred_dot.set_3d_properties([pred[2]])

        gt_with_pos = [g for g in _gt_buf if g is not None]
        if gt_with_pos:
            gt_arr = np.array(gt_with_pos)
            gt_trace.set_data(gt_arr[:, 0], gt_arr[:, 1])
            gt_trace.set_3d_properties(gt_arr[:, 2])
        if gt is not None:
            gt_dot.set_data([gt[0]], [gt[1]])
            gt_dot.set_3d_properties([gt[2]])

        # Error panels
        n         = len(_pred_buf)
        pred_list = list(_pred_buf)
        gt_list   = list(_gt_buf)
        err_idx   = [i for i, g in enumerate(gt_list) if g is not None]
        if err_idx:
            err_xs  = np.array(err_idx)
            err_mat = np.array([pred_list[i] - gt_list[i] for i in err_idx])
            for dim, (line, ax, tb) in enumerate(zip(err_lines, err_axes, err_titles)):
                line.set_data(err_xs, err_mat[:, dim])
                mae = float(np.mean(np.abs(err_mat[:, dim])))
                ax.set_title(f"{tb}  MAE={mae:.2f} mm", fontsize=8)
                ax.set_xlim(0, max(n, 1))
                ax.relim(); ax.autoscale_view(scalex=False)
        else:
            for line, ax, tb in zip(err_lines, err_axes, err_titles):
                line.set_data([], [])
                ax.set_title(f"{tb}  MAE=— mm", fontsize=8)

        # Flow panel
        fl_arr = np.array(list(_flow_buf))
        for dim, line in enumerate(fl_lines):
            line.set_data(np.arange(len(fl_arr)), fl_arr[:, dim])
        ax_fl.set_xlim(0, max(n, 1))
        ax_fl.relim(); ax_fl.autoscale_view(scalex=False)

        if recorder is not None:
            recorder.capture()

    fig._ani = animation.FuncAnimation(fig, _update,
                                       interval=80, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        _quit.set()
        if recorder is not None:
            recorder.close()
        if _ser_reader is not None:
            _ser_reader.stop()
        if _ser is not None:
            try:
                _ser.write(b"0 0 0\n")
                _ser.close()
            except Exception:
                pass
        if opti is not None:
            opti.stop()
        print("[demo] Done.")


if __name__ == "__main__":
    main()
