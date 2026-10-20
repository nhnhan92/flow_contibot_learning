import serial
import serial.tools.list_ports
import csv
import sys
import time
from datetime import datetime
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import numpy as np

try:
    from flowbot.video_recorder import VideoRecorder, CameraRecorder
    _CV2_AVAILABLE = True
except ImportError:
    VideoRecorder    = None
    CameraRecorder   = None
    _CV2_AVAILABLE   = False

try:
    from flowbot.online_optitrack import MotiveNatNetReader, FLOWBOT_MARKER_POSITIONS_M
    _OPTI_AVAILABLE = True
except ImportError:
    MotiveNatNetReader = None
    FLOWBOT_MARKER_POSITIONS_M = None
    _OPTI_AVAILABLE   = False

BAUD_RATE = 115200
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV  = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

MAX_POINTS = 1000

def reader_logger(ser, writer, buffers, stop_flag, done_event=None):
    """
    Pure serial reader — runs in a background thread.
    Reads lines from Arduino, writes CSV, fills shared buffers.
    No matplotlib calls here.

    Arduino CSV column order:
      t_ms, rawFlow1, proc_flow1, rawFlow2, proc_flow2,
      rawFlow3, proc_flow3, rawPress, press_kPa, pwm1, pwm2, pwm3
    (press_kPa: vacuum pressure from the MPS-V35R-NCA, negative = vacuum)

    done_event : set() whenever a "DONE" line arrives (arduino_controller.ino
                 prints this once a PWM command's ramp has actually reached
                 its target -- see replay_pwm_csv, which waits on it instead
                 of a fixed delay).
    """
    header_line = ser.readline().decode("utf-8", errors="ignore").strip()
    print("Header:", header_line)
    if not header_line.startswith("t_ms"):
        header_line = ("t_ms,raw_flow1,proc_flow1,raw_flow2,proc_flow2,"
                       "raw_flow3,proc_flow3,raw_press,press_kPa,"
                       "pwm1_cur,pwm2_cur,pwm3_cur")
    writer.writerow(["current_time"] + header_line.split(","))

    t0 = None
    prev_t_ms = None

    while not stop_flag["stop"]:
        raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw_line:
            continue
        if raw_line == "DONE":
            if done_event is not None:
                done_event.set()
            continue
        current_time = datetime.now().strftime("%H:%M:%S.%f")
        parts = raw_line.split(",")
        if len(parts) < 9:
            continue
        writer.writerow([current_time] + parts)

        try:
            t_ms = float(parts[0])
            r1   = float(parts[1]);  pf1 = float(parts[2])
            r2   = float(parts[3]);  pf2 = float(parts[4])
            r3   = float(parts[5]);  pf3 = float(parts[6])
        except ValueError:
            continue

        if prev_t_ms is not None and t_ms < prev_t_ms:
            print("### Arduino reset — re-zeroing time.")
            t0 = None
            for b in buffers.values():
                b.clear()

        prev_t_ms = t_ms
        if t0 is None:
            t0 = t_ms
        t_s = (t_ms - t0) / 1000.0

        try:
            press = float(parts[8])
        except (ValueError, IndexError):
            press = 0.0

        buffers["t"].append(t_s)
        buffers["r1"].append(r1);  buffers["r2"].append(r2);  buffers["r3"].append(r3)
        buffers["p1"].append(pf1); buffers["p2"].append(pf2); buffers["p3"].append(pf3)
        buffers["press"].append(press)

        if len(buffers["t"]) > MAX_POINTS:
            for b in buffers.values():
                b[:] = b[-MAX_POINTS:]


def replay_pwm_csv(ser, csv_path, settle_s, output_path, buffers, done_event,
                   max_wait_s=3.0, poll_s=0.01, opti=None, opti_origin=None):
    """
    Send each (pwm1, pwm2, pwm3) row of csv_path to the Arduino in turn.

    Instead of a fixed wait, each set now waits for the Arduino's "DONE"
    feedback (arduino_controller.ino prints this once that set's PWM ramp
    has actually reached its target -- see reader_logger/done_event), capped
    at max_wait_s as a safety fallback in case a DONE is missed (e.g. a
    dropped line). After DONE (or the timeout), an extra settle_s dwell
    follows so the sensors can stabilize at the now-steady-state PWM.

    Every new sensor sample from the serial reader (buffers) is written as
    one row labelled with that set's commanded PWM, throughout both phases.
    Ends by commanding 0 0 0.

    opti, opti_origin : optional MotiveNatNetReader + origin (m), as set up
                         by the "csv" input-loop branch -- when given, each
                         logged row also includes the rigid body's manip-
                         frame position (mm, via opti.opti_to_manip), same
                         convention as the "elongation" command's opti_*_mm
                         columns. Left as NaN when opti is None/no sample.
    """
    with open(csv_path, "r", newline="") as fcsv:
        rows = [(int(float(r["pwm1"])), int(float(r["pwm2"])), int(float(r["pwm3"])))
                for r in csv.DictReader(fcsv)]
    print(f"[csv] Replaying {len(rows)} PWM sets from {csv_path} "
          f"(wait for DONE, max {max_wait_s}s, then settle {settle_s}s)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as fout:
        wo = csv.writer(fout)
        wo.writerow(["current_time", "set_idx", "cmd_pwm1", "cmd_pwm2", "cmd_pwm3",
                     "t_s", "raw_flow1", "raw_flow2", "raw_flow3",
                     "proc_flow1", "proc_flow2", "proc_flow3", "press_kPa",
                     "opti_x_mm", "opti_y_mm", "opti_z_mm"])
        last_t = [None]

        def _log_new_sample(i, p):
            if buffers["t"] and buffers["t"][-1] != last_t[0]:
                last_t[0] = buffers["t"][-1]
                press = buffers["press"][-1] if buffers["press"] else float("nan")
                ox = oy = oz = float("nan")
                if opti is not None:
                    sp = opti.get_latest()
                    if sp is not None:
                        mm = opti.opti_to_manip(
                            np.array(sp.pos_xyz, dtype=float),
                            opti_origin if opti_origin is not None else np.zeros(3))
                        ox, oy, oz = float(mm[0]), float(mm[1]), float(mm[2])
                wo.writerow([datetime.now().strftime("%H:%M:%S.%f"), i, *p,
                             f"{last_t[0]:.4f}",
                             f"{buffers['r1'][-1]:.4f}", f"{buffers['r2'][-1]:.4f}",
                             f"{buffers['r3'][-1]:.4f}",
                             f"{buffers['p1'][-1]:.4f}", f"{buffers['p2'][-1]:.4f}",
                             f"{buffers['p3'][-1]:.4f}", f"{press:.5f}",
                             f"{ox:.3f}", f"{oy:.3f}", f"{oz:.3f}"])

        for i, p in enumerate(rows):
            done_event.clear()
            ser.write(f"{p[0]} {p[1]} {p[2]}\n".encode("ascii"))
            print(f"[csv] {i + 1}/{len(rows)} Sent: {p[0]} {p[1]} {p[2]}")

            # Phase 1: wait for the ramp to actually reach this set's target.
            wait_deadline = time.perf_counter() + max_wait_s
            while not done_event.is_set() and time.perf_counter() < wait_deadline:
                _log_new_sample(i, p)
                time.sleep(poll_s)
            if done_event.is_set():
                print(f"  reached target")
            else:
                print(f"  ⚠️  no DONE within {max_wait_s}s -- continuing anyway")

            # Phase 2: extra settle dwell at steady state.
            settle_deadline = time.perf_counter() + settle_s
            while time.perf_counter() < settle_deadline:
                _log_new_sample(i, p)
                time.sleep(poll_s)
            fout.flush()
    ser.write(b"0 0 0\n")
    ser.write(b"r\n")  # stop flow sensor logging (arduino_controller.ino)
    print(f"[csv] Replay done, sent 0 0 0. Saved: {output_path}")


def run_live_plot(buffers, stop_flag, save_fig, name, file_type="eps",
                  record_path=None, record_fps=15.0, record_fig="proc",
                  plot_pressure=False, seal_threshold_kpa=None):
    """
    Live plot — runs on the main thread using FuncAnimation.
    Updates the figures at ~10 Hz independently of the serial read rate.

    record_path : if set, record the chosen figure to this path (.mp4 or .gif)
    record_fps  : target recording frame rate
    record_fig  : "raw" to record fig1, "proc" to record fig2 (default),
                  "press" to record the pressure figure (turns it on)
    plot_pressure      : also show a third figure with the vacuum pressure (kPa)
    seal_threshold_kpa : if set, drawn on the pressure figure as a dashed line
    """
    plot_pressure = plot_pressure or record_fig == "press"
    fig1, ax_raw = plt.subplots(figsize=(9, 4))
    fig1.canvas.manager.set_window_title("Raw Flow  (press Q to stop)")
    lr1, = ax_raw.plot([], [], label="Module 1 (A0)")
    lr2, = ax_raw.plot([], [], label="Module 2 (A1)")
    lr3, = ax_raw.plot([], [], label="Module 3 (A2)")
    ax_raw.set_ylabel("Flow [L/min]")
    ax_raw.set_xlabel("Time [s]")
    ax_raw.set_title("Raw Flow — Module 1 / 2 / 3")
    ax_raw.legend()

    fig2, ax_proc = plt.subplots(figsize=(9, 4))
    fig2.canvas.manager.set_window_title("Processed Flow  (press Q to stop)")
    lp1, = ax_proc.plot([], [], label="Module 1 (A0)")
    lp2, = ax_proc.plot([], [], label="Module 2 (A1)")
    lp3, = ax_proc.plot([], [], label="Module 3 (A2)")
    ax_proc.set_ylabel("Flow [L/min]")
    ax_proc.set_xlabel("Time [s]")
    ax_proc.set_title("Processed Flow — Module 1 / 2 / 3")
    ax_proc.legend()

    fig3 = ax_press = lpr = None
    if plot_pressure:
        fig3, ax_press = plt.subplots(figsize=(9, 4))
        fig3.canvas.manager.set_window_title("Vacuum Pressure  (press Q to stop)")
        lpr, = ax_press.plot([], [], color="tab:purple", label="Pressure (A3)")
        if seal_threshold_kpa is not None:
            ax_press.axhline(seal_threshold_kpa, color="tab:red", linestyle="--",
                             label=f"Seal threshold ({seal_threshold_kpa:g} kPa)")
        ax_press.axhline(0.0, color="gray", linewidth=0.8)
        ax_press.set_ylabel("Pressure [kPa]")
        ax_press.set_xlabel("Time [s]")
        ax_press.set_title("Vacuum Pressure (negative = vacuum)")
        ax_press.legend(loc="lower left")

    # ── Video recorder ────────────────────────────────────────────────────────
    recorder = None
    if record_path is not None and VideoRecorder is not None:
        rec_fig = {"raw": fig1, "press": fig3}.get(record_fig, fig2)
        recorder = VideoRecorder(record_path, fps=record_fps, fig=rec_fig)
    elif record_path is not None:
        print("[video] VideoRecorder not available — recording skipped.")

    def _on_key(event):
        if event.key == "q":
            print("\n[plot] Q pressed — stopping.")
            stop_flag["stop"] = True
            plt.close("all")   # closes all windows → plt.show() returns immediately

    fig1.canvas.mpl_connect("key_press_event", _on_key)
    fig2.canvas.mpl_connect("key_press_event", _on_key)
    if fig3 is not None:
        fig3.canvas.mpl_connect("key_press_event", _on_key)

    def _update(_):
        t = buffers["t"]
        if len(t) < 2:
            return lr1, lr2, lr3, lp1, lp2, lp3

        r1, r2, r3 = buffers["r1"], buffers["r2"], buffers["r3"]
        p1, p2, p3 = buffers["p1"], buffers["p2"], buffers["p3"]

        lr1.set_data(t, r1); lr2.set_data(t, r2); lr3.set_data(t, r3)
        ax_raw.set_xlim(t[0], max(t[-1], t[0] + 1))
        ymin = min(min(r1), min(r2), min(r3))
        ymax = max(max(r1), max(r2), max(r3))
        ax_raw.set_ylim(ymin - 5, ymax + 5)

        lp1.set_data(t, p1); lp2.set_data(t, p2); lp3.set_data(t, p3)
        ax_proc.set_xlim(t[0], max(t[-1], t[0] + 1))
        ymin2 = min(2, min(p1), min(p2), min(p3))
        ymax2 = max(max(p1), max(p2), max(p3))
        ax_proc.set_ylim(ymin2 - 0.1, ymax2 + 0.5)

        fig2.canvas.draw_idle()

        if fig3 is not None:
            pr = buffers["press"][:len(t)]
            lpr.set_data(t[:len(pr)], pr)
            ax_press.set_xlim(t[0], max(t[-1], t[0] + 1))
            lo = min(min(pr), 0.0, seal_threshold_kpa if seal_threshold_kpa is not None else 0.0)
            hi = max(max(pr), 0.0)
            ax_press.set_ylim(lo - 2.0, hi + 2.0)
            fig3.canvas.draw_idle()

        if recorder is not None:
            recorder.capture()

        if stop_flag["stop"]:
            plt.close("all")   # terminal "q" path: close windows from the animation tick
            return lr1, lr2, lr3, lp1, lp2, lp3

        return lr1, lr2, lr3, lp1, lp2, lp3

    # interval=100 ms → 10 Hz redraws, completely independent of serial read rate
    _ani = animation.FuncAnimation(fig1, _update, interval=100, blit=False, cache_frame_data=False)  # noqa: F841 — must be held to prevent GC

    plt.show()   # blocks until all windows are closed

    if recorder is not None:
        recorder.close()

    if save_fig:
        fig1.savefig(f"{name}_raw_flow.{file_type}",  dpi=300, bbox_inches="tight")
        fig2.savefig(f"{name}_proc_flow.{file_type}", dpi=300, bbox_inches="tight")
        print(f"Figures saved: {name}_raw_flow.{file_type}, {name}_proc_flow.{file_type}")
        if fig3 is not None:
            fig3.savefig(f"{name}_pressure.{file_type}", dpi=300, bbox_inches="tight")
            print(f"Figure saved: {name}_pressure.{file_type}")
    

def main():
    import random
    import argparse
    import platform
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, help='Choose object type',required=False)
    parser.add_argument('--module_no', '-n', type=int, help='Choose object type',default=1,required=False)
    parser.add_argument('--record', action='store_true',
                        help='Record the live plot to a video file alongside the CSV')
    parser.add_argument('--record_fps', type=float, default=15.0,
                        help='Recording frame rate for both figure and camera (default 15)')
    parser.add_argument('--plot_pressure', action='store_true',
                        help='Also show a live vacuum-pressure figure (kPa, from the Arduino pressure sensor)')
    parser.add_argument('--seal_threshold_kpa', type=float, default=7.0,
                        help='Draw this seal threshold (kPa, negative) as a dashed line on the pressure figure')
    parser.add_argument('--record_fig', choices=['raw', 'proc', 'press'], default='proc',
                        help='Which figure to record: raw flow, processed flow (default proc), '
                             'or pressure (press -- turns on --plot_pressure)')
    parser.add_argument('--camera', action='store_true',
                        help='Record USB camera video alongside the CSV (requires opencv-python)')
    parser.add_argument('--camera_id', type=int, default=1,
                        help='USB camera device index (default 0)')
    # ── elongation command options ──────────────────────────────────────────────
    parser.add_argument('--settling_s', type=float, default=2.0,
                        help='Seconds to wait after each PWM step before logging (elongation, default 2.0)')
    parser.add_argument('--equil_s', type=float, default=0.1,
                        help='Seconds to log at each steady-state PWM level (elongation, default 1.5)')
    parser.add_argument('--sweep_down', action='store_true',
                        help='Also sweep PWM back down from 25→1 after the up-sweep (elongation)')
    # ── OptiTrack options (used by elongation command) ──────────────────────────
    parser.add_argument('--opti_ip',    default='150.65.146.84')
    parser.add_argument('--local_ip',   default='150.65.146.84')
    parser.add_argument('--opti_id',    type=int, default=1)
    parser.add_argument('--opti_alpha', type=float, default=-178.30969800115128,
                        help='Fallback OptiTrack-to-manipulator rotation angle in degrees, used only '
                             'if MotiveNatNetReader.start()\'s automatic marker-based calibration fails '
                             '(e.g. rig/markers not in view yet) -- normally irrelevant since alpha is '
                             're-derived fresh from the 6 markers every time.')
    parser.add_argument('--no_optitrack', action='store_true',
                        help='Disable OptiTrack for elongation command (opti columns will be NaN)')
    parser.add_argument('--file_type', type=str, default='eps',
                        help='File type for saving figures (default eps)')
    # ── csv replay options ──────────────────────────────────────────────────────
    parser.add_argument('--replay_csv', type=str, default='data/TROpaper_data/perception/testrun_4_perception.csv',
                        help="PWM sets (pwm1,pwm2,pwm3 columns) replayed by typing 'csv' "
                             "(relative paths resolve against the program folder)")
    parser.add_argument('--replay_delay', type=float, default=2.0,
                        help="Extra settle time (s) after Arduino confirms each PWM set has "
                             "reached target, in 'csv' replay (default 1.0)")
    parser.add_argument('--replay_max_wait', type=float, default=5.0,
                        help="Safety timeout (s) to wait for Arduino's DONE feedback per PWM "
                             "set in 'csv' replay, in case it's missed (default 3.0)")

    args = parser.parse_args()
    mode = args.mode
    module_no = args.module_no
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os_name = platform.system().lower()
    if "linux" in os_name:
        serial_port = "/dev/ttyACM0"
    elif "windows" in os_name:
        serial_port = "COM9"
    ser = serial.Serial(serial_port, BAUD_RATE, timeout=1)
    time.sleep(1)
    print(f"[serial] Opened {serial_port}")

    folder_path = f"data/{mode}"
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"log_{mode}_{ts}"
    OUTPUT_CSV = f"data/{mode}/{file_name}.csv"
    f = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f)

    buffers = {"t": [], "r1": [], "r2": [], "r3": [], "p1": [], "p2": [], "p3": [], "press": []}
    stop_flag = {"stop": False}
    save_fig = True
    done_event = threading.Event()  # set by reader_logger on "DONE" -- see replay_pwm_csv

    opti = None
    opti_origin = np.zeros(3)

    reader_thread = threading.Thread(target=reader_logger,
                                     args=(ser, writer, buffers, stop_flag, done_event),
                                     daemon=True)
    reader_thread.start()

    def _input_loop():
        nonlocal opti
        try:
            while not stop_flag["stop"]:
                text = input(">> ").strip()
                if text.lower() in ("q", "quit", "exit"):
                    cmd = "0 0 0\n"
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                    stop_flag["stop"] = True
                    break
                if not text:
                    continue
                cmd = text + "\n"
                ser.write(cmd.encode("ascii"))
                print("[PYTHON] Sent:", text)
                if text.lower() == "p":
                    valve1_pwm = random.randint(5, 25)
                    valve2_pwm = random.randint(5, 25)
                    valve3_pwm = random.randint(5, 25)
                    cmd = f"{valve1_pwm} {valve2_pwm} {valve3_pwm}\n"
                    ser.write(cmd.encode("ascii"))
                    print("[PYTHON] Sent:", cmd.strip())
                elif text.lower() in ("single", "s"):
                    for ite in range(2):
                        TIME_STAMP = f"data/{mode}/{file_name}_timestamp_{ite}.csv"
                        f_timestamp = open(TIME_STAMP, "w", newline="")
                        writer_timestamp = csv.writer(f_timestamp)
                        writer_timestamp.writerow(["current_time", "pwm_incr"])
                        for i in range(1, 26):
                            current_time = datetime.now().strftime("%H:%M:%S")
                            ser.write(f"{i} 0 0\n".encode("ascii"))
                            print("[PYTHON] Sent:", f"{i} 0 0")
                            time.sleep(1.5)
                            writer_timestamp.writerow([current_time, i])
                        ser.write(b"0 0 0\n")
                        time.sleep(4)
                        f_timestamp.close()
                    stop_flag["stop"] = True
                    break
                elif text.lower() in ("double", "d"):
                    fn2 = f"log_module{module_no}vs3_{mode}_{ts}"
                    TIME_STAMP = f"data/{mode}/{fn2}_timestamp_0.csv"
                    f_timestamp = open(TIME_STAMP, "w", newline="")
                    writer_timestamp = csv.writer(f_timestamp)
                    writer_timestamp.writerow(["current_time", "pwm_incr_module1", "pwm_incr_module2"])
                    for module_2 in range(1, 26):
                        for module_1 in range(1, 26):
                            current_time = datetime.now().strftime("%H:%M:%S")
                            ser.write(f"{module_1} 0 {module_2}\n".encode("ascii"))
                            print("[PYTHON] Sent:", f"{module_1} 0 {module_2}")
                            time.sleep(1.5)
                            writer_timestamp.writerow([current_time, module_1, module_2])
                        ser.write(b"0 0 0\n")
                        time.sleep(4)
                    f_timestamp.close()
                    stop_flag["stop"] = True
                    break
                elif text.lower() in ("elongation", "elong"):
                    # ── Pure-elongation calibration sweep ───────────────────
                    # Sends equal PWM to all 3 actuators (1→25) for kinematic
                    # model recalibration: pwm2flow, flow2press, stiffness.

                    # Lazy OptiTrack init (skipped for all other commands)
                    if opti is None and not args.no_optitrack and MotiveNatNetReader is not None:
                        print(f"[optitrack] Connecting to {args.opti_ip} ...")
                        opti = MotiveNatNetReader(
                            server_ip=args.opti_ip,
                            local_ip=args.local_ip,
                            use_multicast=False,
                            rigid_body_id=args.opti_id,
                            alpha=np.deg2rad(args.opti_alpha),  # fallback only -- start() auto-calibrates below
                            local_marker_positions_m=FLOWBOT_MARKER_POSITIONS_M,
                        )
                        opti.start()  # auto-calibrates alpha from the 6 markers, no manual paste needed
                        time.sleep(1.0)
                        sp = opti.get_latest()
                        if sp is not None:
                            opti_origin[:] = np.array(sp.pos_xyz, dtype=float)
                            print(f"[optitrack] Rest origin: {opti_origin} m")
                        else:
                            print("[optitrack] WARNING: no sample — origin stays [0,0,0].")

                    # pwm_levels = list(range(3, 26))
                    pwm_levels = list(range(4, 22, 2))
                    if args.sweep_down:
                        pwm_levels += list(range(24, 0, -1))

                    calib_path = f"data/{mode}/{file_name}_elongation.csv"
                    fc = open(calib_path, "w", newline="")
                    cw = csv.writer(fc)
                    cw.writerow(["t_s", "pwm_cmd",
                                 "proc_flow1", "proc_flow2", "proc_flow3",
                                 "proc_press",
                                 "opti_x_mm", "opti_y_mm", "opti_z_mm"])

                    t0_elong = time.perf_counter()

                    def _log_elong_rows(pwm_val, duration):
                        deadline = time.perf_counter() + duration
                        n = 0
                        while time.perf_counter() < deadline:
                            loop_t = time.perf_counter()
                            if buffers["p1"]:
                                t_now = time.perf_counter() - t0_elong
                                pf1   = buffers["p1"][-1]
                                pf2   = buffers["p2"][-1]
                                pf3   = buffers["p3"][-1]
                                press = buffers["press"][-1] if buffers["press"] else float("nan")
                                ox = oy = oz = float("nan")
                                if opti is not None:
                                    sp = opti.get_latest()
                                    if sp is not None:
                                        mm = opti.opti_to_manip(
                                            np.array(sp.pos_xyz, dtype=float), opti_origin)
                                        ox, oy, oz = float(mm[0]), float(mm[1]), float(mm[2])
                                cw.writerow([f"{t_now:.4f}", pwm_val,
                                             f"{pf1:.4f}", f"{pf2:.4f}", f"{pf3:.4f}",
                                             f"{press:.5f}",
                                             f"{ox:.3f}", f"{oy:.3f}", f"{oz:.3f}"])
                                n += 1
                            rem = 0.05 - (time.perf_counter() - loop_t)
                            if rem > 0:
                                time.sleep(rem)
                        fc.flush()
                        return n

                    print(f"\n[elongation] Output: {calib_path}")
                    print(f"[elongation] Sweep 1→25, settling={args.settling_s}s, log={args.equil_s}s")

                    # Rest state baseline
                    print("[elongation] PWM=0 (rest)")
                    n = _log_elong_rows(0, args.equil_s)
                    print(f"  → {n} rows")
                    ser.write(f"{13} {13} {13}\n".encode("ascii"))
                    input("Press Enter to go next...")
                    ser.write(f"{3} {3} {3}\n".encode("ascii"))
                    input("Press Enter to start the elongation sweep (or Ctrl+C to cancel)...")
                    for pwm_val in pwm_levels:
                        print(f"[elongation] PWM={pwm_val}")
                        ser.write(f"{pwm_val} {pwm_val} {pwm_val}\n".encode("ascii"))
                        time.sleep(args.settling_s)
                        n = _log_elong_rows(pwm_val, args.equil_s)
                        fl = f"{buffers['p1'][-1]:.3f}" if buffers["p1"] else "n/a"
                        pr = f"{buffers['press'][-1]:.5f}" if buffers["press"] else "n/a"
                        print(f"  → {n} rows  flow1={fl} L/min  press={pr} kPa")

                    # Return to rest
                    ser.write(b"0 0 0\n")
                    print("[elongation] PWM=0 (return to rest)")
                    time.sleep(args.settling_s)
                    n = _log_elong_rows(0, args.equil_s)
                    print(f"  → {n} rows")

                    fc.flush()
                    fc.close()
                    print(f"[elongation] Done. Saved: {calib_path}")
                    stop_flag["stop"] = True
                    break

                elif text.lower() == "csv":
                    if opti is None and not args.no_optitrack and MotiveNatNetReader is not None:
                        print(f"[optitrack] Connecting to {args.opti_ip} ...")
                        opti = MotiveNatNetReader(
                            server_ip=args.opti_ip,
                            local_ip=args.local_ip,
                            use_multicast=False,
                            rigid_body_id=args.opti_id,
                            alpha=np.deg2rad(args.opti_alpha),  # fallback only -- start() auto-calibrates below
                            local_marker_positions_m=FLOWBOT_MARKER_POSITIONS_M,
                        )
                        opti.start()  # auto-calibrates alpha from the 6 markers, no manual paste needed
                        time.sleep(1.0)
                        sp = opti.get_latest()
                        if sp is not None:
                            opti_origin[:] = np.array(sp.pos_xyz, dtype=float)
                            print(f"[optitrack] Rest origin: {opti_origin} m")
                        else:
                            print("[optitrack] WARNING: no sample — origin stays [0,0,0].")
                    csv_path = args.replay_csv
                    if not os.path.isabs(csv_path):
                        csv_path = os.path.join(PROJECT_DIR, csv_path)
                    if not os.path.exists(csv_path):
                        print(f"[csv] File not found: {csv_path}")
                        continue
                    ser.write(f"{8} {8} {8}\n".encode("ascii"))
                    ser.write(b"sf\n")
                    input("Press Enter to go next...")
                    replay_pwm_csv(
                        ser, csv_path, args.replay_delay,
                        f"data/TROpaper_data/perception/csv_replay_{ts}.csv",
                        buffers, done_event, max_wait_s=args.replay_max_wait,
                        opti=opti, opti_origin=opti_origin,
                    )
        except KeyboardInterrupt:
            stop_flag["stop"] = True

    input_thread = threading.Thread(target=_input_loop, daemon=True)
    input_thread.start()

    # Start camera before plot so both streams begin at the same time
    camera_rec = None
    if args.camera:
        if not _CV2_AVAILABLE:
            print("[camera] opencv-python not installed — camera recording skipped.")
        else:
            cam_path = f"data/{mode}/ext_cam.mp4"
            camera_rec = CameraRecorder(cam_path, camera_id=args.camera_id,
                                        fps=args.record_fps, stop_flag=stop_flag)
            camera_rec.start()

    # Main thread owns the plot (required by matplotlib)
    rec_path = None
    plot_file_name = f"{args.record_fig}_{ts}"
    if args.record:
        rec_path = f"data/{mode}/{plot_file_name}.mp4"
    run_live_plot(buffers, stop_flag, save_fig, plot_file_name,
                  record_path=rec_path, record_fps=args.record_fps,
                  record_fig=args.record_fig, file_type=args.file_type,
                  plot_pressure=args.plot_pressure, seal_threshold_kpa=args.seal_threshold_kpa)

    if camera_rec is not None:
        camera_rec.stop()

    input_thread.join(timeout=2.0)
    reader_thread.join(timeout=2.0)
    f.flush()
    f.close()
    ser.write(b"0 0 0\n")
    ser.write(b"r\n")
    ser.close()
    if opti is not None:
        opti.stop()
    print("Closed serial, data saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
