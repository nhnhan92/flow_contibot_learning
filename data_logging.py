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

try:
    from flowbot.video_recorder import VideoRecorder, CameraRecorder
    _CV2_AVAILABLE = True
except ImportError:
    VideoRecorder    = None
    CameraRecorder   = None
    _CV2_AVAILABLE   = False

BAUD_RATE = 115200


def _default_port():
    """Auto-detect the first available Arduino-like serial port."""
    ports = list(serial.tools.list_ports.comports())
    # Prefer ports with Arduino/CH340/CP210x in the description
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ("arduino", "ch340", "cp210", "usb serial")):
            return p.device
    # Fall back: first available port, or OS-specific guess
    if ports:
        return ports[0].device
    return "COM9" if sys.platform == "win32" else "/dev/ttyACM0"


SERIAL_PORT = _default_port()
OUTPUT_CSV  = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

MAX_POINTS = 1000

def reader_logger(ser, writer, buffers, stop_flag):
    """
    Pure serial reader — runs in a background thread.
    Reads lines from Arduino, writes CSV, fills shared buffers.
    No matplotlib calls here.

    Arduino CSV column order:
      t_ms, rawFlow1, proc_flow1, rawFlow2, proc_flow2,
      rawFlow3, proc_flow3, rawPress, pressMPa, pwm1, pwm2, pwm3
    """
    header_line = ser.readline().decode("utf-8", errors="ignore").strip()
    print("Header:", header_line)
    if not header_line.startswith("t_ms"):
        header_line = ("t_ms,raw_flow1,proc_flow1,raw_flow2,proc_flow2,"
                       "raw_flow3,proc_flow3,raw_press,press_MPa,"
                       "pwm1_cur,pwm2_cur,pwm3_cur")
    writer.writerow(["current_time"] + header_line.split(","))

    t0 = None
    prev_t_ms = None

    while not stop_flag["stop"]:
        raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw_line:
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

        buffers["t"].append(t_s)
        buffers["r1"].append(r1);  buffers["r2"].append(r2);  buffers["r3"].append(r3)
        buffers["p1"].append(pf1); buffers["p2"].append(pf2); buffers["p3"].append(pf3)

        if len(buffers["t"]) > MAX_POINTS:
            for b in buffers.values():
                b[:] = b[-MAX_POINTS:]


def run_live_plot(buffers, stop_flag, save_fig, name,
                  record_path=None, record_fps=15.0, record_fig="proc"):
    """
    Live plot — runs on the main thread using FuncAnimation.
    Updates two figures at ~10 Hz independently of the serial read rate.

    record_path : if set, record the chosen figure to this path (.mp4 or .gif)
    record_fps  : target recording frame rate
    record_fig  : "raw" to record fig1, "proc" to record fig2 (default)
    """
    fig1, ax_raw = plt.subplots(figsize=(9, 4))
    fig1.canvas.manager.set_window_title("Raw Flow  (press Q to stop)")
    lr1, = ax_raw.plot([], [], label="Module 1 (A0)")
    lr2, = ax_raw.plot([], [], label="Module 2 (A1)")
    lr3, = ax_raw.plot([], [], label="Module 3 (A2)")
    ax_raw.set_ylabel("Raw ADC count")
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

    # ── Video recorder ────────────────────────────────────────────────────────
    recorder = None
    if record_path is not None and VideoRecorder is not None:
        rec_fig = fig1 if record_fig == "raw" else fig2
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
        fig1.savefig(f"{name}_raw_flow.png",  dpi=300, bbox_inches="tight")
        fig2.savefig(f"{name}_proc_flow.png", dpi=300, bbox_inches="tight")
        print(f"Figures saved: {name}_raw_flow.png, {name}_proc_flow.png")
    

def main():
    import random
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=str, default=SERIAL_PORT,
                        help='Serial port (e.g. COM3 on Windows, /dev/ttyACM0 on Linux)')
    parser.add_argument('--mode', '-m', type=str, help='Choose object type',required=False)
    parser.add_argument('--module_no', '-n', type=int, help='Choose object type',default=1,required=False)
    parser.add_argument('--record', action='store_true',
                        help='Record the live plot to a video file alongside the CSV')
    parser.add_argument('--record_fps', type=float, default=15.0,
                        help='Recording frame rate for both figure and camera (default 15)')
    parser.add_argument('--record_fig', choices=['raw', 'proc'], default='proc',
                        help='Which figure to record: raw flow or processed flow (default proc)')
    parser.add_argument('--camera', action='store_true',
                        help='Record USB camera video alongside the CSV (requires opencv-python)')
    parser.add_argument('--camera_id', type=int, default=1,
                        help='USB camera device index (default 0)')

    args = parser.parse_args()
    mode = args.mode
    module_no = args.module_no
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[serial] Auto-detected port: {SERIAL_PORT}  (override with --port)")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    time.sleep(1)
    print(f"[serial] Opened {args.port}")

    folder_path = f"data/{mode}"
    os.makedirs(folder_path, exist_ok=True)
    file_name = f"log_{mode}_{ts}"
    OUTPUT_CSV = f"data/{mode}/{file_name}.csv"
    f = open(OUTPUT_CSV, "w", newline="")
    writer = csv.writer(f)

    buffers = {"t": [], "r1": [], "r2": [], "r3": [], "p1": [], "p2": [], "p3": []}
    stop_flag = {"stop": False}
    save_fig = True

    reader_thread = threading.Thread(target=reader_logger,
                                     args=(ser, writer, buffers, stop_flag),
                                     daemon=True)
    reader_thread.start()

    def _input_loop():
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
                elif text.lower() in ("triple", "t"):
                    fn3 = f"log_module{mode}_{ts}"
                    pairs = [(m1, m2, m3)
                             for m1 in range(1, 26)
                             for m2 in range(1, 26)
                             for m3 in range(m2, 26)]
                    count = 20
                    n_full = len(pairs) // count
                    TIME_STAMP = f"data/{mode}/{fn3}_timestamp_0.csv"
                    f_timestamp = open(TIME_STAMP, "w", newline="")
                    writer_timestamp = csv.writer(f_timestamp)
                    writer_timestamp.writerow(["current_time", "pwm_incr_module1",
                                               "pwm_incr_module2", "pwm_incr_module3"])
                    for i in range(n_full):
                        for j in range(count):
                            p = pairs[count * i + j]
                            current_time = datetime.now().strftime("%H:%M:%S")
                            ser.write(f"{p[0]} {p[1]} {p[2]}\n".encode("ascii"))
                            print("[PYTHON] Sent:", f"{p[0]} {p[1]} {p[2]}")
                            time.sleep(1.5)
                            writer_timestamp.writerow([current_time, *p])
                        ser.write(b"0 0 0\n")
                        time.sleep(4)
                    for p in pairs[count * n_full:count * n_full + 25]:
                        current_time = datetime.now().strftime("%H:%M:%S")
                        ser.write(f"{p[0]} {p[1]} {p[2]}\n".encode("ascii"))
                        time.sleep(1.5)
                        writer_timestamp.writerow([current_time, *p])
                    ser.write(b"0 0 0\n")
                    time.sleep(4)
                    f_timestamp.close()
                    stop_flag["stop"] = True
                    break
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
                  record_fig=args.record_fig)

    if camera_rec is not None:
        camera_rec.stop()

    input_thread.join(timeout=2.0)
    reader_thread.join(timeout=2.0)
    f.flush()
    f.close()
    ser.close()
    print("Closed serial, data saved to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
