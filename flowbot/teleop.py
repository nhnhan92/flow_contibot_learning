

from __future__ import annotations

import time
import threading
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import serial
from kinematic_modeling import Flow_driven_bellow
from workspace import workspace_using_fwdmodel
from online_optitrack import MotiveNatNetReader
import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)
from learning.hardware.spacemouse import _build_spacemouse

# -------------------------
# Control settings
# -------------------------
PLOT_HZ = 25.0              # refresh plot at this rate
SEND_HZ = 30.0              # send PWM to Arduino at this rate

POS_GAIN = 1.0             # SpaceMouse -> task velocity gain (workspace units/s at full deflection)
DEADZONE = 0.1      # ignore tiny SpaceMouse noise
MAX_SPEED = 80.0            # clamp max task velocity magnitude (units/s)
MAX_STEP = 8.0              # clamp per-control-step displacement magnitude

OUTSIDE_POLICY = "backtrack"  # "hold" or "backtrack"

# Visual hull complexity (reduce triangles -> faster UI)
HULL_DOWNSAMPLE = 6

# PWM safety clamp
PWM_MIN = 0
PWM_MAX = 255

OPTITRACK_SCALE = 1000.0  # meters -> millimeters
# OPTITRACK_ORIGIN_M = np.array([0.026365965604782104, -0.20626311004161835, 0.182607], dtype=float)
OPTITRACK_ORIGIN_M = np.array([0.891567139, 0.335721802, -0.00522339], dtype=float)
# 891.567139	335.721802	-5.22339
OPTITRACK_TRAIL_LEN = 50    # keep last N optitrack points in trail

def main():
    from learning.hardware import flowbot
    import platform
    import select

    ap = argparse.ArgumentParser()
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--device-index", type=int, default=1, help="Index for the spacemouse")
    ap.add_argument("--pwm-min", "-mi", type=int, default=1)
    ap.add_argument("--pwm-max", "-ma", type=int, default=25)
    ap.add_argument("--send-hz", type=float, default=SEND_HZ)
    ap.add_argument("--plot-history", type=int, default=50)
    ap.add_argument("--optitrack","-opt", action="store_true",default=False, help="Enable OptiTrack NatNet streaming (Motive/NatNet).")
    ap.add_argument("--pressure-model", '-p', choices=["learned", "linear"], default="linear",
                    help="Pressure model: 'learned' (pkl models) or 'linear' (a*pwm+b).")
    args = ap.parse_args()

    os_name = platform.system().lower()
    if "linux" in os_name:
        serial_port = "/dev/ttyACM0"
    elif "windows" in os_name:
        serial_port = "COM9"
    CONTROL_HZ = 30.0          # integrate pc at this rate

    ### Flowbot
    fb = flowbot.flowbot(serial_port = serial_port,
                 pwm_min= 0,
                 pwm_max= 26,
                 enable_plot = True,
                frequency = CONTROL_HZ,
                max_pos_speed = 50,
                pressure_model = args.pressure_model)
    # --- OptiTrack (optional) ---
    opti = None
    opti_origin_m = OPTITRACK_ORIGIN_M.copy()
    optitrack_init = True   # True until first valid reading sets the origin
    opti_trail_buf = []

    if args.optitrack:
        opti = MotiveNatNetReader(
            server_ip="150.65.146.84",
            local_ip="150.65.146.84",
            use_multicast=False,
            rigid_body_id=1,
        )
        opti.start()
    PLOT_HZ = 30.0
    plot_dt = 1.0 / PLOT_HZ
    t_next_plot = time.perf_counter()
    ## --- SpaceMouse flowbot thread ---
    sm_fb = _build_spacemouse(os_name=os_name)
    sm_fb.start()

    fb.start()

    # --- keyboard: 'r' resets pc to init and sends [0,0,0] ---
    def _on_key(event):
        if event.key == 'r':
            fb.reset()
            print("Reset: pc ->", fb.pc, "  pwm -> [0,0,0]")

    if fb.fig is not None:
        fb.fig.canvas.mpl_connect('key_press_event', _on_key)

    print("Teleop running. Press 'r' to reset. Close plot window or Ctrl+C to stop.")
    try:
        while (fb.fig is None) or plt.fignum_exists(fb.fig.number) :
            t_now = time.perf_counter()
            
            xyz = sm_fb.get_latest_xyz()
            button_status = sm_fb.get_button_status()
            xyz[2] = -xyz[2] 
            xyz = np.where(np.abs(xyz) < DEADZONE, 0.0, xyz)
            fb.step(xyz)
            if fb.pl is not None and t_now >= t_next_plot:
                # Update OptiTrack handles first (before canvas flush)
                if opti is not None:
                    s = opti.get_latest()
                    if s is not None:
                        if optitrack_init:
                            opti_origin_m = np.array(s.pos_xyz, dtype=float)
                            opti_origin_m[1] += (fb.flowbot.l0 + fb.flowbot.lu) / 1000.0
                            optitrack_init = False

                        if not optitrack_init:
                            transformed = opti.opti_to_manip(
                                np.array(s.pos_xyz, dtype=float),
                                opti_origin_m,
                            )
                            transformed[0] = -transformed[0]   # flip X
                            transformed[1] = -transformed[1]   # flip Y
                            fb.pl.update_opti_handle(fb.opti_handles, transformed)

                            opti_trail_buf.append(transformed)
                            if len(opti_trail_buf) > OPTITRACK_TRAIL_LEN:
                                opti_trail_buf = opti_trail_buf[-OPTITRACK_TRAIL_LEN:]
                            fb.pl.update_trail_handle(fb.trail_handles, np.vstack(opti_trail_buf))

                # Update pc handle + flush canvas once (catches both pc and opti updates)
                fb.update_plot()

                t_next_plot += plot_dt
    except KeyboardInterrupt:
        pass
    finally:
        sm_fb.stop()
        fb.stop()
        time.sleep(0.5)  # give threads a moment to exit
        if opti is not None:
            opti.stop()
        
        

if __name__ == "__main__":
    main()
