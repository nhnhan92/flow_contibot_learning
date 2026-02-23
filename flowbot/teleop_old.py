# spacemouse_teleop_robot_serial.py
# Combine SpaceMouse position control + workspace constraint + kinematic IK + Arduino serial commands.
#
# Merges ideas from:
#   - spacemouse_control.py (SpaceMouse thread + workspace + live plot)
#   - robot_controller.py   (serial + kinematic IK -> PWM command)
#
# Requirements:
#   pip install pyspacemouse pyserial numpy matplotlib scipy
#
# Usage example:
#   python spacemouse_teleop_robot_serial.py --port COM9 --baud 115200 --device-index 1
#
# Notes:
# - Workspace membership check uses convex-hull (Delaunay) built from sampled FK points.
# - If your true workspace is non-convex, hull is an approximation.
# - This script throttles serial sending to avoid flooding Arduino.

from __future__ import annotations

import time
import threading
import argparse
from typing import Optional

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
# -------------------------
# Control settings
# -------------------------
CONTROL_HZ = 250.0          # integrate pc at this rate
PLOT_HZ = 25.0              # refresh plot at this rate
SEND_HZ = 30.0              # send PWM to Arduino at this rate

POS_GAIN = 15.0             # SpaceMouse -> task velocity gain (workspace units/s at full deflection)
DEADZONE = 0.3        # ignore tiny SpaceMouse noise
MAX_SPEED = 80.0            # clamp max task velocity magnitude (units/s)
MAX_STEP = 5.0              # clamp per-control-step displacement magnitude

OUTSIDE_POLICY = "backtrack"  # "hold" or "backtrack"

# Visual hull complexity (reduce triangles -> faster UI)
HULL_DOWNSAMPLE = 6

# PWM safety clamp
PWM_MIN = 0
PWM_MAX = 255

OPTITRACK_SCALE = 1000.0  # meters -> millimeters
OPTITRACK_ORIGIN_M = np.array([0.026365965604782104, -0.20626311004161835, 0.182607], dtype=float)
OPTITRACK_TRAIL_LEN = 300
# -------------------------
# Serial helpers
# -------------------------
def drain_serial(ser: serial.Serial, stop_flag: dict):
    """Continuously read from serial so input buffer doesn't grow (optional)."""
    while not stop_flag["stop"]:
        try:
            ser.readline()
        except Exception:
            pass


# -------------------------
# SpaceMouse reader thread
# -------------------------
import platform

class _BaseSpaceMouse:
    """Threaded SpaceMouse reader with cached latest xyz."""
    def __init__(self, device_index: int):
        self.device_index = device_index
        self._lock = threading.Lock()
        self._latest_xyz = np.zeros(3, dtype=float)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_latest_xyz(self) -> np.ndarray:
        with self._lock:
            return self._latest_xyz.copy()

    # subclass must implement this
    def _read_xyz_once(self) -> np.ndarray:
        raise NotImplementedError

    def _run(self):
        # choose a polling rate. 200 Hz is usually enough.
        dt = 1.0 / 200.0
        while self._running.is_set():
            try:
                xyz = self._read_xyz_once()
                xyz = np.asarray(xyz, dtype=float).reshape(3,)
                with self._lock:
                    self._latest_xyz[:] = xyz
            except Exception:
                # don’t kill thread on transient read errors
                pass
            # tiny sleep to avoid 100% CPU
            threading.Event().wait(dt)


def _build_spacemouse(device_index: int = 1,os_name: str = 'linux') -> _BaseSpaceMouse:
    os_name = platform.system().lower()

    # Linux: libspnav spacemouse.py
    if "linux" in os_name:
        from learning.custom.spacemouse import SpaceMouse as _SpaceMouseLibspnav

        class _LinuxSpaceMouse(_BaseSpaceMouse):
            def __init__(self, device_index: int):
                super().__init__(device_index)
                self._sm = _SpaceMouseLibspnav(deadzone=DEADZONE, max_value=350.0)

            def _read_xyz_once(self) -> np.ndarray:
                state6 = self._sm.get_motion_state_transformed()  # [x,y,z,rx,ry,rz]
                xyz = np.asarray(state6[:3], dtype=float)
                # axis/sign convention (edit if needed)
                return np.array([xyz[0], xyz[1], xyz[2]], dtype=float)

            def stop(self):
                super().stop()
                try:
                    self._sm.close()
                except Exception:
                    pass

        sm = _LinuxSpaceMouse(device_index)
        sm.start()
        return sm

    # Windows: pyspacemouse spacemouse_control.py
    if "windows" in os_name:
        from spacemouse_control import SpaceMouseReader as _SpaceMouseReaderWin

        class _WindowsSpaceMouse(_BaseSpaceMouse):
            def __init__(self, device_index: int):
                super().__init__(device_index)
                self._reader = _SpaceMouseReaderWin(device_index=device_index)
                self._reader.start()

            def _read_xyz_once(self) -> np.ndarray:
                xyz = np.asarray(self._reader.get_latest_xyz(), dtype=float)
                return np.array([xyz[0], xyz[1], -xyz[2]], dtype=float)

            def stop(self):
                super().stop()
                try:
                    self._reader.stop()
                except Exception:
                    pass

        sm = _WindowsSpaceMouse(device_index)
        sm.start()
        return sm

    raise RuntimeError(f"Unsupported OS: {platform.system()}")



# -------------------------
# Workspace + plotting
# -------------------------
def load_workspace(robot: Flow_driven_bellow, pwm_min: int = 5, pwm_max: int = 20):
    ws = workspace_using_fwdmodel(robot=robot, pwm_min=pwm_min, pwm_max=pwm_max)
    hull = ws.build_workspace_hull_checker(ws.P)
    tri = hull["tri"]
    bbox = hull["bbox"]
    return ws, tri, bbox


def setup_plot(points: np.ndarray):
    P_vis = points[::max(1, int(HULL_DOWNSAMPLE)), :]
    hull = ConvexHull(P_vis)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        P_vis[:, 0], P_vis[:, 1], P_vis[:, 2],
        triangles=hull.simplices,
        alpha=0.25,
        linewidth=0.2,
        edgecolor=(0.2, 0.2, 0.2, 0.35),
        color=(0.15, 0.7, 1.0, 1.0),
    )

    # Teleop kinematics point (task-space pc)
    pc_handle = ax.scatter([P_vis[0, 0]], [P_vis[0, 1]], [P_vis[0, 2]], s=70, c="red", label="Teleop pc")

    # OptiTrack tracked point (origin-referenced, scaled into teleop units)
    opti_handle = ax.scatter([P_vis[0, 0]], [P_vis[0, 1]], [P_vis[0, 2]], s=50, c="blue", label="OptiTrack (origin-ref)")
    (opti_trail,) = ax.plot([], [], [], linewidth=1.0, alpha=0.9, label="OptiTrack trail")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Workspace boundary + Teleop pc (red) + OptiTrack (blue)")
    ax.grid(True)
    ax.legend(loc="best")

    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = 0.5 * (mn + mx)
    span = (mx - mn).max()
    ax.set_xlim(center[0] - 0.5 * span, center[0] + 0.5 * span)
    ax.set_ylim(center[1] - 0.5 * span, center[1] + 0.5 * span)
    ax.set_zlim(center[2] - 0.5 * span, center[2] + 0.5 * span)

    plt.ion()
    plt.tight_layout()
    plt.show()
    return fig, ax, pc_handle, opti_handle, opti_trail


def update_point_handle(pt_handle, pc: np.ndarray):
    pc = np.asarray(pc, dtype=float).reshape(3,)
    pt_handle._offsets3d = ([pc[0]], [pc[1]], [pc[2]])
def update_trail_handle(line_handle, trail_xyz: np.ndarray):
    """Update a 3D line with a Nx3 trail."""
    if trail_xyz is None or len(trail_xyz) == 0:
        line_handle.set_data([], [])
        line_handle.set_3d_properties([])
        return
    trail_xyz = np.asarray(trail_xyz, dtype=float)
    line_handle.set_data(trail_xyz[:, 0], trail_xyz[:, 1])
    line_handle.set_3d_properties(trail_xyz[:, 2])

def apply_workspace_constraint(ws, tri, pc: np.ndarray, pc_proposed: np.ndarray, policy: str = OUTSIDE_POLICY) -> np.ndarray:
    if ws.is_inside_workspace(pc_proposed, tri):
        return pc_proposed

    if policy == "hold":
        return pc

    if policy == "backtrack":
        d = pc_proposed - pc
        if float(np.linalg.norm(d)) < 1e-12:
            return pc
        for a in np.linspace(1.0, 0.0, 40):
            cand = pc + a * d
            if ws.is_inside_workspace(cand, tri):
                return cand
        return pc

    raise ValueError(f"Unknown outside policy: {policy}")


# -------------------------
# Teleop loop
# -------------------------
def clamp_pwm(pwm: np.ndarray) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=int).reshape(3,)
    return pwm.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--device-index", type=int, default=1, help="Index for the spacemouse")
    ap.add_argument("--pwm-min", "-mi", type=int, default=1)
    ap.add_argument("--pwm-max", "-ma", type=int, default=20)
    ap.add_argument("--send-hz", type=float, default=SEND_HZ)
    ap.add_argument("--optitrack", action="store_true",default=False, help="Enable OptiTrack NatNet streaming (Motive/NatNet).")
    args = ap.parse_args()

    # --- Serial ---
    ser = serial.Serial(args.port, args.baud, timeout=0.1, write_timeout=0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    time.sleep(1.0)
    print("Opened serial:", args.port, args.baud)

    stop_flag = {"stop": False}
    t_reader = threading.Thread(target=drain_serial, args=(ser, stop_flag), daemon=True)
    t_reader.start()

    # --- Robot model (same params as your existing scripts) ---
    robot = Flow_driven_bellow(
        D_in=5,
        D_out=16.5,
        l0=82,
        d=28.17,
        lb=0.0,
        lu=13.5,
        k_model=lambda deltal: 0.18417922367667078 + 0.1511268093994831 * (1.0 - np.exp(-0.18801952663756039 * deltal)),
        a_delta=0,
        b_delta=0,
        a_pwm2press=0.004227,
        b_pwm2press=0.012059,
    )

    ws, tri, bbox = load_workspace(robot, pwm_min=args.pwm_min, pwm_max=args.pwm_max)

    # Start pc at workspace "origin": pwm = [5,5,5]
    pwm0 = np.array([args.pwm_min, args.pwm_min, args.pwm_min], dtype=int)  # [5,5,5] if pwm_min=5
    pb0 = robot.pwm_to_pressure(pwm0)
    fk0 = robot.forward_kinematics_from_pressures(pb0)
    pc = np.asarray(fk0["pc"], dtype=float).reshape(3,)
    init_pc = pc
    print("Init pwm:", pwm0, "Init pc:", pc)

    cmd0 = f"{int(pwm0[0])} {int(pwm0[1])} {int(pwm0[2])}\n"
    try:
        ser.write(cmd0.encode("ascii"))
        print("[PYTHON] Sent:", cmd0.strip())
    except Exception as e:
        print("Serial write failed (init):", e)
    

    # --- Plot (optional) ---
    if not args.no_plot:
        fig, ax, pc_handle, opti_handle, opti_trail = setup_plot(ws.P)
        update_point_handle(pc_handle, pc)
    else:
        fig = None
        pc_handle = None
        opti_handle = None
        opti_trail = None
    # Optitrack part
    # --- OptiTrack (optional) ---
    opti = None
    opti_origin_m = OPTITRACK_ORIGIN_M.copy()
    optitrack_init = True
    opti_trail_buf = []
    if args.optitrack:
        opti = MotiveNatNetReader(
        server_ip="192.168.11.15",
        local_ip="192.168.11.15",
        use_multicast=False,
        rigid_body_id=1,
            )
        opti.start()
        
    alpha = -30*np.pi/180
    # --- SpaceMouse thread ---
    os_name = platform.system().lower()

    sm = _build_spacemouse(os_name=os_name)
    sm.start()

    control_dt = 1.0 / CONTROL_HZ
    plot_dt = 1.0 / PLOT_HZ
    send_dt = 1.0 / float(args.send_hz)

    t0 = time.perf_counter()
    t_next_control = t0
    t_next_plot = t0
    t_next_send = t0

    last_pwm = None

    print("Teleop running. Close plot window or Ctrl+C to stop.")
    try:
        while (fig is None) or plt.fignum_exists(fig.number):
            t_now = time.perf_counter()

            # 1) Control integration (fast)
            while t_now >= t_next_control:
                xyz = sm.get_latest_xyz()

                # Optional axis mapping (edit if your axes feel swapped/inverted)
                # xyz = np.array([ xyz[0], xyz[1], xyz[2] ], dtype=float)

                xyz = np.where(np.abs(xyz) < DEADZONE, 0.0, xyz)

                v = POS_GAIN * xyz  # units/s

                # Clamp velocity
                vnorm = float(np.linalg.norm(v))
                if vnorm > MAX_SPEED:
                    v = v * (MAX_SPEED / vnorm)

                dpc = v * control_dt
                step_norm = float(np.linalg.norm(dpc))
                if step_norm > MAX_STEP:
                    dpc = dpc * (MAX_STEP / step_norm)
                pc_proposed = pc + dpc
            
                pc = apply_workspace_constraint(ws, tri, pc, pc_proposed)
                
                t_next_control += control_dt

            # 2) Send command to Arduino (throttled)
            if t_now >= t_next_send:
                try:
                    ik = robot.inverse_pressures_from_position(pc)
                    pwm = clamp_pwm(ik["pwm"])
                    # print(pwm)
                except Exception as e:
                    # If IK fails (numerical), do not send junk
                    print("IK failed:", e)
                    pwm = np.array([0, 0, 0], dtype=np.int32)

                # Send only if changed
                if last_pwm is None or not np.array_equal(pwm, last_pwm):
                    cmd = f"{int(pwm[0])} {int(pwm[1])} {int(pwm[2])}\n"
                    try:
                        ser.write(cmd.encode("ascii"))
                        print("[PYTHON] Sent:", cmd.strip(), "pc:", pc)
                    except Exception as e:
                        print("Serial write failed:", e)
                    last_pwm = pwm

                t_next_send += send_dt

            # 3) Plot refresh (slow)
            if fig is not None and (t_now >= t_next_plot):
                update_point_handle(pc_handle, pc)
                if opti is not None and opti_handle is not None:
                    s = opti.get_latest()
                    if optitrack_init:
                        opti_origin_m = np.array(s.pos_xyz, dtype=float)
                        # print(opti_origin_m)
                        opti_origin_m[1] += (robot.l0+robot.lu)/1000
                        optitrack_init = False
                    if s is not None and not optitrack_init:
                        transformed_rel = opti.opti_to_manip(np.array(s.pos_xyz, dtype=float),opti_origin_m,alpha)
                        # print(transformed_rel)
                        update_point_handle(opti_handle, transformed_rel)
                        opti_trail_buf.append(transformed_rel)
                        # Keep last N points for performance
                        if len(opti_trail_buf) > 50:
                            opti_trail_buf = opti_trail_buf[-50:]
                        if opti_trail is not None:
                            update_trail_handle(opti_trail, np.vstack(opti_trail_buf))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                t_next_plot += plot_dt

            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        sm.stop()
        if opti is not None:
            opti.stop()
        # Stop the robot
        try:
            cmd = "0 0 0\n"
            ser.write(cmd.encode("ascii"))
            print("[PYTHON] Sent:", cmd.strip())
        except Exception:
            pass

        stop_flag["stop"] = True
        try:
            ser.close()
        except Exception:
            pass

        print("Closed serial. Exited.")

if __name__ == "__main__":
    main()
