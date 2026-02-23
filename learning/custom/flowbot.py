

from __future__ import annotations

import time
import threading
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

import os
import sys
FILE_DIR = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)
import serial
from typing import Callable, Optional, Sequence

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
def clamp_pwm(pwm: np.ndarray) -> np.ndarray:
    pwm = np.asarray(pwm, dtype=int).reshape(3,)
    return pwm.astype(np.int32)

class flowbot:
    from flowbot.kinematic_modeling import Flow_driven_bellow
    from flowbot.workspace import workspace_using_fwdmodel
    from flowbot.online_optitrack import MotiveNatNetReader
    from flowbot.plot_helper import plot_helper
    def __init__(self,serial_port: str = "/dev/ttyACM0",
                 baud: int = 115200,
                 pwm_min: int = 1,
                 pwm_max: int = 25,
                 enable_plot: bool = False,
                frequency: float = 30.0,
                max_pos_speed: float = 150.0,
                initial_pwm: Optional[Sequence[float]] = None,):
        # # --- Serial ---
        self.serial_port = serial_port
        self.baud = baud
        self.ser = serial.Serial(self.serial_port, self.baud, timeout=0.1, write_timeout=0.1)
        
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        time.sleep(1.0)
        print("Opened serial:", self.serial_port, self.baud)
        self.stop_flag = {"stop": False}
        t_reader = threading.Thread(target=drain_serial, args=(self.ser, self.stop_flag), daemon=True)
        t_reader.start()
        self.pwm_min = pwm_min
        self.pwm_max = pwm_max
        self.frequency = float(frequency)
        self.dt = 1.0 / self.frequency
        self.max_pos_speed = float(max_pos_speed)
        self._running = False
        self._lock = threading.Lock()
        # Init robot
        self.flowbot = self.flowbot_init()
        
        # Load workspace
        self.ws, self.tri, self.bbox = self.load_workspace(self.flowbot)
        
        if initial_pwm is None:
            # Start pc at workspace "origin": pwm = [5,5,5]
            pwm = np.array([self.pwm_min, self.pwm_min, self.pwm_min], dtype=int)  # [5,5,5] if pwm_min=5
            pb0 = self.flowbot.pwm_to_pressure(pwm)
            fk0 = self.flowbot.forward_kinematics_from_pressures(pb0)
            self.pc =  np.asarray(fk0["pc"], dtype=float).reshape(3,)
            print("Init pwm:", pwm, "Init pc:", self.pc)
            self.serial_sending(pwm)
            self.last_pwm = pwm
        else:
            pwm = np.asarray(initial_pwm, dtype=float).copy()
            pb0 = self.flowbot.pwm_to_pressure(pwm)
            fk0 = self.flowbot.forward_kinematics_from_pressures(pb0)
            self.pc =  np.asarray(fk0["pc"], dtype=float).reshape(3,)
            print("Init pwm:", pwm, "Init pc:", self.pc)
            self.serial_sending(pwm)
            self.last_pwm = pwm
        if enable_plot:
            plt.ion()
            self.pl = self.plot_helper()
            # self.fig, self.ax, self.pc_handle, self.opti_handle, self.opti_trail = self.pl.setup_plot(self.ws.P)
            self.fig, self.axes, self.pc_handles, self.opti_handles, self.trail_handles = self.pl.setup_plot(self.ws.P)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.show(block=False)
            self.pl.update_point_handle(self.pc_handles, self.pc)
        else:
            self.fig = None
            self.pc_handle = None
            self.opti_handle = None
            self.opti_trail = None

    def flowbot_init(self):
        # --- Robot model (same params as your existing scripts) ---
        robot = self.Flow_driven_bellow(
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
        return robot
    
    def serial_sending(self,pwm):
        cmd0 = f"{int(pwm[0])} {int(pwm[1])} {int(pwm[2])}\n"
        try:
            self.ser.write(cmd0.encode("ascii"))
            print("[PYTHON] Sent:", cmd0.strip())
        except Exception as e:
            print("Serial write failed (init):", e)
    # -------------------------
    # Workspace 
    # -------------------------
    def load_workspace(self,robot: Flow_driven_bellow):
        ws = self.workspace_using_fwdmodel(robot=robot, pwm_min=self.pwm_min, pwm_max=self.pwm_max)
        hull = ws.build_workspace_hull_checker(ws.P)
        tri = hull["tri"]
        bbox = hull["bbox"]
        return ws, tri, bbox
    
    def apply_workspace_constraint(self,
        pc: np.ndarray,
        pc_proposed: np.ndarray,
        policy: str = "backtrack",
    ) -> np.ndarray:
        if self.ws.is_inside_workspace(pc_proposed, self.tri):
            return pc_proposed

        if policy == "hold":
            return pc

        if policy == "backtrack":
            d = pc_proposed - pc
            norm = float(np.linalg.norm(d))
            if norm < 1e-12:
                return pc
            direction = d / norm

            # Try stepping from proposed back toward current until inside
            steps = 30
            for a in np.linspace(1.0, 0.0, steps):
                cand = pc + a * d
                if self.ws.is_inside_workspace(cand,self.tri):
                    return cand
            return pc

        raise ValueError(f"Unknown outside policy: {policy}")

    def start(self) -> None:
        self._running = True


    def _solve_pwm(self, p_task: np.ndarray) -> np.ndarray:
        if self.kin is not None:
            if not hasattr(self.kin, "solve"):
                raise AttributeError("kin must have method solve(p_task)->pwm")
            pwm = self.kin.solve(p_task)  # type: ignore[attr-defined]
        else:
            pwm = self.ik_fn(p_task)  # type: ignore[misc]
        return np.asarray(pwm, dtype=float)
    
    def stop(self) -> None:
        # Stop the robot
        try:
            cmd = "0 0 0\n"
            self.ser.write(cmd.encode("ascii"))
            print("[PYTHON] Sent:", cmd.strip())
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

        print("Closed serial. Exited.")
        self._running = False
        
        if self.pl is not None:
            try:
                self.pl.close()
            except Exception:
                pass
        self.stop_flag["stop"] = True
        try:
            self.ser.close()
        except Exception:
            pass
    def step(self,dpc) -> np.ndarray:
        if not self._running:
            raise RuntimeError("Call start() before step().")

        t0 = time.time()
        dpc = dpc[:3] * self.max_pos_speed * self.dt

        with self._lock:
            self.pc  = self.apply_workspace_constraint(self.pc,self.pc + dpc,"backtrack")
            # print(self.pc)

        try:
            ik = self.flowbot.inverse_pressures_from_position(self.pc)
            pwm = np.asarray(ik["pwm"], dtype=int).reshape(3,)
        except Exception as e:
            # If IK fails (numerical), do not send junk
            print("IK failed:", e)
            pwm = np.array([0, 0, 0], dtype=np.int32)

        if self.last_pwm is None or not np.array_equal(pwm, self.last_pwm):
            cmd = f"{int(pwm[0])} {int(pwm[1])} {int(pwm[2])}\n"
            try:
                self.ser.write(cmd.encode("ascii"))
                # print("[PYTHON] Sent:", cmd.strip(), "pc:", self.pc)
            except Exception as e:
                print("Serial write failed:", e)
            self.last_pwm = pwm

        elapsed = time.time() - t0
        if elapsed < self.dt:
            print(self.dt - elapsed)
            time.sleep(self.dt - elapsed)

        return pwm

    def update_plot(self):
        if self.pl is not None:
            self.pl.update_point_handle(self.pc_handles,self.pc)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            
    def set_position(self, pc: Sequence[float]) -> None:
        p = self.apply_workspace_constraint(self.pc,np.asarray(pc, dtype=float))
        with self._lock:
            self.pc = p
        if self.pl is not None:
            self.pl.update_point_handle(self.pc_handles,p)

    def release(self):
        cmd = "r\n"
        try:
            self.ser.write(cmd.encode("ascii"))
        except Exception as e:
            print("Serial write failed:", e)
        time.sleep(2)


