from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import pyspacemouse


# =========================
# User config
# =========================
DEVICE_INDEX = 1

# Control rates
CONTROL_HZ = 250.0     # pc integration frequency
PLOT_HZ = 25.0         # plot refresh frequency

# SpaceMouse scaling (tune)
POS_GAIN = 5.0        # position gain (workspace units per second at full deflection)
DEADZONE = 0.04        # ignore tiny noise
MAX_SPEED = 80.0       # clamp speed magnitude (units/s)

# Workspace boundary behavior
OUTSIDE_POLICY = "backtrack"  # "hold" or "backtrack"

# Hull surface complexity (reduce triangles -> faster UI)
HULL_DOWNSAMPLE = 6     # use every Nth point for hull surface (visual only)

# =========================
# Workspace adapter
# =========================

def load_workspaces(robot):
    """
    You said you revised the workspace script as a class.
    Edit the import and the instantiation below to match your class.

    Requirement:
      Your workspace object should provide either:
        - workspace.points  (N,3), or
        - workspace.P       (N,3), or
        - workspace.get_points() -> (N,3)
    """
    from workspace import workspace_using_fwdmodel 


    ws = workspace_using_fwdmodel(robot=robot,pwm_min=5, pwm_max=20) 
    hull = ws.build_workspace_hull_checker(ws.P)
    tri = hull["tri"]
    bbox = hull["bbox"]

    return ws,tri,bbox


# =========================
# SpaceMouse reading helpers
# =========================
class SpaceMouseReader:
    def __init__(self, device_index: int):
        self.device_index = device_index
        self._lock = threading.Lock()
        self._latest_xyz = np.zeros(3, dtype=float)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _extract_xyz(state) -> np.ndarray:
        if hasattr(state, "x") and hasattr(state, "y") and hasattr(state, "z"):
            return np.array([float(state.x), float(state.y), float(state.z)], dtype=float)
        if isinstance(state, dict) and all(k in state for k in ("x", "y", "z")):
            return np.array([float(state["x"]), float(state["y"]), float(state["z"])], dtype=float)
        raise AttributeError("Cannot read x,y,z from SpaceMouse state.")

    def start(self):
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest_xyz(self) -> np.ndarray:
        with self._lock:
            return self._latest_xyz.copy()

    def _run(self):
        try:
            ctx = pyspacemouse.open(device_index=self.device_index)
        except TypeError:
            # older pyspacemouse
            ctx = pyspacemouse.open()

        with ctx as dev:
            while self._running.is_set():
                st = dev.read()
                xyz = self._extract_xyz(st)

                with self._lock:
                    self._latest_xyz[:] = xyz

# =========================
# Plot helpers
# =========================

class plot_helper:
    def setup_plot(self,points: np.ndarray):
        """
        Draw a convex hull surface (visual boundary) once, then update only the moving point.
        """
        # Downsample for hull drawing (visual only)
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
            color=(0.15, 0.7, 1.0, 1.0),  # easy-to-see light cyan
        )

        # Moving point
        pt = ax.scatter([P_vis[0, 0]], [P_vis[0, 1]], [P_vis[0, 2]], s=70, c="red")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Workspace boundary (surface) + SpaceMouse pc")
        ax.grid(True)

        # Set axis limits
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

        return fig, ax, pt


    def update_point_handle(self,pt_handle, pc: np.ndarray):
        pc = np.asarray(pc, dtype=float).reshape(3,)
        pt_handle._offsets3d = ([pc[0]], [pc[1]], [pc[2]])


    # =========================
    # Main control loop
    # =========================

    def apply_workspace_constraint(self,
        ws,
        tri,
        pc: np.ndarray,
        pc_proposed: np.ndarray,
        policy: str = "backtrack",
    ) -> np.ndarray:
        if ws.is_inside_workspace(pc_proposed, tri):
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
                if ws.is_inside_workspace(cand,tri):
                    return cand
            return pc

        raise ValueError(f"Unknown outside policy: {policy}")


def main():
    from kinematic_modeling import Flow_driven_bellow

    robot = Flow_driven_bellow(
            D_in = 5,
            D_out = 16.5,
            l0=82,
            d=28.17,
            lb=0.0,
            lu=13.5,
            k_model= lambda deltal: 0.18417922367667078 + 0.1511268093994831 * (1.0 - np.exp(-0.18801952663756039 * deltal)),
            a_delta = 0,
            b_delta= 0,
            a_pwm2press= 0.004227,
            b_pwm2press= 0.012059,
        )
    ws,tri,bbox = load_workspaces(robot)

    # Start pc at workspace center (bbox center)
    mn = bbox[0]
    mx = bbox[1]
    pc = 0.5 * (mn + mx)

    # 2) Setup plot
    fig, ax, pt_handle = setup_plot(ws.P)
    update_point_handle(pt_handle, pc)

    # Start SpaceMouse thread
    sm = SpaceMouseReader(device_index=DEVICE_INDEX)
    sm.start()
    # Timers for control and plotting
    control_dt = 1.0 / CONTROL_HZ
    plot_dt = 1.0 / PLOT_HZ

    t_last = time.perf_counter()
    t_next_control = t_last
    t_next_plot = t_last

    try:
        while plt.fignum_exists(fig.number):
            t_now = time.perf_counter()
            # Run control step(s) at CONTROL_HZ
            while t_now >= t_next_control:
                xyz = sm.get_latest_xyz()

                # deadzone
                xyz = np.where(np.abs(xyz) < DEADZONE, 0.0, xyz)

                # Map to velocity (units/s). Use xyz directly as a "command" signal.
                v = POS_GAIN * xyz

                # Clamp speed
                vnorm = float(np.linalg.norm(v))
                if vnorm > MAX_SPEED:
                    v = v * (MAX_SPEED / vnorm)

                pc_proposed = pc + v * control_dt
                pc = apply_workspace_constraint(ws,tri, pc, pc_proposed)

                t_next_control += control_dt

            # Plot refresh at PLOT_HZ
            if t_now >= t_next_plot:
                update_point_handle(pt_handle, pc)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                t_next_plot += plot_dt

            # tiny sleep to reduce CPU
            time.sleep(0.001)

    finally:
        sm.stop()


if __name__ == "__main__":
    main()
