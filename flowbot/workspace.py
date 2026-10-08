# workspace_flow_driven_bellow.py
# Build and plot the workspace by sweeping PWM signals, and sample random points inside it.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, ConvexHull
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, Dict

class workspace_using_fwdmodel:
    def __init__(self,robot,pwm_min = 5, pwm_max = 20, l_s: float = 100.0):
        self.l_s = l_s
        self.P, self.U, self.T = self.build_workspace_points(robot, pwm_min=pwm_min, pwm_max=pwm_max, step=1)
        self.Ps = self.P + self.l_s * self.T
    def build_workspace_points(self,
        robot,
        pwm_min: int = 5,
        pwm_max: int = 20,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sweep pwm1,pwm2,pwm3 in [pwm_min, pwm_max] with step, compute pc for each combo.

        Returns:
        P: (N,3) end-effector positions
        U: (N,3) PWM triplets corresponding to each point
        T: (N,3) unit tip tangents (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
        """
        pwms = np.arange(pwm_min, pwm_max + 1, step, dtype=int)

        pts = []
        cmds = []
        tans = []

        for p1 in pwms:
            for p2 in pwms:
                for p3 in pwms:
                    pwm = np.array([p1, p2, p3], dtype=int)
                    pb = robot.pwm_to_pressure(pwm)
                    fk = robot.forward_kinematics_from_pressures(pb)
                    pc = np.asarray(fk["pc"], dtype=float).reshape(3,)
                    if np.all(np.isfinite(pc)):
                        theta = float(np.asarray(fk["theta"]).reshape(-1)[0])
                        phi = float(np.asarray(fk["phi"]).reshape(-1)[0])
                        tan = np.array([np.sin(theta) * np.cos(phi),
                                        np.sin(theta) * np.sin(phi),
                                        np.cos(theta)], dtype=float)
                        pts.append(pc)
                        cmds.append(pwm)
                        tans.append(tan)

        P = np.asarray(pts, dtype=float)
        U = np.asarray(cmds, dtype=int)
        T = np.asarray(tans, dtype=float)
        return P, U, T


    def _draw_workspace_volume(self, ax, P: np.ndarray, color="tab:blue", alpha: float = 0.15,
                               autoscale: bool = True):
        """
        Draw the convex hull of P as a semi-transparent volume.
        """
        hull = ConvexHull(P)
        faces = [P[simplex] for simplex in hull.simplices]
        poly = Poly3DCollection(faces, facecolors=color, edgecolors=(0, 0, 0, 0.25),
                                linewidths=0.3, alpha=alpha)
        ax.add_collection3d(poly)
        if autoscale:
            ax.auto_scale_xyz(P[:, 0], P[:, 1], P[:, 2])


    def workspace_boundary_info(self, P: np.ndarray, name: str = "workspace") -> Dict[str, object]:
        """
        Print and return the boundary of a workspace point cloud:
        per-axis min/max (bounding box), convex hull volume and vertex count.
        """
        hull = ConvexHull(P)
        mn = P.min(axis=0)
        mx = P.max(axis=0)
        info = {
            "name": name,
            "n_points": int(P.shape[0]),
            "min_xyz": mn,
            "max_xyz": mx,
            "extent_xyz": mx - mn,
            "hull_volume": float(hull.volume),
            "hull_vertices": int(len(hull.vertices)),
        }
        print(f"[{name}] points: {info['n_points']}")
        print(f"  x: [{mn[0]:.2f}, {mx[0]:.2f}] mm   (extent {mx[0]-mn[0]:.2f} mm)")
        print(f"  y: [{mn[1]:.2f}, {mx[1]:.2f}] mm   (extent {mx[1]-mn[1]:.2f} mm)")
        print(f"  z: [{mn[2]:.2f}, {mx[2]:.2f}] mm   (extent {mx[2]-mn[2]:.2f} mm)")
        print(f"  convex hull volume: {info['hull_volume']:.1f} mm^3, hull vertices: {info['hull_vertices']}")
        return info


    def plot_workspace_pair(self, P: np.ndarray, Ps: np.ndarray, title: str = "Workspace", 
                            plot_ws_suction: bool = False, save_fig: bool = False):
        """
        Overlay the end-point workspace (P) and the suction-cup workspace (Ps)
        in one 3D figure, as two semi-transparent volumes.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._draw_workspace_volume(ax, P, color="tab:blue", alpha=0.15, autoscale=False)
        if plot_ws_suction:
            self._draw_workspace_volume(ax, Ps, color="tab:orange", alpha=0.15, autoscale=False)
            both = np.vstack([P, Ps])
            ax.auto_scale_xyz(both[:, 0], both[:, 1], both[:, 2])
            handles = [Patch(facecolor="tab:blue", alpha=0.4, label="P (flowbot end point)"),
                    Patch(facecolor="tab:orange", alpha=0.4, label=f"P_s (suction cup, l_s={self.l_s:g} mm)")]
        else:
            ax.auto_scale_xyz(P[:, 0], P[:, 1], P[:, 2])
            handles = [Patch(facecolor="tab:blue", alpha=0.4, label="P (flowbot end point)")]
        ax.legend(handles=handles)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.invert_zaxis()
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        if save_fig:
            fig_path = Path(f"workspace.eps")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            print(f"Saved workspace figure to: {fig_path}")


    def plot_workspace_3d(self,P: np.ndarray, title: str = "Workspace (pc)"):
        """
        3D semi-transparent volume of the workspace (convex hull of P).
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._draw_workspace_volume(ax, P)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.invert_zaxis()
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        plt.show()


    def build_workspace_hull_checker(self,P: np.ndarray) -> Dict[str, object]:
        """
        Build a convex hull membership checker using Delaunay triangulation.
        For a point q, inside check is hull.find_simplex(q) >= 0.

        Returns:
        dict with keys:
            - "tri": Delaunay object
            - "bbox": (min_xyz, max_xyz)
        """
        if P.shape[0] < 4:
            raise ValueError("Need at least 4 non-coplanar points for a 3D hull.")

        tri = Delaunay(P)
        mn = P.min(axis=0)
        mx = P.max(axis=0)
        return {"tri": tri, "bbox": (mn, mx)}


    def is_inside_workspace(self,q: np.ndarray, tri: Delaunay) -> bool:
        q = np.asarray(q, dtype=float).reshape(1, 3)
        return tri.find_simplex(q)[0] >= 0


    def sample_random_point_in_workspace(self,
        tri: Delaunay,
        bbox: Tuple[np.ndarray, np.ndarray],
        max_tries: int = 100000,
    ) -> np.ndarray:
        """
        Rejection sample uniformly from the axis-aligned bounding box, accept if inside hull.
        """
        mn, mx = bbox
        for _ in range(max_tries):
            q = mn + (mx - mn) * np.random.rand(3)
            if self.is_inside_workspace(q, tri):
                return q
        raise RuntimeError("Failed to sample a point inside workspace (increase max_tries or check hull).")


    def plot_workspace_with_random_samples(self,P: np.ndarray, samples: np.ndarray, title: str):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        self._draw_workspace_volume(ax, P)
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=40, marker="x", label="random inside")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.invert_zaxis()
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import os, sys
    from pathlib import Path
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(FILE_DIR))
    from kinematic_modeling import Flow_driven_bellow
    from flowbot.kinematic_modeling_linear import Flow_driven_bellow_linear
    from flowbot.pwm2flow import Pwm2FlowModel
    from flowbot.pressure_flow_model import Flow2PressModel, Press2FlowModel
    from learning.hardware import flowbot
    pwm2flow   = Pwm2FlowModel.load(  Path(FILE_DIR) / "pwm2flow.pkl")
    flow2press = Flow2PressModel.load( Path(FILE_DIR) / "flow2press.pkl")
    press2flow = Press2FlowModel.load( Path(FILE_DIR) / "press2flow.pkl")

    _k_model = lambda deltal: 0.18417922367667078 + 0.1511268093994831 * (1.0 - np.exp(-0.18801952663756039 * deltal))
    _common = dict(D_in=5, D_out=16.5, l0=82, d=28.17, lb=0.0, lu=13.5,
                    k_model=_k_model, a_delta=0, b_delta=0)
    robot = Flow_driven_bellow_linear(
                    **_common,
                    a_pwm2press=0.004227,
                    b_pwm2press=0.012059,
                )
    # 1) Build workspace point cloud
    workspace = workspace_using_fwdmodel(robot=robot,pwm_min=1, pwm_max=26, l_s=100.0)
    # Boundary info for each workspace
    workspace.workspace_boundary_info(workspace.P, name="P (flowbot end point)")
    workspace.workspace_boundary_info(workspace.Ps, name=f"P_s (suction cup, l_s={workspace.l_s:g} mm)")

    # 2) Plot workspace: P (flowbot end point) and P_s (suction cup) in one figure
    workspace.plot_workspace_pair(workspace.P, workspace.Ps,
                                  title="Workspace from PWM sweep (1..26): P and P_s",
                                  plot_ws_suction=True,
                                  save_fig = False)

    # 3) Build inside-checker (convex hull approx)
    hull = workspace.build_workspace_hull_checker(workspace.P)
    tri = hull["tri"]
    bbox = hull["bbox"]

    # # 4) Random sampling inside workspace
    # n_samples = 20
    # samples = np.vstack([workspace.sample_random_point_in_workspace(tri, bbox) for _ in range(n_samples)])
    # print("Random inside samples:\n", samples)

    # # 5) Plot workspace + samples
    # workspace.plot_workspace_with_random_samples(workspace.P, samples, title="Workspace + random points inside (convex hull)")
