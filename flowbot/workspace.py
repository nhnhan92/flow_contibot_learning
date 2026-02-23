# workspace_flow_driven_bellow.py
# Build and plot the workspace by sweeping PWM signals, and sample random points inside it.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Tuple, Dict

class workspace_using_fwdmodel:
    def __init__(self,robot,pwm_min = 5, pwm_max = 20):
        self.P, self.U = self.build_workspace_points(robot, pwm_min=pwm_min, pwm_max=pwm_max, step=1)
    def build_workspace_points(self,
        robot,
        pwm_min: int = 5,
        pwm_max: int = 20,
        step: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sweep pwm1,pwm2,pwm3 in [pwm_min, pwm_max] with step, compute pc for each combo.

        Returns:
        P: (N,3) end-effector positions
        U: (N,3) PWM triplets corresponding to each point
        """
        pwms = np.arange(pwm_min, pwm_max + 1, step, dtype=int)

        pts = []
        cmds = []

        for p1 in pwms:
            for p2 in pwms:
                for p3 in pwms:
                    pwm = np.array([p1, p2, p3], dtype=int)
                    pb = robot.pwm_to_pressure(pwm)
                    fk = robot.forward_kinematics_from_pressures(pb)
                    pc = np.asarray(fk["pc"], dtype=float).reshape(3,)
                    if np.all(np.isfinite(pc)):
                        pts.append(pc)
                        cmds.append(pwm)

        P = np.asarray(pts, dtype=float)
        U = np.asarray(cmds, dtype=int)
        return P, U


    def plot_workspace_3d(self,P: np.ndarray, title: str = "Workspace (pc)"):
        """
        3D scatter plot of workspace points.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
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
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, label="workspace points")
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=40, marker="x", label="random inside")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
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

    # 1) Build workspace point cloud
    workspace = workspace_using_fwdmodel(robot=robot,pwm_min=5, pwm_max=20)
    # P, U = workspace.build_workspace_points(robot, pwm_min=5, pwm_max=20, step=1)
    print("Workspace points:", workspace.P.shape[0])

    # 2) Plot workspace
    workspace.plot_workspace_3d(workspace.P, title="Workspace from PWM sweep (5..20)")

    # 3) Build inside-checker (convex hull approx)
    hull = workspace.build_workspace_hull_checker(workspace.P)
    tri = hull["tri"]
    bbox = hull["bbox"]

    # 4) Random sampling inside workspace
    n_samples = 20
    samples = np.vstack([workspace.sample_random_point_in_workspace(tri, bbox) for _ in range(n_samples)])
    print("Random inside samples:\n", samples)

    # 5) Plot workspace + samples
    workspace.plot_workspace_with_random_samples(workspace.P, samples, title="Workspace + random points inside (convex hull)")
