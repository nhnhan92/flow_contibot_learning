# =========================
# Plot helpers
# =========================

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# =========================
# User config
# =========================

# Hull surface complexity (reduce triangles -> faster UI)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

HULL_DOWNSAMPLE = 30  # keep your constant

class plot_helper:
    def setup_plot(self, points: np.ndarray):
        points = np.asarray(points, dtype=float)
        P_vis = points[::max(1, int(HULL_DOWNSAMPLE)), :]

        mn = points.min(axis=0)
        mx = points.max(axis=0)

        plt.ion()
        fig = plt.figure(figsize=(9, 7))

        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[1.5, 1.5, 0.5],   # right column for legend
            height_ratios=[1.2, 1.0],        # XY larger (optional)
            wspace=0.35, hspace=0.35
        )

        ax_xy  = fig.add_subplot(gs[0, 0:2])  # span two columns (centered)
        ax_xz  = fig.add_subplot(gs[1, 0])
        ax_yz  = fig.add_subplot(gs[1, 1])
        ax_leg = fig.add_subplot(gs[:, 2])    # legend area (spans rows)
        ax_leg.axis("off")

        ax_xy.set_title("XY"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
        ax_xz.set_title("XZ"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
        ax_yz.set_title("YZ"); ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")

        for ax in (ax_xy, ax_xz, ax_yz):
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")

        def _plot_hull_2d(ax, pts2):
            pts2 = np.asarray(pts2, dtype=float)
            if len(pts2) < 3:
                ax.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.4)
                return
            try:
                hull = ConvexHull(pts2)
                poly = pts2[hull.vertices]
                poly = np.vstack([poly, poly[0]])
                ax.plot(poly[:, 0], poly[:, 1], linewidth=1.0, alpha=0.6)
            except Exception:
                ax.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.4)

        # Draw projected hulls
        _plot_hull_2d(ax_xy, P_vis[:, [0, 1]])
        _plot_hull_2d(ax_xz, P_vis[:, [0, 2]])
        _plot_hull_2d(ax_yz, P_vis[:, [1, 2]])

        # Set limits
        ax_xy.set_xlim(mn[0], mx[0]); ax_xy.set_ylim(mn[1], mx[1])
        ax_xz.set_xlim(mn[0], mx[0]); ax_xz.set_ylim(mn[2], mx[2])
        ax_yz.set_xlim(mn[1], mx[1]); ax_yz.set_ylim(mn[2], mx[2])
        ax_xz.invert_yaxis()
        ax_yz.invert_yaxis()
        # Initial placeholder point
        p0 = P_vis[0]

        # Handles
        pc_xy = ax_xy.scatter([p0[0]], [p0[1]], s=60, c="red", label="pc")
        pc_xz = ax_xz.scatter([p0[0]], [p0[2]], s=60, c="red", label="pc")
        pc_yz = ax_yz.scatter([p0[1]], [p0[2]], s=60, c="red", label="pc")

        opti_xy = ax_xy.scatter([p0[0]], [p0[1]], s=45, c="blue", label="opti")
        opti_xz = ax_xz.scatter([p0[0]], [p0[2]], s=45, c="blue", label="opti")
        opti_yz = ax_yz.scatter([p0[1]], [p0[2]], s=45, c="blue", label="opti")

        (trail_xy,) = ax_xy.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
        (trail_xz,) = ax_xz.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
        (trail_yz,) = ax_yz.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
        
        handles = [pc_xy, opti_xy, trail_xy]
        labels  = ["pc", "opti", "trail"]

        leg = ax_leg.legend(
            handles, labels,
            loc="center",
            frameon=True,
            fontsize=14,          # bigger text
            markerscale=1.6,      # bigger scatter markers in legend
            handlelength=2.2,     # longer line sample
            handletextpad=0.8,    # space between handle and text
            labelspacing=0.8,     # vertical spacing between entries
            borderpad=1.2,        # padding inside the legend box
            borderaxespad=0.0
        )

        fig.show()
        plt.pause(0.001)

        axes = {"xy": ax_xy, "xz": ax_xz, "yz": ax_yz}
        pc_handles = {"xy": pc_xy, "xz": pc_xz, "yz": pc_yz}
        opti_handles = {"xy": opti_xy, "xz": opti_xz, "yz": opti_yz}
        trail_handles = {"xy": trail_xy, "xz": trail_xz, "yz": trail_yz}

        return fig, axes, pc_handles, opti_handles, trail_handles

    def update_point_handle(self, pc_handles, pc: np.ndarray):
        pc = np.asarray(pc, dtype=float).reshape(3,)
        pc_handles["xy"].set_offsets([[pc[0], pc[1]]])
        pc_handles["xz"].set_offsets([[pc[0], pc[2]]])
        pc_handles["yz"].set_offsets([[pc[1], pc[2]]])

    def update_opti_handle(self, opti_handles, p: np.ndarray):
        p = np.asarray(p, dtype=float).reshape(3,)
        opti_handles["xy"].set_offsets([[p[0], p[1]]])
        opti_handles["xz"].set_offsets([[p[0], p[2]]])
        opti_handles["yz"].set_offsets([[p[1], p[2]]])

    def update_trail_handle(self, trail_handles, trail_xyz: np.ndarray):
        if trail_xyz is None or len(trail_xyz) == 0:
            trail_handles["xy"].set_data([], [])
            trail_handles["xz"].set_data([], [])
            trail_handles["yz"].set_data([], [])
            return

        trail_xyz = np.asarray(trail_xyz, dtype=float)
        trail_handles["xy"].set_data(trail_xyz[:, 0], trail_xyz[:, 1])
        trail_handles["xz"].set_data(trail_xyz[:, 0], trail_xyz[:, 2])
        trail_handles["yz"].set_data(trail_xyz[:, 1], trail_xyz[:, 2])

# class plot_helper:
#     def setup_plot(self, points: np.ndarray):
#         points = np.asarray(points, dtype=float)
#         P_vis = points[::max(1, int(HULL_DOWNSAMPLE)), :]

#         mn = points.min(axis=0)
#         mx = points.max(axis=0)
#         center = 0.5 * (mn + mx)
#         span = float((mx - mn).max())

#         plt.ion()
#         fig = plt.figure(figsize=(12, 10))

#         # 2x2 grid: 3D at top-left, then XY top-right, XZ bottom-left, YZ bottom-right
#         ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
#         ax_xy = fig.add_subplot(2, 2, 2)
#         ax_xz = fig.add_subplot(2, 2, 3)
#         ax_yz = fig.add_subplot(2, 2, 4)

#         # -------------------------
#         # 3D workspace surface (convex hull)
#         # -------------------------
#         hull3d = ConvexHull(P_vis)
#         ax_3d.plot_trisurf(
#             P_vis[:, 0], P_vis[:, 1], P_vis[:, 2],
#             triangles=hull3d.simplices,
#             alpha=0.20,
#             linewidth=0.2,
#             edgecolor=(0.2, 0.2, 0.2, 0.25),
#         )

#         ax_3d.set_title("3D Workspace + pc (red) + Opti (blue)")
#         ax_3d.set_xlabel("x")
#         ax_3d.set_ylabel("y")
#         ax_3d.set_zlabel("z")
#         ax_3d.grid(True)

#         ax_3d.set_xlim(center[0] - 0.5 * span, center[0] + 0.5 * span)
#         ax_3d.set_ylim(center[1] - 0.5 * span, center[1] + 0.5 * span)
#         ax_3d.set_zlim(center[2] - 0.5 * span, center[2] + 0.5 * span)

#         # -------------------------
#         # 2D hulls on each plane
#         # -------------------------
#         def _plot_hull_2d(ax, pts2):
#             pts2 = np.asarray(pts2, dtype=float)
#             if len(pts2) < 3:
#                 ax.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.3)
#                 return
#             try:
#                 h = ConvexHull(pts2)
#                 poly = pts2[h.vertices]
#                 poly = np.vstack([poly, poly[0]])
#                 ax.plot(poly[:, 0], poly[:, 1], linewidth=1.0, alpha=0.6)
#             except Exception:
#                 ax.scatter(pts2[:, 0], pts2[:, 1], s=2, alpha=0.3)

#         ax_xy.set_title("XY"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
#         ax_xz.set_title("XZ"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
#         ax_yz.set_title("YZ"); ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")

#         for ax in (ax_xy, ax_xz, ax_yz):
#             ax.grid(True)
#             ax.set_aspect("equal", adjustable="box")

#         _plot_hull_2d(ax_xy, P_vis[:, [0, 1]])
#         _plot_hull_2d(ax_xz, P_vis[:, [0, 2]])
#         _plot_hull_2d(ax_yz, P_vis[:, [1, 2]])

#         ax_xy.set_xlim(mn[0], mx[0]); ax_xy.set_ylim(mn[1], mx[1])
#         ax_xz.set_xlim(mn[0], mx[0]); ax_xz.set_ylim(mn[2], mx[2])
#         ax_yz.set_xlim(mn[1], mx[1]); ax_yz.set_ylim(mn[2], mx[2])

#         # -------------------------
#         # Handles (pc, opti, trail) for 3D + 2D
#         # -------------------------
#         p0 = P_vis[0]

#         # 3D scatters
#         pc_3d = ax_3d.scatter([p0[0]], [p0[1]], [p0[2]], s=70, c="red", label="pc")
#         opti_3d = ax_3d.scatter([p0[0]], [p0[1]], [p0[2]], s=55, c="blue", label="opti")
#         (trail_3d,) = ax_3d.plot([], [], [], linewidth=1.0, alpha=0.9, label="opti trail")

#         # 2D scatters
#         pc_xy = ax_xy.scatter([p0[0]], [p0[1]], s=60, c="red", label="pc")
#         pc_xz = ax_xz.scatter([p0[0]], [p0[2]], s=60, c="red", label="pc")
#         pc_yz = ax_yz.scatter([p0[1]], [p0[2]], s=60, c="red", label="pc")

#         opti_xy = ax_xy.scatter([p0[0]], [p0[1]], s=45, c="blue", label="opti")
#         opti_xz = ax_xz.scatter([p0[0]], [p0[2]], s=45, c="blue", label="opti")
#         opti_yz = ax_yz.scatter([p0[1]], [p0[2]], s=45, c="blue", label="opti")

#         (trail_xy,) = ax_xy.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
#         (trail_xz,) = ax_xz.plot([], [], linewidth=1.0, alpha=0.8, label="trail")
#         (trail_yz,) = ax_yz.plot([], [], linewidth=1.0, alpha=0.8, label="trail")

#         ax_3d.legend(loc="best")
#         ax_xy.legend(loc="best")
#         ax_xz.legend(loc="best")
#         ax_yz.legend(loc="best")

#         fig.tight_layout()
#         fig.show()
#         plt.pause(0.001)

#         axes = {"3d": ax_3d, "xy": ax_xy, "xz": ax_xz, "yz": ax_yz}
#         pc_handles = {"3d": pc_3d, "xy": pc_xy, "xz": pc_xz, "yz": pc_yz}
#         opti_handles = {"3d": opti_3d, "xy": opti_xy, "xz": opti_xz, "yz": opti_yz}
#         trail_handles = {"3d": trail_3d, "xy": trail_xy, "xz": trail_xz, "yz": trail_yz}

#         return fig, axes, pc_handles, opti_handles, trail_handles

#     def update_point_handle(self, pc_handles, pc: np.ndarray):
#         pc = np.asarray(pc, dtype=float).reshape(3,)

#         # 3D
#         pc_handles["3d"]._offsets3d = ([pc[0]], [pc[1]], [pc[2]])

#         # 2D: set_offsets expects Nx2
#         pc_handles["xy"].set_offsets([[pc[0], pc[1]]])
#         pc_handles["xz"].set_offsets([[pc[0], pc[2]]])
#         pc_handles["yz"].set_offsets([[pc[1], pc[2]]])

#     def update_opti_handle(self, opti_handles, p: np.ndarray):
#         p = np.asarray(p, dtype=float).reshape(3,)

#         opti_handles["3d"]._offsets3d = ([p[0]], [p[1]], [p[2]])
#         opti_handles["xy"].set_offsets([[p[0], p[1]]])
#         opti_handles["xz"].set_offsets([[p[0], p[2]]])
#         opti_handles["yz"].set_offsets([[p[1], p[2]]])

#     def update_trail_handle(self, trail_handles, trail_xyz: np.ndarray):
#         if trail_xyz is None or len(trail_xyz) == 0:
#             trail_handles["3d"].set_data([], [])
#             trail_handles["3d"].set_3d_properties([])

#             trail_handles["xy"].set_data([], [])
#             trail_handles["xz"].set_data([], [])
#             trail_handles["yz"].set_data([], [])
#             return

#         trail_xyz = np.asarray(trail_xyz, dtype=float)

#         # 3D line
#         trail_handles["3d"].set_data(trail_xyz[:, 0], trail_xyz[:, 1])
#         trail_handles["3d"].set_3d_properties(trail_xyz[:, 2])

#         # 2D projections
#         trail_handles["xy"].set_data(trail_xyz[:, 0], trail_xyz[:, 1])
#         trail_handles["xz"].set_data(trail_xyz[:, 0], trail_xyz[:, 2])
#         trail_handles["yz"].set_data(trail_xyz[:, 1], trail_xyz[:, 2])
