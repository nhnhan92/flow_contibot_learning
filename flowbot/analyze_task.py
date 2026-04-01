"""
analyze_task.py  –  Post-process a task CSV log.

For each waypoint, extracts the OptiTrack position at the END of the hold
(phase 2) and compares it with the commanded target position (cmd_pc).

Outputs:
  - Console table: waypoint index, target, measured, error
  - Figure: per-waypoint error in X, Y, Z and Euclidean distance

Usage:
    python analyze_task.py data/task_logs/circle_xy_20250314_123456.csv
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# OptiTrack → manipulator-frame transform
# (mirrors online_optitrack.OptiTrack.opti_to_manip + axis flips in execute_task)
# ──────────────────────────────────────────────────────────────────────────────
R_MW = np.array([[0.0,  0.0,  1.0],
                 [-1.0, 0.0,  0.0],
                 [0.0, -1.0,  0.0]])


def _Rz(alpha: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])


def opti_to_manip_mm(pos_W_m: np.ndarray,
                     origin_W_m: np.ndarray,
                     alpha: float) -> np.ndarray:
    """Transform raw OptiTrack world-frame position (m) to manipulator frame (mm)."""
    p_rel = np.asarray(pos_W_m, dtype=float) - np.asarray(origin_W_m, dtype=float)
    pM = _Rz(alpha) @ (R_MW @ p_rel)
    pM *= 1000.0          # m → mm
    pM[0] = -pM[0]        # axis flip (same as execute_task)
    pM[1] = -pM[1]
    return pM


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_hold_endpoints(df: pd.DataFrame,
                           tol: float = 0.01) -> list[int]:
    """
    Return row indices corresponding to the LAST tick of each hold phase.

    A hold phase is a block of consecutive rows where cmd_pc doesn't change
    (within `tol` mm).  The last row of each block = end of phase 2.
    """
    cmd = df[["cmd_pc_x", "cmd_pc_y", "cmd_pc_z"]].values
    # Compute step-to-step displacement in cmd_pc
    delta = np.linalg.norm(np.diff(cmd, axis=0), axis=1)  # length N-1

    hold_end_indices: list[int] = []
    in_hold = False
    hold_start = None

    for i, d in enumerate(delta):
        moving = d > tol
        if not moving:
            if not in_hold:
                in_hold = True
                hold_start = i          # first row of this hold
        else:
            if in_hold:
                # Row i is still part of the hold (delta[i] = cmd[i+1]-cmd[i])
                hold_end_indices.append(i)
                in_hold = False

    # Handle hold that runs to the very end of the file
    if in_hold:
        hold_end_indices.append(len(df) - 1)

    return hold_end_indices


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analyse task execution error from CSV log.")
    ap.add_argument("csv", help="Path to task log CSV file.")
    ap.add_argument("--l0",    type=float, default=82.0,
                    help="Lower bellow active length l0 (mm, default 82).")
    ap.add_argument("--lu",    type=float, default=13.5,
                    help="Upper passive structure length lu (mm, default 13.5).")
    ap.add_argument("--alpha", type=float, default=-30.0,
                    help="OptiTrack rotation angle alpha (degrees, default -30).")
    ap.add_argument("--tol",   type=float, default=0.01,
                    help="cmd_pc stability tolerance to detect hold phase (mm, default 0.01).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[info] Loaded {len(df)} rows from {csv_path.name}")

    # ── Reconstruct OptiTrack origin from first valid reading ──────────────
    alpha_rad = np.deg2rad(args.alpha)
    l0_lu_m   = (args.l0 + args.lu) / 1000.0   # mm → m

    first_valid = df[df["opti_x"].notna() & (df["opti_x"] != 0.0)].iloc[0]
    opti_origin = np.array([
        first_valid["opti_x"],
        first_valid["opti_y"] + l0_lu_m,
        first_valid["opti_z"],
    ], dtype=float)
    print(f"[info] OptiTrack origin (world m): {np.round(opti_origin, 4)}")

    # ── Detect hold-phase end rows ────────────────────────────────────────
    hold_rows = extract_hold_endpoints(df, tol=args.tol)
    print(f"[info] Detected {len(hold_rows)} hold-phase endpoints")

    # Drop the last one if it's the return-to-home (optional heuristic:
    # home is near [0,0,z_init] so cmd_pc_x ≈ 0 and cmd_pc_y ≈ 0)
    waypoint_rows = []
    for idx in hold_rows:
        row = df.iloc[idx]
        cx, cy = row["cmd_pc_x"], row["cmd_pc_y"]
        if abs(cx) < 0.1 and abs(cy) < 0.1:
            continue   # skip home position
        waypoint_rows.append(idx)

    if len(waypoint_rows) == 0:
        print("[warn] No non-home waypoints found; using all hold endpoints.")
        waypoint_rows = hold_rows

    print(f"[info] Task waypoints: {len(waypoint_rows)}")

    # ── Build results table ───────────────────────────────────────────────
    results = []
    for wp_i, row_idx in enumerate(waypoint_rows):
        row = df.iloc[row_idx]
        target = np.array([row["cmd_pc_x"], row["cmd_pc_y"], row["cmd_pc_z"]])

        pos_W = np.array([row["opti_x"], row["opti_y"], row["opti_z"]])
        measured = opti_to_manip_mm(pos_W, opti_origin, alpha_rad)
        err = measured - target

        results.append({
            "wp":      wp_i + 1,
            "t_s":     row["t_s"],
            "tgt_x":   target[0],  "tgt_y":  target[1],  "tgt_z":  target[2],
            "meas_x":  measured[0],"meas_y": measured[1], "meas_z": measured[2],
            "err_x":   err[0],     "err_y":  err[1],      "err_z":  err[2],
            "dist_mm": np.linalg.norm(err) if not np.isnan(err[0]) else np.nan,
        })

    rdf = pd.DataFrame(results[0:60])  # results DataFrame

    # ── Print table ───────────────────────────────────────────────────────
    print("\n── Waypoint errors ──────────────────────────────────────────────────")
    print(f"{'WP':>4}  {'t(s)':>7}  "
          f"{'tgt_x':>7} {'tgt_y':>7} {'tgt_z':>7}  "
          f"{'err_x':>7} {'err_y':>7} {'err_z':>7}  {'dist':>7}")
    print("-" * 75)
    for _, r in rdf.iterrows():
        print(f"{int(r.wp):>4}  {r.t_s:>7.2f}  "
              f"{r.tgt_x:>7.2f} {r.tgt_y:>7.2f} {r.tgt_z:>7.2f}  "
              f"{r.err_x:>7.2f} {r.err_y:>7.2f} {r.err_z:>7.2f}  "
              f"{r.dist_mm:>7.2f}")

    if not rdf["dist_mm"].isna().all():
        print("-" * 75)
        print(f"{'MEAN':>4}  {'':>7}  {'':>7} {'':>7} {'':>7}  "
              f"{rdf.err_x.mean():>7.2f} {rdf.err_y.mean():>7.2f} "
              f"{rdf.err_z.mean():>7.2f}  {rdf.dist_mm.mean():>7.2f}")
        print(f"{'RMSE':>4}  {'':>7}  {'':>7} {'':>7} {'':>7}  "
              f"{np.sqrt((rdf.err_x**2).mean()):>7.2f} "
              f"{np.sqrt((rdf.err_y**2).mean()):>7.2f} "
              f"{np.sqrt((rdf.err_z**2).mean()):>7.2f}  "
              f"{np.sqrt((rdf.dist_mm**2).mean()):>7.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    wp_idx = rdf["wp"].values

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f"Waypoint Tracking Error — {csv_path.stem}", fontsize=12)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_dist = fig.add_subplot(gs[0, :])   # top row, full width
    ax_ex   = fig.add_subplot(gs[1, 0])
    ax_ey   = fig.add_subplot(gs[1, 1])

    kw_line = dict(linewidth=1.5, marker="o", markersize=4)

    # ── Distance error (top) ──────────────────────────────────────────────
    ax_dist.plot(wp_idx, rdf["dist_mm"], color="steelblue", **kw_line, label="dist error")
    ax_dist.axhline(rdf["dist_mm"].mean(), color="red", linestyle="--",
                    linewidth=1.0, label=f"mean = {rdf['dist_mm'].mean():.2f} mm")
    ax_dist.set_xlabel("Waypoint index")
    ax_dist.set_ylabel("Distance error (mm)")
    ax_dist.set_title("Euclidean distance error")
    ax_dist.legend()
    ax_dist.set_xticks(wp_idx)
    ax_dist.grid(alpha=0.4)

    # ── X error ───────────────────────────────────────────────────────────
    ax_ex.plot(wp_idx, rdf["err_x"], color="tab:red", **kw_line)
    ax_ex.axhline(0, color="black", linewidth=0.8)
    ax_ex.axhline(rdf["err_x"].abs().mean(), color="red", linestyle="--",
                  linewidth=1.0, label=f"mean|err| = {rdf['err_x'].abs().mean():.2f} mm")
    ax_ex.set_xlabel("Waypoint index")
    ax_ex.set_ylabel("Error (mm)")
    ax_ex.set_title("X error")
    ax_ex.legend(fontsize=8)
    ax_ex.set_xticks(wp_idx)
    ax_ex.grid(alpha=0.4)

    # ── Y error ───────────────────────────────────────────────────────────
    ax_ey.plot(wp_idx, rdf["err_y"], color="tab:orange", **kw_line)
    ax_ey.axhline(0, color="black", linewidth=0.8)
    ax_ey.axhline(rdf["err_y"].abs().mean(), color="red", linestyle="--",
                  linewidth=1.0, label=f"mean|err| = {rdf['err_y'].abs().mean():.2f} mm")
    ax_ey.set_xlabel("Waypoint index")
    ax_ey.set_ylabel("Error (mm)")
    ax_ey.set_title("Y error")
    ax_ey.legend(fontsize=8)
    ax_ey.set_xticks(wp_idx)
    ax_ey.grid(alpha=0.4)

    # ── Z error (only if task has Z variation) ────────────────────────────
    z_range = rdf["tgt_z"].max() - rdf["tgt_z"].min()
    if z_range > 0.5:   # task has meaningful Z motion
        gs2 = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)
        ax_dist.set_subplotspec(gs2[0, :])
        ax_ex.set_subplotspec(gs2[1, 0])
        ax_ey.set_subplotspec(gs2[1, 1])
        ax_ez = fig.add_subplot(gs2[2, :])
        ax_ez.plot(wp_idx, rdf["err_z"], color="tab:green", **kw_line)
        ax_ez.axhline(0, color="black", linewidth=0.8)
        ax_ez.axhline(rdf["err_z"].abs().mean(), color="red", linestyle="--",
                      linewidth=1.0, label=f"mean|err| = {rdf['err_z'].abs().mean():.2f} mm")
        ax_ez.set_xlabel("Waypoint index")
        ax_ez.set_ylabel("Error (mm)")
        ax_ez.set_title("Z error")
        ax_ez.legend(fontsize=8)
        ax_ez.set_xticks(wp_idx)
        ax_ez.grid(alpha=0.4)

    # Thin the x-tick labels if there are many waypoints
    for ax in fig.axes:
        if len(wp_idx) > 20:
            step = max(1, len(wp_idx) // 10)
            ax.set_xticks(wp_idx[::step])

    plt.show()


if __name__ == "__main__":
    main()
