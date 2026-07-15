"""
analyze_task.py  –  Post-process task CSV logs.

Supports two CSV formats automatically:
  OLD: columns include opti_x / opti_y / opti_z  (raw world-frame metres)
       → applies opti_to_manip_mm transform internally
  NEW: columns include opti_mm_x / opti_mm_y / opti_mm_z  (manip-frame mm)
       → used directly, no transform needed

For each waypoint the last row of each hold phase is taken as the
settled measurement.  Home position (|x|<0.1 and |y|<0.1 mm) is skipped.

Usage — single file:
    python flowbot/analyze_task.py data/task_logs/circle_r15_xxx.csv

Usage — whole folder (batch):
    python flowbot/analyze_task.py --folder paper_data/
    python flowbot/analyze_task.py --folder paper_data/ --save-figs
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── OptiTrack → manipulator-frame transform (old format only) ─────────────────
_R_MW = np.array([[0.0,  0.0,  1.0],
                  [-1.0, 0.0,  0.0],
                  [0.0, -1.0,  0.0]])


def _Rz(alpha: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])


def opti_to_manip_mm(pos_W_m, origin_W_m, alpha_rad) -> np.ndarray:
    p_rel = np.asarray(pos_W_m, dtype=float) - np.asarray(origin_W_m, dtype=float)
    pM = _Rz(alpha_rad) @ (_R_MW @ p_rel)
    pM *= 1000.0
    pM[0] = -pM[0]
    pM[1] = -pM[1]
    return pM


# ── Format detection ──────────────────────────────────────────────────────────
def _detect_format(df: pd.DataFrame) -> str:
    if "opti_mm_x" in df.columns:
        return "new"
    if "opti_x" in df.columns:
        return "old"
    return "unknown"


# ── Hold-phase endpoint detection ─────────────────────────────────────────────
def extract_hold_endpoints(df: pd.DataFrame, tol: float = 0.01) -> list[int]:
    """Return row index of the LAST tick of each hold phase."""
    cmd   = df[["cmd_pc_x", "cmd_pc_y", "cmd_pc_z"]].values
    delta = np.linalg.norm(np.diff(cmd, axis=0), axis=1)

    hold_end_indices: list[int] = []
    in_hold = False

    for i, d in enumerate(delta):
        if d <= tol:
            if not in_hold:
                in_hold = True
        else:
            if in_hold:
                hold_end_indices.append(i)
                in_hold = False

    if in_hold:
        hold_end_indices.append(len(df) - 1)

    return hold_end_indices


# ── Per-file analysis ─────────────────────────────────────────────────────────
def analyze_file(csv_path: Path,
                 l0: float = 82.0,
                 lu: float = 13.5,
                 alpha_deg: float = -30.0,
                 tol: float = 0.01,
                 save_fig: bool = False,
                 show_fig: bool = True) -> pd.DataFrame | None:
    """
    Analyse one CSV.  Returns a DataFrame of per-waypoint results, or None on error.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[error] Cannot read {csv_path.name}: {e}")
        return None

    fmt = _detect_format(df)
    if fmt == "unknown":
        print(f"[warn] {csv_path.name}: no recognised optitrack columns, skipping.")
        return None

    print(f"\n── {csv_path.name}  (format={fmt}, {len(df)} rows) ──")

    # ── Get measured position per row ─────────────────────────────────────
    if fmt == "new":
        # Already in manipulator frame (mm)
        meas_cols = ["opti_mm_x", "opti_mm_y", "opti_mm_z"]
        meas_xyz  = df[meas_cols].values.astype(float)

    else:  # old format — apply transform
        alpha_rad = np.deg2rad(alpha_deg)
        l0_lu_m   = (l0 + lu) / 1000.0

        valid = df[df["opti_x"].notna() & (df["opti_x"] != 0.0)]
        if valid.empty:
            print(f"[warn] {csv_path.name}: no valid opti readings.")
            return None
        first = valid.iloc[0]
        origin = np.array([first["opti_x"],
                           first["opti_y"] + l0_lu_m,
                           first["opti_z"]], dtype=float)
        print(f"  opti origin (world m): {np.round(origin, 4)}")

        raw_xyz = df[["opti_x", "opti_y", "opti_z"]].values.astype(float)
        meas_xyz = np.array([
            opti_to_manip_mm(row, origin, alpha_rad) for row in raw_xyz
        ])

    # ── Detect hold endpoints ─────────────────────────────────────────────
    hold_rows = extract_hold_endpoints(df, tol=tol)
    print(f"  hold endpoints detected: {len(hold_rows)}")

    # Filter out home position (|x|<0.1 and |y|<0.1 mm)
    waypoint_rows = [
        idx for idx in hold_rows
        if not (abs(df.iloc[idx]["cmd_pc_x"]) < 0.1 and
                abs(df.iloc[idx]["cmd_pc_y"]) < 0.1)
    ]
    if not waypoint_rows:
        print(f"  [warn] No non-home waypoints found; using all endpoints.")
        waypoint_rows = hold_rows

    print(f"  task waypoints: {len(waypoint_rows)}")

    # ── Build results ─────────────────────────────────────────────────────
    results = []
    for wp_i, row_idx in enumerate(waypoint_rows):
        row    = df.iloc[row_idx]
        target = np.array([row["cmd_pc_x"], row["cmd_pc_y"], row["cmd_pc_z"]], dtype=float)
        meas   = meas_xyz[row_idx]
        err    = meas - target
        results.append({
            "wp":      wp_i + 1,
            "t_s":     row["t_s"],
            "tgt_x":   target[0], "tgt_y":  target[1], "tgt_z":  target[2],
            "meas_x":  meas[0],   "meas_y": meas[1],   "meas_z": meas[2],
            "err_x":   err[0],    "err_y":  err[1],     "err_z":  err[2],
            "dist_mm": float(np.linalg.norm(err)) if not np.any(np.isnan(err)) else np.nan,
        })

    rdf = pd.DataFrame(results)

    # ── Console table ─────────────────────────────────────────────────────
    print(f"\n  {'WP':>4}  {'t(s)':>7}  "
          f"{'tgt_x':>7} {'tgt_y':>7} {'tgt_z':>7}  "
          f"{'err_x':>7} {'err_y':>7} {'err_z':>7}  {'dist':>7}")
    print("  " + "-" * 73)
    for _, r in rdf.iterrows():
        dist_s = f"{r.dist_mm:>7.2f}" if not np.isnan(r.dist_mm) else f"{'NaN':>7}"
        print(f"  {int(r.wp):>4}  {r.t_s:>7.2f}  "
              f"{r.tgt_x:>7.2f} {r.tgt_y:>7.2f} {r.tgt_z:>7.2f}  "
              f"{r.err_x:>7.2f} {r.err_y:>7.2f} {r.err_z:>7.2f}  {dist_s}")

    valid = rdf.dropna(subset=["dist_mm"])
    if not valid.empty:
        print("  " + "-" * 73)
        print(f"  {'MEAN':>4}  {'':>7}  {'':>7} {'':>7} {'':>7}  "
              f"{valid.err_x.mean():>7.2f} {valid.err_y.mean():>7.2f} "
              f"{valid.err_z.mean():>7.2f}  {valid.dist_mm.mean():>7.2f}")
        print(f"  {'RMSE':>4}  {'':>7}  {'':>7} {'':>7} {'':>7}  "
              f"{np.sqrt((valid.err_x**2).mean()):>7.2f} "
              f"{np.sqrt((valid.err_y**2).mean()):>7.2f} "
              f"{np.sqrt((valid.err_z**2).mean()):>7.2f}  "
              f"{np.sqrt((valid.dist_mm**2).mean()):>7.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    wp_idx  = rdf["wp"].values
    kw_line = dict(linewidth=1.5, marker="o", markersize=6)

    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"Waypoint error — {csv_path.stem}", fontsize=11)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.2)

    ax_dist = fig.add_subplot(gs[0, :])
    ax_ex   = fig.add_subplot(gs[1, 0])
    ax_ey   = fig.add_subplot(gs[1, 1])

    ax_dist.plot(wp_idx, rdf["dist_mm"], color="steelblue", **kw_line, label="dist error")
    if not valid.empty:
        ax_dist.axhline(valid["dist_mm"].mean(), color="red", linestyle="--",
                        linewidth=1.0, label=f"mean={valid['dist_mm'].mean():.2f} mm")
    ax_dist.set_xlabel("Waypoint"); ax_dist.set_ylabel("Distance error (mm)")
    ax_dist.set_title("Euclidean distance error"); ax_dist.legend(); ax_dist.grid(alpha=0.4)

    ax_ex.plot(wp_idx, rdf["err_x"], color="tab:red", **kw_line, label=f"mean|e|={valid['err_x'].abs().mean():.2f} mm")
    ax_ex.axhline(0, color="black", linewidth=0.8)
    # if not valid.empty:
    #     ax_ex.axhline(valid["err_x"].abs().mean(), color="red", linestyle="--", linewidth=1.0,
    #                   label=f"mean|e|={valid['err_x'].abs().mean():.2f} mm")
    ax_ex.set_xlabel("Waypoint"); ax_ex.set_ylabel("Error (mm)")
    ax_ex.set_title("X error"); ax_ex.legend(fontsize=8); ax_ex.grid(alpha=0.4)

    ax_ey.plot(wp_idx, rdf["err_y"], color="tab:orange", **kw_line, label=f"mean|e|={valid['err_y'].abs().mean():.2f} mm")
    ax_ey.axhline(0, color="black", linewidth=0.8)
    # if not valid.empty:
    #     ax_ey.axhline(valid["err_y"].abs().mean(), color="red", linestyle="--", linewidth=1.0,
    #                   label=f"mean|e|={valid['err_y'].abs().mean():.2f} mm")
    ax_ey.set_xlabel("Waypoint"); ax_ey.set_ylabel("Error (mm)")
    ax_ey.set_title("Y error"); ax_ey.legend(fontsize=8); ax_ey.grid(alpha=0.4)

    ax_ez = fig.add_subplot(gs[2, :])
    ax_ez.plot(wp_idx, rdf["err_z"], color="tab:green", **kw_line, label=f"mean|e|={valid['err_z'].abs().mean():.2f} mm")
    ax_ez.axhline(0, color="black", linewidth=0.8)
    # if not valid.empty:
    #     ax_ez.axhline(valid["err_z"].abs().mean(), color="red", linestyle="--", linewidth=1.0,
    #                   label=f"mean|e|={valid['err_z'].abs().mean():.2f} mm")
    ax_ez.set_xlabel("Waypoint"); ax_ez.set_ylabel("Error (mm)")
    ax_ez.set_title("Z error"); ax_ez.legend(fontsize=8); ax_ez.grid(alpha=0.4)

    if len(wp_idx) > 20:
        step = max(1, len(wp_idx) // 10)
        for ax in fig.axes:
            ax.set_xticks(wp_idx[::step])
    else:
        for ax in fig.axes:
            ax.set_xticks(wp_idx)

    plt.tight_layout()

    if save_fig:
        fig_path = csv_path.with_name(csv_path.stem + "_eval.eps")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"  Figure saved: {fig_path.name}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)

    return rdf


# ── Batch folder analysis ─────────────────────────────────────────────────────
def analyze_folder(folder: Path, save_figs: bool, **kwargs) -> None:
    csvs = sorted(folder.glob("*.csv"))
    if not csvs:
        print(f"[error] No CSV files found in {folder}")
        return

    print(f"Found {len(csvs)} CSV file(s) in {folder}\n")

    summary_rows = []
    for csv_path in csvs:
        rdf = analyze_file(csv_path, save_fig=save_figs, show_fig=False, **kwargs)
        if rdf is None:
            continue
        valid = rdf.dropna(subset=["dist_mm"])
        if valid.empty:
            continue
        summary_rows.append({
            "file":          csv_path.name,
            "n_waypoints":   len(rdf),
            "mean_dist_mm":  valid["dist_mm"].mean(),
            "rmse_dist_mm":  float(np.sqrt((valid["dist_mm"] ** 2).mean())),
            "mean_err_x":    valid["err_x"].mean(),
            "mean_err_y":    valid["err_y"].mean(),
            "mean_err_z":    valid["err_z"].mean(),
            "rmse_x":        float(np.sqrt((valid["err_x"] ** 2).mean())),
            "rmse_y":        float(np.sqrt((valid["err_y"] ** 2).mean())),
            "rmse_z":        float(np.sqrt((valid["err_z"] ** 2).mean())),
        })

    if not summary_rows:
        print("[warn] No usable results.")
        return

    sdf = pd.DataFrame(summary_rows)
    print("\n\n══ FOLDER SUMMARY ══════════════════════════════════════════════════")
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
    print(f"\nOverall mean dist : {sdf['mean_dist_mm'].mean():.2f} mm")
    print(f"Overall RMSE dist : {sdf['rmse_dist_mm'].mean():.2f} mm")

    out_csv = folder / "summary.csv"
    sdf.to_csv(out_csv, index=False)
    print(f"\nSummary saved to: {out_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analyse task execution error from CSV log(s).")
    ap.add_argument("csv",        nargs="?", default=None,
                    help="Path to a single task log CSV.")
    ap.add_argument("--output",   default=None,
                    help="Process all CSVs in this folder and print a summary.")
    ap.add_argument("--save-fig", action="store_true",
                    help="Save each figure as a PNG alongside the CSV.")
    ap.add_argument("--l0",    type=float, default=82.0)
    ap.add_argument("--lu",    type=float, default=13.5)
    ap.add_argument("--alpha", type=float, default=-30.0,
                    help="OptiTrack rotation angle (degrees, old format only).")
    ap.add_argument("--tol",   type=float, default=0.01,
                    help="cmd_pc stability tolerance for hold detection (mm).")
    args = ap.parse_args()

    kwargs = dict(l0=args.l0, lu=args.lu, alpha_deg=args.alpha, tol=args.tol)

    if args.output:
        folder = Path(args.output)
        if not folder.is_dir():
            print(f"[error] Not a directory: {folder}")
            sys.exit(1)
        analyze_folder(folder, save_figs=args.save_fig, **kwargs)

    elif args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"[error] File not found: {csv_path}")
            sys.exit(1)
        analyze_file(csv_path, save_fig=args.save_fig, show_fig=True, **kwargs)

    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
