#!/usr/bin/env python3
"""
Animate a csv_replay_*.csv recording (see data_logging.py's 'csv' replay
mode -- replay_pwm_csv): raw_flow and processed_flow vs time (starting from
zero), each sensor module as its own line, with a vertical cursor that
sweeps across the plot at real-time speed (matching the recording's actual
elapsed duration, not sped up or slowed down).

Video saving goes through flowbot/video_recorder.py's VideoRecorder (same
one data_logging.py uses) -- MP4 via imageio+imageio-ffmpeg if available,
otherwise an animated GIF automatically, no system ffmpeg install needed.

Usage:
    python python plot_csv_replay.py --csv data/TROpaper_data/perception/csv_replay_20261007_154407.csv 
    --save data/TROpaper_data/perception/test.mp4

    # Save headlessly, no window (deterministic frame timing at --fps)
    python plot_csv_replay.py --csv <path> --save out.mp4 --fps 30

"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from flowbot.video_recorder import VideoRecorder

RAW_COLS  = ["raw_flow1", "raw_flow2", "raw_flow3"]
PROC_COLS = ["proc_flow1", "proc_flow2", "proc_flow3"]
MODULE_LABELS = ["Module 1", "Module 2", "Module 3"]
BELLOW_LABELS = ["Bellow 1", "Bellow 2", "Bellow 3"]  # animate_multi's publication-style naming

OPTI_COLS = ["opti_x_mm", "opti_y_mm", "opti_z_mm"]
OPTI_LABELS = ["X", "Y", "Z"]
AXIS_COLORS = ["#0000FF", "#FF00FF", "#006600"]  # X, Y, Z -- exact hex per reference figure


def load_replay(csv_path):
    df = pd.read_csv(csv_path)
    df["t"] = df["t_s"] - df["t_s"].iloc[0]  # start the time axis from zero
    return df


def normalize_across(dfs, cols):

    for col in cols:
        all_vals = np.concatenate([df[col].to_numpy(dtype=float) for df in dfs if col in df])
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        span = vmax - vmin
        for df in dfs:
            if col in df:
                df[col] = 0.5 if span == 0 else (df[col] - vmin) / span


def compute_ylim(dfs, cols, margin_frac=0.1):
    """
    (ymin, ymax) with a margin_frac headroom/footroom, pooling `cols` across
    every dataframe in `dfs` -- so passing all the dataframes being plotted
    together yields one shared range to apply to every one of their figures,
    making them visually comparable instead of each autoscaling to its own data.
    """
    all_vals = np.concatenate([
        df[col].to_numpy(dtype=float) for df in dfs for col in cols if col in df
    ])
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) == 0:
        return (0.0, 1.0)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    span = vmax - vmin
    pad = span * margin_frac if span > 0 else (abs(vmax) * margin_frac or 1.0)
    return (vmin - pad, vmax + pad)


def _shade_sets(ax, df):
    """Alternating light-blue/white vertical bands, one per PWM set (set_idx)."""
    for i, set_idx in enumerate(df["set_idx"].unique()):
        seg = df[df["set_idx"] == set_idx]
        t0, t1 = seg["t"].iloc[0], seg["t"].iloc[-1]
        if i % 2 == 1:
            ax.axvspan(t0, t1, color="#cfe3f7", alpha=0.5, zorder=0)


def _run_cursor_animation(fig, axes, t_max, save_path=None, fps=30):
    """
    Shared tail end of animate_df/animate_multi: a red vertical cursor swept
    across every panel in `axes` at real-time speed (0..t_max seconds), with
    optional video + .eps export. Used once per figure regardless of how
    many cases/dataframes that figure's panels were built from.
    """
    cursors = [ax.axvline(0, color="red", linewidth=1.5, zorder=5) for ax in axes]

    def _set_cursor(cur_t):
        for c in cursors:
            c.set_xdata([cur_t, cur_t])
        return tuple(cursors)

    plt.tight_layout()

    recorder = VideoRecorder(save_path, fps=fps, fig=fig) if save_path else None

    if save_path:
        # Live playback (optionally also recording, with --show): cursor
        # position driven by actual elapsed wall-clock time, so it tracks
        # real time even if a frame render is slow.
        state = {"t0": None, "closed": False}

        def _finish():
            if not state["closed"]:
                state["closed"] = True
                if recorder is not None:
                    recorder.close()
                if save_path:
                    eps_path = os.path.splitext(save_path)[0] + ".eps"
                    fig.savefig(eps_path, format="eps", bbox_inches="tight")
                    print(f"Saved figure: {eps_path}")

        def _update(_frame):
            if state["t0"] is None:
                state["t0"] = time.time()
            elapsed = time.time() - state["t0"]
            cur_t = min(elapsed, t_max)
            artists = _set_cursor(cur_t)
            if recorder is not None:
                recorder.capture()
            if cur_t >= t_max:
                ani.event_source.stop()
                _finish()
            return artists

        fig.canvas.mpl_connect("close_event", lambda _evt: _finish())
        ani = animation.FuncAnimation(fig, _update, interval=1000.0 / fps,
                                      blit=False, cache_frame_data=False)
        plt.show()
        _finish()  # in case the window was closed before reaching t_max


def animate_df(df, title, save_path=None, fps=30, normalized=False, ylims=None):
    """
    Animate one already-loaded (and possibly normalized) replay dataframe.

    ylims : optional {"raw": (ymin,ymax), "proc": (...), "opti": (...)} --
            when given, that panel's y-axis is fixed to this range instead
            of autoscaling to its own data (see compute_ylim).
    """
    t = df["t"].to_numpy()
    t_max = float(t[-1]) if len(t) else 0.0

    has_opti = all(col in df.columns for col in OPTI_COLS) and \
        not df[OPTI_COLS].isna().all().all()

    n_rows = 3 if has_opti else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True)
    ax_raw, ax_proc = axes[0], axes[1]
    ax_opti = axes[2] if has_opti else None
    fig.canvas.manager.set_window_title(title)

    flow_unit = "Normalized Value" if normalized else "flowrate [L/min]"
    panels = [
        ("raw",  ax_raw,  RAW_COLS,  MODULE_LABELS, f"Raw {flow_unit}",  "Raw Flowrate"),
        ("proc", ax_proc, PROC_COLS, MODULE_LABELS, f"Denoised {flow_unit}", "Denoised Flowrate"),
    ]
    if has_opti:
        panels.append(("opti", ax_opti, OPTI_COLS, OPTI_LABELS, "Position [mm]", "Task Point ($\bf{p}$) Position"))

    for key, ax, cols, labels, ylabel, panel_title in panels:
        _shade_sets(ax, df)
        for col, label in zip(cols, labels):
            ax.plot(t, df[col], label=label, linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, max(t_max, 1e-6))
        if ylims and key in ylims:
            ax.set_ylim(*ylims[key])
        else:
            ax.margins(y=0.1)

    axes[-1].set_xlabel("Time [s]")
    _run_cursor_animation(fig, axes, t_max, save_path=save_path, fps=fps)


LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]


def animate_multi(dfs, case_labels, save_path=None, fps=30, ylims=None, normalized=True):
    """
    Plot several cases (already-loaded, already cross-normalized dataframes)
    in ONE publication-style figure: Raw Flowrate, Denoised Flowrate (both
    by Bellow), and Task Point Position (by local axis X/Y/Z) below. Unlike
    animate_df's single-file view, the per-PWM-set background shading is
    omitted here, matching the comparison-figure style this is for.

    Color identifies bellow/axis (consistent across cases); line style
    identifies which case/file (consistent across bellows/axes). Two
    separate legends per panel make that explicit -- e.g. "Bellow"/color on
    the right, "Load Cases"/line-style on the left.
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    t_arrays = [df["t"].to_numpy() for df in dfs]
    t_max = max((float(t[-1]) if len(t) else 0.0) for t in t_arrays)

    has_opti = any(
        all(col in df.columns for col in OPTI_COLS) and not df[OPTI_COLS].isna().all().all()
        for df in dfs
    )
    n_rows = 3 if has_opti else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3 * n_rows), sharex=True, squeeze=False)
    axes = axes[:, 0]
    ax_raw, ax_proc = axes[0], axes[1]
    ax_opti = axes[2] if has_opti else None
    fig.canvas.manager.set_window_title(" vs ".join(case_labels))

    bellow_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:3]

    flow_unit = "Normalized Value" if normalized else "Flowrate (L/min)"
    # (key, axis, columns, sub-labels, colors, y-label, panel title, legend title)
    panels = [
        ("raw",  ax_raw,  RAW_COLS,  BELLOW_LABELS, bellow_colors, flow_unit,
         "Raw Flowrate", "Bellow"),
        ("proc", ax_proc, PROC_COLS, BELLOW_LABELS, bellow_colors, flow_unit,
         "Denoised Flowrate", "Bellow"),
    ]
    if has_opti:
        panels.append(("opti", ax_opti, OPTI_COLS, OPTI_LABELS, AXIS_COLORS, "Position (mm)",
                       r"Task Point ($\bf{p}$) Position", "Local Axes"))

    for key, ax, cols, sub_labels, colors, ylabel, panel_title, _legend_title in panels:
        for case_i, (df, t) in enumerate(zip(dfs, t_arrays)):
            ls = LINESTYLES[case_i % len(LINESTYLES)]
            for col_i, col in enumerate(cols):
                if col not in df.columns:
                    continue
                ax.plot(t, df[col], color=colors[col_i % len(colors)], linestyle=ls, linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.set_xlim(0, max(t_max, 1e-6))
        if ylims and key in ylims:
            ax.set_ylim(*ylims[key])
        else:
            ax.margins(y=0.1)

    from matplotlib.lines import Line2D
    case_handles = [Line2D([0], [0], color="black", linestyle=LINESTYLES[i % len(LINESTYLES)], lw=1.3)
                    for i in range(len(dfs))]
    for _key, ax, _cols, sub_labels, colors, _ylabel, _panel_title, legend_title in panels:
        id_handles = [Line2D([0], [0], color=colors[i % len(colors)], lw=1.5)
                      for i in range(len(sub_labels))]
        leg1 = ax.legend(id_handles, sub_labels, loc="upper right", fontsize=8, title=legend_title)
        ax.add_artist(leg1)  # keep leg1 drawn even after the next ax.legend() call below
        ax.legend(case_handles, case_labels, loc="upper left", fontsize=8, title="Load Cases")

    axes[-1].set_xlabel("Time (s)")
    _run_cursor_animation(fig, axes, t_max, save_path=save_path, fps=fps)


def _panel_ylims(dfs):
    """Fixed y-range per panel (raw/proc/opti), pooled across all of `dfs`."""
    ylims = {
        "raw": compute_ylim(dfs, RAW_COLS),
        "proc": compute_ylim(dfs, PROC_COLS),
    }
    if any(all(col in df.columns for col in OPTI_COLS) and not df[OPTI_COLS].isna().all().all()
           for df in dfs):  
        ylims["opti"] = compute_ylim(dfs, OPTI_COLS)
    return ylims


def _default_case_label(csv_path):
    """
    'free_load.csv' -> 'Free Load', '100g.csv' -> '100g' -- used as the
    "Load Cases" legend text when --case_labels isn't given explicitly.
    Underscores/dashes become spaces and words are title-cased, except a
    word that already mixes letters and digits (e.g. "100g") is left as-is
    so unit suffixes like g/mm/kg aren't capitalized into "100G".
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    words = stem.replace("_", " ").replace("-", " ").split()
    return " ".join(w if any(c.isdigit() for c in w) else w.capitalize() for w in words)


def animate_replay(csv_path, save_path=None, fps=30, show=False):
    """Single-file convenience wrapper: load csv_path and animate it as-is (no normalization)."""
    df = load_replay(csv_path)
    animate_df(df, title=csv_path, save_path=save_path, fps=fps, normalized=False,
               ylims=_panel_ylims([df]))


def main():
    parser = argparse.ArgumentParser(description="Animate a csv_replay_*.csv recording")
    parser.add_argument("--csv", type=str, required=True, nargs="+",
                        help="Path(s) to the csv_replay_*.csv file(s) to animate. With two or "
                             "more paths, flow data (raw_flow1-3, proc_flow1-3) is first "
                             "min-max normalized -- each module's column pooled across all "
                             "given files, so they share one scale -- then all files are drawn "
                             "together in ONE figure: color still identifies the module, line "
                             "style identifies which file/case.")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the animation to this path (.mp4 or .gif, via flowbot/video_recorder.py's "
                             "VideoRecorder) instead of (or alongside, with --show) showing it live.")
    parser.add_argument("--fps", type=int, default=30, help="Animation frame rate (default 30)")
    parser.add_argument("--case_labels", type=str, nargs="+", default=None,
                        help="Exact 'Load Cases' legend text for each --csv file, in the same order "
                             "(e.g. --case_labels \"Free Load\" \"100g\" \"200g\"). One label per "
                             "--csv file is required if given. Defaults to each file's name, cleaned "
                             "up (underscores/dashes -> spaces, title-cased).")
    args = parser.parse_args()

    if len(args.csv) == 1:
        animate_replay(args.csv[0], save_path=args.save, fps=args.fps)
        return

    dfs = [load_replay(p) for p in args.csv]
    normalize_across(dfs, RAW_COLS + PROC_COLS)
    ylims = _panel_ylims(dfs)
    if args.case_labels:
        if len(args.case_labels) != len(args.csv):
            parser.error(f"--case_labels got {len(args.case_labels)} label(s) but --csv got "
                         f"{len(args.csv)} file(s) -- need exactly one label per file.")
        case_labels = args.case_labels
    else:
        case_labels = [_default_case_label(p) for p in args.csv]
    animate_multi(dfs, case_labels, save_path=args.save, fps=args.fps, ylims=ylims)


if __name__ == "__main__":
    main()
