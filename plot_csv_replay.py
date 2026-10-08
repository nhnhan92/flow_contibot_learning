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


def load_replay(csv_path):
    df = pd.read_csv(csv_path)
    df["t"] = df["t_s"] - df["t_s"].iloc[0]  # start the time axis from zero
    return df


def _shade_sets(ax, df):
    """Alternating light-blue/white vertical bands, one per PWM set (set_idx)."""
    for i, set_idx in enumerate(df["set_idx"].unique()):
        seg = df[df["set_idx"] == set_idx]
        t0, t1 = seg["t"].iloc[0], seg["t"].iloc[-1]
        if i % 2 == 1:
            ax.axvspan(t0, t1, color="#cfe3f7", alpha=0.5, zorder=0)


def animate_replay(csv_path, save_path=None, fps=30, show=False):
    df = load_replay(csv_path)
    t = df["t"].to_numpy()
    t_max = float(t[-1]) if len(t) else 0.0

    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.canvas.manager.set_window_title(csv_path)

    for ax, cols, ylabel, title in [
        (ax_raw,  RAW_COLS,  "Raw flowrate [L/min]",  "Raw Flow"),
        (ax_proc, PROC_COLS, "Denoised flowrate [L/min]", "Denoised Flow"),
    ]:
        _shade_sets(ax, df)
        for col, label in zip(cols, MODULE_LABELS):
            ax.plot(t, df[col], label=label, linewidth=1.3)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, max(t_max, 1e-6))
        ax.margins(y=0.1)

    ax_proc.set_xlabel("Time [s]")

    # Vertical cursor: a red line through both panels, with a circular
    # handle at the top of the upper panel (see the reference image).
    cursor_raw  = ax_raw.axvline(0, color="red", linewidth=1.5, zorder=5)
    cursor_proc = ax_proc.axvline(0, color="red", linewidth=1.5, zorder=5)

    def _set_cursor(cur_t):
        cursor_raw.set_xdata([cur_t, cur_t])
        cursor_proc.set_xdata([cur_t, cur_t])
        return cursor_raw, cursor_proc

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


def main():
    parser = argparse.ArgumentParser(description="Animate a csv_replay_*.csv recording")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to the csv_replay_*.csv file to animate")
    parser.add_argument("--save", type=str, default=None,
                        help="Save the animation to this path (.mp4 or .gif, via flowbot/video_recorder.py's "
                             "VideoRecorder) instead of (or alongside, with --show) showing it live")
    parser.add_argument("--fps", type=int, default=30, help="Animation frame rate (default 30)")
    args = parser.parse_args()
    animate_replay(args.csv, save_path=args.save, fps=args.fps)


if __name__ == "__main__":
    main()
