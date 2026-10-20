#!/usr/bin/env python3
"""
Plot deployment logs saved by DeploymentLogger (deploy/deployment_logger.py,
used by deploy_flowbot_w_policy.py and deploy_baseline_w_policy.py).

Reads one .npz episode log (or every .npz in a directory) and renders one
figure per episode, on a shared time axis (seconds from episode start):
    event timeline   phases (UR5 / Flowbot / Release), sealed, and suction on
                     (baselines only -- Flowbot/fluxflex has no suction action,
                     so nothing about suction on/off is plotted for it)
    vacuum pressure  with the seal threshold and suction / seal / release marks
    suction command  policy's raw suction output (baselines only)
    TCP position, PWM commanded, 3D TCP path, inference latency
    info panel       operator report (objects, success, failure categories),
                     key times, system, checkpoint, video file

Sealed = pressure <= seal threshold (kPa, negative). It is recomputed here, so a
threshold can be chosen after recording: --seal_threshold_kpa overrides the
logged one, and an invalid logged threshold (missing or >= 0) is ignored.

For a directory, a per-system summary table (success rate, failure counts,
completion time, time to seal) is printed after the figures.

Usage:
    python plot_deployment_log.py path/to/episode_000_20261020_112004.npz
    python plot_deployment_log.py path/to/log_dir/ --seal_threshold_kpa -15
    python plot_deployment_log.py path/to/log_dir/ --output_dir path/to/figs --show
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Palette (reference instance, see the dataviz skill's palette.md) ──────────
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
               '#e87ba4', '#008300', '#6250d6', '#e34948']
GRID_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'
MUTED_INK = '#898781'
PRIMARY_INK = '#0b0b0b'
SECONDARY_INK = '#52514e'
SURFACE = '#fcfcfb'

# Sequential blue ramp (100 -> 700) for progression (time along the path) only.
_BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
              '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list('seq_blue', _BLUE_RAMP)

# Phases are drawn in neutral steps with text labels (identity never by color
# alone), so they don't reuse the series hues of the TCP / PWM panels.
PHASES = {  # op_mode (ur5_active, flowbot_active) -> (label, fill, label ink)
    (1, 0): ('UR5', '#e1e0d9', SECONDARY_INK),
    (0, 1): ('Flowbot', '#c3c2b7', PRIMARY_INK),
    (1, 1): ('Release', '#52514e', SURFACE),
    (0, 0): ('Idle', SURFACE, MUTED_INK),
}
STATE_FILL = SECONDARY_INK   # suction-on / sealed bars
COUPLED_SYSTEMS = ('flowbot', 'fluxflex')   # suction implicit: no suction action to plot
TCP_AXIS_LABELS = ['x', 'y', 'z', 'rx', 'ry', 'rz']
FAILURE_LABELS = [
    ('wrong_object', 'wrong object'),
    ('premature_suction', 'premature suction'),
    ('late_suction_no_seal', 'late suction / no seal'),
    ('dropped_or_placing_error', 'dropped / placing error'),
    ('collision', 'collision'),
    ('timeout', 'timeout'),
    ('other', 'other'),
]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def _scalar(data, key, default):
    if key not in data:
        return default
    v = data[key]
    return v.item() if hasattr(v, 'item') else v


def _runs(mask_or_keys):
    """Contiguous runs: [(start, end_exclusive, value), ...]."""
    keys = list(mask_or_keys)
    runs, s = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[s]:
            runs.append((s, i, keys[s]))
            s = i
    return runs


def _first_time(mask, t):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else float('nan')


def load_episode(npz_path):
    """Load one DeploymentLogger .npz into a dict. Older logs (before op_mode /
    pressure / outcome were logged) still load; missing fields stay absent."""
    data = dict(np.load(npz_path, allow_pickle=True))
    actions = data['executed_actions']
    tcp_dims = data['tcp_poses'].shape[1]
    if 'op_mode' not in data and actions.shape[1] >= tcp_dims + 5:
        data['op_mode'] = np.clip(np.round(actions[:, tcp_dims + 3:tcp_dims + 5]), 0, 1).astype(int)
    # Flowbot/fluxflex has no suction action (suction is implicit in the flow):
    # drop suction fields even if an older log recorded them, so nothing about
    # suction on/off is plotted or summarized for it.
    if str(_scalar(data, 'system', '')).lower() in COUPLED_SYSTEMS:
        data.pop('suction_on', None)
        data.pop('suction_cmd_raw', None)
    ts = data['timestamps']
    t0 = float(_scalar(data, 'episode_start_time', float('nan')))
    if not np.isfinite(t0):
        t0 = float(ts[0]) if len(ts) else 0.0
    data['t'] = ts - t0
    return data


def resolve_threshold(data, override):
    """Seal threshold to use, and a note on where it came from."""
    if override is not None:
        return float(override), 'from --seal_threshold_kpa'
    logged = float(_scalar(data, 'seal_threshold_kpa', float('nan')))
    if np.isfinite(logged) and logged < 0:
        return logged, 'as logged'
    if np.isfinite(logged):
        return None, f'logged {logged:g} kPa ignored (must be negative)'
    return None, 'none logged'


def episode_stats(data, threshold):
    t = data['t']
    op = data.get('op_mode')
    pressure = data.get('pressure_kpa')
    sealed = (pressure <= threshold) if (pressure is not None and threshold is not None) else None
    suction_on = data.get('suction_on')
    release = (op[:, 0] == 1) & (op[:, 1] == 1) if op is not None else np.zeros(len(t), bool)
    return {
        'sealed': sealed,
        'completion_s': _first_time(release, t),
        'suction_s': _first_time(suction_on, t) if suction_on is not None else float('nan'),
        'seal_s': _first_time(sealed, t) if sealed is not None else float('nan'),
    }


EVENT_LINES = (('suction_s', 'suction on', ':'), ('seal_s', 'sealed', '--'),
               ('completion_s', 'release', '-.'))


def _mark_events(ax, stats):
    """Thin vertical lines at suction on / sealed / release (keyed in the
    timeline's legend, by line style)."""
    for key, _, style in EVENT_LINES:
        x = stats[key]
        if np.isfinite(x):
            ax.axvline(x, color=MUTED_INK, linewidth=0.9, linestyle=style, zorder=1)


def _top_legend(ax, handles=None):
    """Legend above the axes' top-right corner, clear of the data."""
    kw = {'handles': handles} if handles is not None else {}
    n = len(handles) if handles is not None else len(ax.get_legend_handles_labels()[0])
    ax.legend(**kw, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=n, frameon=False,
              fontsize=7.5, labelcolor=SECONDARY_INK, borderaxespad=0.2, handlelength=1.8)


def _phase_boundaries(ax, data):
    op = data.get('op_mode')
    if op is None:
        return
    t = data['t']
    for s, _, _ in _runs(tuple(int(v) for v in row) for row in op)[1:]:
        ax.axvline(t[s], color=GRID_COLOR, linewidth=1.0, zorder=0.5)


def _timeline(ax, data, stats):
    """Rows, top to bottom: phase segments (labelled), suction on (only for
    systems with an explicit suction action), sealed."""
    t = data['t']
    if len(t) == 0:
        return
    dt = np.diff(t)
    ends = np.append(t[1:], t[-1] + (np.median(dt) if len(dt) else 0.1))
    state_rows = ([('suction on', data['suction_on'])] if 'suction_on' in data else []) \
        + [('sealed', stats['sealed'])]
    top = len(state_rows)   # phase row y; state rows below it
    op = data.get('op_mode')
    if op is not None:
        for s, e, key in _runs(tuple(int(v) for v in row) for row in op):
            label, fill, ink = PHASES.get(key, (str(key), SURFACE, MUTED_INK))
            ax.broken_barh([(t[s], ends[e - 1] - t[s])], (top - 0.4, 0.8), facecolors=fill,
                           edgecolors=SURFACE, linewidth=1.0)
            width = ends[e - 1] - t[s]
            if width > 0.04 * (ends[-1] - t[0]):
                ax.text(t[s] + width / 2, top, label, ha='center', va='center', fontsize=7, color=ink)
    for i, (name, mask) in enumerate(state_rows):
        y = top - 1 - i
        if mask is None:
            ax.text(t[0], y, '  not available (no seal threshold)', va='center', fontsize=7, color=MUTED_INK)
            continue
        for s, e, on in _runs(mask.tolist()):
            if on:
                ax.broken_barh([(t[s], ends[e - 1] - t[s])], (y - 0.3, 0.6), facecolors=STATE_FILL)
    ax.set_yticks([top] + [top - 1 - i for i in range(len(state_rows))])
    ax.set_yticklabels(['phase'] + [name for name, _ in state_rows])
    ax.set_ylim(-0.6, top + 0.6)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    ax.tick_params(axis='y', length=0, labelcolor=SECONDARY_INK)
    ax.set_title('Event timeline', color=PRIMARY_INK, fontsize=10, loc='left')


def plot_episode(data, title, save_path=None, show=False, dpi=130, threshold_override=None,
                 video_name=''):
    t = data['t']
    tcp = data['tcp_poses']
    pwm_cmd = data['pwm_commanded']
    plan_ms = data['plan_times_ms']
    plan_idx = data['plan_step_indices'].astype(int)
    T, tcp_dims = tcp.shape
    threshold, threshold_note = resolve_threshold(data, threshold_override)
    stats = episode_stats(data, threshold)
    pressure = data.get('pressure_kpa')
    cmd_raw = data.get('suction_cmd_raw')
    has_cmd = cmd_raw is not None and np.isfinite(cmd_raw).any()

    left_rows = ['timeline', 'pressure'] + (['cmd'] if has_cmd else []) + ['tcp', 'pwm']
    heights = {'timeline': 0.75, 'pressure': 1.3, 'cmd': 0.8, 'tcp': 1.0, 'pwm': 1.0}
    fig = plt.figure(figsize=(16, 2.2 * sum(heights[r] for r in left_rows) + 1.0),
                     dpi=dpi, facecolor=SURFACE)
    gs = fig.add_gridspec(len(left_rows), 2, width_ratios=[2.2, 1],
                          height_ratios=[heights[r] for r in left_rows], hspace=0.45, wspace=0.18)
    fig.suptitle(title, color=PRIMARY_INK, fontsize=12, fontweight='bold')
    axes = {}
    for i, name in enumerate(left_rows):
        axes[name] = fig.add_subplot(gs[i, 0], sharex=axes.get('timeline'))

    # ── Event timeline ──────────────────────────────────────────────────────
    _timeline(axes['timeline'], data, stats)
    _mark_events(axes['timeline'], stats)
    key = [Patch(facecolor=PHASES[k][1], edgecolor=AXIS_COLOR, label=PHASES[k][0])
           for k in ((1, 0), (0, 1), (1, 1))]
    key += [Line2D([0], [0], color=MUTED_INK, linewidth=0.9, linestyle=style, label=name)
            for key_name, name, style in EVENT_LINES
            if key_name != 'suction_s' or 'suction_on' in data]
    _top_legend(axes['timeline'], key)

    # ── Vacuum pressure ─────────────────────────────────────────────────────
    ax = axes['pressure']
    _style_axes(ax)
    if pressure is not None and np.isfinite(pressure).any():
        ax.plot(t, pressure, color=CATEGORICAL[0], linewidth=2.0, zorder=3)
        ax.axhline(0.0, color=AXIS_COLOR, linewidth=0.8, zorder=1)
        if threshold is not None:
            ax.axhline(threshold, color=SECONDARY_INK, linewidth=1.0, linestyle='--', zorder=2)
            ax.text(t[-1], threshold, f' seal threshold {threshold:g} kPa', va='bottom', ha='right',
                    fontsize=7, color=SECONDARY_INK)
        _phase_boundaries(ax, data)
        _mark_events(ax, stats)
        ax.set_ylabel('pressure (kPa)')
    else:
        ax.text(0.5, 0.5, 'pressure not logged', ha='center', va='center', color=MUTED_INK,
                fontsize=9, transform=ax.transAxes)
    ax.set_title('Vacuum pressure (negative = vacuum)', color=PRIMARY_INK, fontsize=10, loc='left')

    # ── Raw suction command (baselines only) ────────────────────────────────
    if has_cmd:
        ax = axes['cmd']
        _style_axes(ax)
        ax.plot(t, cmd_raw, color=CATEGORICAL[0], linewidth=2.0, zorder=3)
        ax.axhline(0.5, color=SECONDARY_INK, linewidth=1.0, linestyle='--')
        ax.text(t[-1], 0.5, ' on threshold 0.5', va='bottom', ha='right', fontsize=7, color=SECONDARY_INK)
        _phase_boundaries(ax, data)
        _mark_events(ax, stats)
        ax.set_ylabel('raw command')
        ax.set_title('Policy suction command (pre-threshold)', color=PRIMARY_INK, fontsize=10, loc='left')

    # ── TCP position ────────────────────────────────────────────────────────
    ax = axes['tcp']
    _style_axes(ax)
    if not np.allclose(tcp, 0):
        for i in range(tcp_dims):
            ax.plot(t, tcp[:, i], color=CATEGORICAL[i], linewidth=2.0,
                    label=TCP_AXIS_LABELS[i] if i < len(TCP_AXIS_LABELS) else f'dim{i}')
        _top_legend(ax)
        _phase_boundaries(ax, data)
        ax.set_ylabel('position (m)')
    else:
        ax.text(0.5, 0.5, "'tcp' not in state_keys (not logged)", ha='center', va='center',
                color=MUTED_INK, fontsize=9, transform=ax.transAxes)
    ax.set_title('TCP position', color=PRIMARY_INK, fontsize=10, loc='left')

    # ── PWM commanded ───────────────────────────────────────────────────────
    ax = axes['pwm']
    _style_axes(ax)
    for i in range(pwm_cmd.shape[1]):
        ax.step(t, pwm_cmd[:, i], where='post', color=CATEGORICAL[i], linewidth=2.0,
                label=f'actuator {i + 1}')
    _top_legend(ax)
    _phase_boundaries(ax, data)
    ax.set_ylabel('PWM')
    ax.set_xlabel('time (s)')
    ax.set_title('Flowbot PWM commanded', color=PRIMARY_INK, fontsize=10, loc='left')
    for name in left_rows[:-1]:
        plt.setp(axes[name].get_xticklabels(), visible=False)

    # ── Right column: 3D path, latency, info ────────────────────────────────
    right = gs[:, 1].subgridspec(3, 1, height_ratios=[1.3, 0.8, 1.4], hspace=0.45)
    ax3d = fig.add_subplot(right[0], projection='3d')
    ax3d.set_facecolor(SURFACE)
    if not np.allclose(tcp, 0) and tcp_dims >= 3:
        sc = ax3d.scatter(tcp[:, 0], tcp[:, 1], tcp[:, 2], c=t, cmap=SEQUENTIAL_BLUE, s=8, linewidths=0)
        ax3d.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=GRID_COLOR, linewidth=0.6, zorder=0)
        cbar = fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.12)
        cbar.set_label('time (s)', color=SECONDARY_INK, fontsize=8)
        cbar.ax.tick_params(colors=MUTED_INK, labelsize=7)
        for setter, lab in ((ax3d.set_xlabel, 'x'), (ax3d.set_ylabel, 'y'), (ax3d.set_zlabel, 'z')):
            setter(lab, color=SECONDARY_INK, fontsize=8)
        ax3d.tick_params(colors=MUTED_INK, labelsize=7)
    else:
        ax3d.text2D(0.5, 0.5, 'not logged', ha='center', va='center', color=MUTED_INK,
                    fontsize=9, transform=ax3d.transAxes)
    ax3d.set_title('TCP path (color = time)', color=PRIMARY_INK, fontsize=10)

    ax = fig.add_subplot(right[1])
    _style_axes(ax)
    if plan_ms.size > 0:
        plan_t = t[np.clip(plan_idx, 0, max(T - 1, 0))] if T else plan_idx
        ax.plot(plan_t, plan_ms, color=CATEGORICAL[0], linewidth=2.0, marker='o', markersize=4)
        ax.axhline(float(np.mean(plan_ms)), color=MUTED_INK, linewidth=0.8, linestyle=':')
        ax.text(1.0, float(np.mean(plan_ms)), f' mean {np.mean(plan_ms):.0f} ms',
                transform=ax.get_yaxis_transform(), va='bottom', ha='right', fontsize=7, color=SECONDARY_INK)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('latency (ms)')
    ax.set_title('Inference latency per plan', color=PRIMARY_INK, fontsize=10, loc='left')

    ax = fig.add_subplot(right[2])
    ax.set_facecolor(SURFACE)
    ax.axis('off')
    ax.text(0.0, 1.0, '\n'.join(_info_lines(data, stats, threshold, threshold_note, video_name)),
            transform=ax.transAxes, va='top', ha='left', color=SECONDARY_INK, fontsize=8.5,
            family='monospace')

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, facecolor=SURFACE, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _fmt_s(x):
    return f'{x:.1f} s' if np.isfinite(x) else '-'


def _info_lines(data, stats, threshold, threshold_note, video_name):
    reported = bool(_scalar(data, 'outcome_reported', False))
    lines = [
        f"System:        {_scalar(data, 'system', '?')}",
        f"True object:   {_scalar(data, 'true_object', '') or '-'}",
        f"Predicted obj: {_scalar(data, 'predicted_object', '') or '-'}",
    ]
    if reported:
        success = bool(_scalar(data, 'success', False))
        lines.append(f"Outcome:       {'SUCCESS' if success else 'FAILURE'}")
        if not success:
            fails = [lab for key, lab in FAILURE_LABELS if bool(_scalar(data, key, False))]
            lines.append(f"  failure:     {', '.join(fails) if fails else '(none selected)'}")
            target = _scalar(data, 'wrong_object_target', '')
            if target:
                lines.append(f"  targeted:    {target}")
            note = _scalar(data, 'other_note', '')
            if note:
                lines.append(f"  note:        {note}")
    else:
        lines.append("Outcome:       not reported")
    plan_ms = data['plan_times_ms']
    lines += [
        '',
        f"Completion:    {_fmt_s(stats['completion_s'])}  (first release)",
        *([f"Suction cmd:   {_fmt_s(stats['suction_s'])}"] if 'suction_on' in data else []),
        f"Sealed:        {_fmt_s(stats['seal_s'])}",
        f"Seal thresh.:  {f'{threshold:g} kPa' if threshold is not None else '-'} ({threshold_note})",
        f"Duration:      {float(_scalar(data, 'duration_s', 0.0)):.1f} s, "
        f"{int(_scalar(data, 'total_steps', 0))} steps",
    ]
    if plan_ms.size:
        lines.append(f"Plans:         {plan_ms.size}, {plan_ms.mean():.0f} ± {plan_ms.std():.0f} ms "
                     f"(max {plan_ms.max():.0f})")
    ckpt = str(_scalar(data, 'checkpoint_path', ''))
    lines += ['', f"Checkpoint:    {'/'.join(Path(ckpt).parts[-3:]) if ckpt else '-'}"]
    if video_name:
        lines.append(f"Video:         {video_name}")
    return lines


def print_summary(rows):
    """Per-system table over all episodes in a directory."""
    by_system = defaultdict(list)
    for r in rows:
        by_system[r['system']].append(r)
    print('\nSummary per system')
    header = (f"{'system':<10} {'eps':>4} {'reported':>8} {'success':>8} "
              + ' '.join(f'{lab[:12]:>12}' for _, lab in FAILURE_LABELS)
              + f" {'complete s':>10} {'seal s':>7}")
    print(header)
    for system, rs in sorted(by_system.items()):
        rep = [r for r in rs if r['reported']]
        succ = sum(r['success'] for r in rep)
        rate = f'{100 * succ / len(rep):.0f}%' if rep else '-'
        fails = ' '.join(f"{sum(r[k] for r in rep):>12d}" for k, _ in FAILURE_LABELS)
        mean = lambda key: np.nanmean([r[key] for r in rs]) if any(np.isfinite(r[key]) for r in rs) else float('nan')
        print(f"{system:<10} {len(rs):>4} {len(rep):>8} {rate:>8} {fails} "
              f"{mean('completion_s'):>10.1f} {mean('seal_s'):>7.1f}")


def main():
    parser = argparse.ArgumentParser(description='Plot DeploymentLogger .npz episode logs')
    parser.add_argument('path', type=str, help='A single .npz file or a directory containing them')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Where to save PNGs (default: next to each .npz file)')
    parser.add_argument('--seal_threshold_kpa', type=float, default=-10.0,
                        help='Seal threshold (kPa, negative) to use instead of the logged one')
    parser.add_argument('--show', action='store_true', help='Also open each figure interactively')
    parser.add_argument('--dpi', type=int, default=130)
    args = parser.parse_args()
    if args.seal_threshold_kpa is not None and args.seal_threshold_kpa >= 0:
        parser.error('--seal_threshold_kpa must be negative (vacuum), e.g. -15')

    src = Path(args.path)
    npz_files = sorted(src.glob('*.npz')) if src.is_dir() else [src]
    if not npz_files:
        print(f"No .npz files found in {src}")
        return 1
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for npz_path in npz_files:
        print(f"Loading {npz_path.name} ...")
        data = load_episode(npz_path)
        threshold, note = resolve_threshold(data, args.seal_threshold_kpa)
        if 'ignored' in note:
            print(f"  seal threshold {note} -- pass --seal_threshold_kpa (e.g. -15) to show 'sealed'")
        video = npz_path.with_suffix('.mp4')
        save_path = (out_dir / f'{npz_path.stem}.png') if out_dir is not None else npz_path.with_suffix('.png')
        plot_episode(data, title=npz_path.stem, save_path=save_path, show=args.show, dpi=args.dpi,
                     threshold_override=args.seal_threshold_kpa,
                     video_name=video.name if video.exists() else '')
        stats = episode_stats(data, threshold)
        rows.append({'system': str(_scalar(data, 'system', '?')),
                     'reported': bool(_scalar(data, 'outcome_reported', False)),
                     'success': bool(_scalar(data, 'success', False)),
                     **{k: bool(_scalar(data, k, False)) for k, _ in FAILURE_LABELS},
                     'completion_s': stats['completion_s'], 'seal_s': stats['seal_s']})
    if len(rows) > 1 or src.is_dir():
        print_summary(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
