#!/usr/bin/env python3
"""
Plot deployment logs saved by DeploymentLogger (see deploy_flowbot_w_policy.py).

Reads one .npz episode log (or every .npz in a directory) and renders a
multi-panel figure: TCP position over time, the TCP path in 3D, PWM
actual-vs-commanded per actuator, the raw (pre-threshold) op_mode signal,
and per-plan DDIM inference latency.

Usage:
    python plot_deployment_log.py path/to/episode_000_20260228_120000.npz
    python plot_deployment_log.py path/to/log_dir/ --output_dir path/to/figs
    python plot_deployment_log.py path/to/log_dir/ --show
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Palette (fixed categorical order + chart chrome; see dataviz skill) ────────
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GRID_COLOR   = '#e1e0d9'
AXIS_COLOR   = '#c3c2b7'
MUTED_INK    = '#898781'
PRIMARY_INK  = '#0b0b0b'
SECONDARY_INK = '#52514e'
SURFACE      = '#fcfcfb'

# Sequential blue ramp (100 -> 700), used only for magnitude/progression
# encodings (e.g. "time along the trajectory") -- never a rainbow colormap.
_BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
              '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list('seq_blue', _BLUE_RAMP)

TCP_AXIS_LABELS = ['x', 'y', 'z', 'rx', 'ry', 'rz']


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)


def load_episode(npz_path):
    """Load one DeploymentLogger .npz into a plain dict (decodes 0-d arrays)."""
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    out['total_steps'] = int(out['total_steps'])
    out['duration_s'] = float(out['duration_s'])
    out['checkpoint_path'] = str(out['checkpoint_path'])
    return out


def plot_episode(data, title, save_path=None, show=False, dpi=130):
    tcp = data['tcp_poses']            # (T, tcp_dims)
    pwm_actual = data['pwm_actual']    # (T, 3)
    pwm_cmd = data['pwm_commanded']    # (T, 3)
    actions = data['executed_actions']  # (T, tcp_dims + 3 + 2)
    plan_ms = data['plan_times_ms']     # (N_plans,)
    plan_idx = data['plan_step_indices']  # (N_plans,)

    T, tcp_dims = tcp.shape
    steps = np.arange(T)
    tcp_is_logged = not np.allclose(tcp, 0)
    op_mode_raw = actions[:, tcp_dims + 3: tcp_dims + 5] if actions.shape[1] >= tcp_dims + 5 else None

    fig, axes = plt.subplot_mosaic(
        [['tcp', 'traj3d', 'opmode'],
         ['pwm', 'latency', 'info']],
        figsize=(15, 8), dpi=dpi, facecolor=SURFACE,
    )
    fig.suptitle(title, color=PRIMARY_INK, fontsize=12, fontweight='bold')

    # ── TCP position vs step ────────────────────────────────────────────────
    ax = axes['tcp']
    _style_axes(ax)
    if tcp_is_logged:
        for i in range(tcp_dims):
            ax.plot(steps, tcp[:, i], color=CATEGORICAL[i], linewidth=1.5,
                     label=TCP_AXIS_LABELS[i] if i < len(TCP_AXIS_LABELS) else f'dim{i}')
        ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc='best')
        ax.set_ylabel('TCP position (m)')
    else:
        ax.text(0.5, 0.5, "'tcp' not in state_keys\n(not logged)", ha='center', va='center',
                 color=MUTED_INK, fontsize=9, transform=ax.transAxes)
    ax.set_xlabel('step')
    ax.set_title('TCP position', color=PRIMARY_INK, fontsize=10)

    # ── 3D TCP trajectory, colored by progression through the episode ──────
    ax3d = axes['traj3d']
    ax3d.remove()
    ax3d = fig.add_subplot(2, 3, 2, projection='3d')
    ax3d.set_facecolor(SURFACE)
    if tcp_is_logged and tcp_dims >= 3:
        sc = ax3d.scatter(tcp[:, 0], tcp[:, 1], tcp[:, 2], c=steps, cmap=SEQUENTIAL_BLUE,
                            s=6, linewidths=0)
        ax3d.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=GRID_COLOR, linewidth=0.6, zorder=0)
        cbar = fig.colorbar(sc, ax=ax3d, shrink=0.6, pad=0.1)
        cbar.set_label('step', color=SECONDARY_INK, fontsize=8)
        cbar.ax.tick_params(colors=MUTED_INK, labelsize=7)
        ax3d.set_xlabel('x', color=SECONDARY_INK, fontsize=8)
        ax3d.set_ylabel('y', color=SECONDARY_INK, fontsize=8)
        ax3d.set_zlabel('z', color=SECONDARY_INK, fontsize=8)
        ax3d.tick_params(colors=MUTED_INK, labelsize=7)
    else:
        ax3d.text2D(0.5, 0.5, "not logged", ha='center', va='center',
                     color=MUTED_INK, fontsize=9, transform=ax3d.transAxes)
    ax3d.set_title('TCP trajectory (color = step)', color=PRIMARY_INK, fontsize=10)

    # ── Raw op_mode signal (pre-threshold model output) ─────────────────────
    ax = axes['opmode']
    _style_axes(ax)
    if op_mode_raw is not None:
        ax.plot(steps, op_mode_raw[:, 0], color=CATEGORICAL[0], linewidth=1.3, label='arm active')
        ax.plot(steps, op_mode_raw[:, 1], color=CATEGORICAL[1], linewidth=1.3, label='flowbot active')
        ax.axhline(0.5, color=AXIS_COLOR, linewidth=0.8, linestyle=':')
        ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc='best')
        ax.set_ylabel('raw op_mode (pre-threshold)')
    else:
        ax.text(0.5, 0.5, 'not available', ha='center', va='center',
                 color=MUTED_INK, fontsize=9, transform=ax.transAxes)
    ax.set_xlabel('step')
    ax.set_title('Predicted op_mode', color=PRIMARY_INK, fontsize=10)

    # ── PWM actual vs commanded, per actuator ────────────────────────────────
    ax = axes['pwm']
    _style_axes(ax)
    n_act = pwm_actual.shape[1]
    for i in range(n_act):
        ax.plot(steps, pwm_actual[:, i], color=CATEGORICAL[i], linewidth=1.5,
                 linestyle='-', label=f'actuator {i} actual')
        ax.plot(steps, pwm_cmd[:, i], color=CATEGORICAL[i], linewidth=1.2,
                 linestyle='--', alpha=0.75, label=f'actuator {i} commanded')
    ax.legend(frameon=False, fontsize=7, labelcolor=SECONDARY_INK, ncol=2, loc='best')
    ax.set_xlabel('step')
    ax.set_ylabel('PWM')
    ax.set_title('Flowbot PWM: actual (solid) vs commanded (dashed)', color=PRIMARY_INK, fontsize=10)

    # ── DDIM planning latency ────────────────────────────────────────────────
    ax = axes['latency']
    _style_axes(ax)
    if plan_ms.size > 0:
        ax.plot(plan_idx, plan_ms, color=CATEGORICAL[0], linewidth=1.3, marker='o', markersize=3)
        ax.axhline(float(np.mean(plan_ms)), color=MUTED_INK, linewidth=0.8, linestyle=':',
                    label=f'mean {np.mean(plan_ms):.1f} ms')
        ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK, loc='best')
    ax.set_xlabel('step (plan triggered)')
    ax.set_ylabel('latency (ms)')
    ax.set_title('DDIM inference latency', color=PRIMARY_INK, fontsize=10)

    # ── Episode info panel (text, no axes chrome) ───────────────────────────
    ax = axes['info']
    ax.set_facecolor(SURFACE)
    ax.axis('off')
    ckpt_name = Path(data['checkpoint_path']).name if data['checkpoint_path'] else 'unknown'
    lines = [
        f"Total steps:  {data['total_steps']}",
        f"Duration:     {data['duration_s']:.1f} s",
        f"Plans issued: {plan_ms.size}",
    ]
    if plan_ms.size > 0:
        lines.append(f"Plan latency: {plan_ms.mean():.1f} ± {plan_ms.std():.1f} ms "
                      f"(max {plan_ms.max():.1f})")
    lines.append(f"Checkpoint:   {ckpt_name}")
    ax.text(0.02, 0.95, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
             color=SECONDARY_INK, fontsize=9, family='monospace')

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, facecolor=SURFACE)
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot DeploymentLogger .npz episode logs')
    parser.add_argument('path', type=str, help='A single .npz file or a directory containing them')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                         help='Where to save PNGs (default: next to each .npz file)')
    parser.add_argument('--show', action='store_true', help='Also open each figure interactively')
    parser.add_argument('--dpi', type=int, default=130)
    args = parser.parse_args()

    src = Path(args.path)
    if src.is_dir():
        npz_files = sorted(src.glob('*.npz'))
        if not npz_files:
            print(f"No .npz files found in {src}")
            return 1
    else:
        npz_files = [src]

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in npz_files:
        print(f"Loading {npz_path.name} ...")
        data = load_episode(npz_path)
        save_path = (out_dir / f'{npz_path.stem}.png') if out_dir is not None \
            else npz_path.with_suffix('.png')
        plot_episode(data, title=npz_path.stem, save_path=save_path, show=args.show, dpi=args.dpi)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
