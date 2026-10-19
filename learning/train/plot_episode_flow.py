#!/usr/bin/env python3
"""
plot_episode_flow.py — flow sensor readings over an episode, with the
operation phases (UR5 / Flowbot / Release) shaded, to see what suction-cup
contact looks like in the flow signal: when the cup seals on the object, the
flow through the line changes. Meant for picking a seal-detection rule for
the suction-trigger relabel (decoupled-Flowbot baseline).

Per episode, three rows sharing the time axis:
    1. flowrate (data/flowrate, one line per module)
    2. d(flowrate)/dt  -- a seal shows up as a sharp step here
    3. PWM commands (data/pwm_signals)
plus, with --thumbs N, a strip of N camera frames spread over the Flowbot
phase, each marked on the plots with a dashed line -- to check visually which
frame is the actual contact.

Usage:
    # specific episodes, interactive window
    python learning/train/plot_episode_flow.py --dataset <dataset.zarr> --episodes 0 1 2

    # first 5 episodes of one object, saved as PNGs, with wrist-camera thumbnails
    python learning/train/plot_episode_flow.py --dataset <dataset.zarr> \\
        --object white2 --n 5 --thumbs 6 --out_dir flow_plots
"""
import argparse
import os

import matplotlib
import numpy as np
import zarr

PHASES = {  # op_mode [ur5_active, flowbot_active] -> (label, colour)
    (1, 0): ('UR5', '#cfe3f7'),
    (0, 1): ('Flowbot', '#fde2c4'),
    (1, 1): ('Release', '#f6c6c6'),
    (0, 0): ('Idle', '#e6e6e6'),
}


def episode_bounds(root, ep):
    ends = root['meta/episode_ends'][:]
    return (0 if ep == 0 else int(ends[ep - 1])), int(ends[ep])


def phase_segments(op_mode):
    """Contiguous runs of the same op_mode: [(start, end_exclusive, (u, f)), ...]."""
    keys = [tuple(int(v) for v in row) for row in op_mode]
    segs, s = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[s]:
            segs.append((s, i, keys[s]))
            s = i
    return segs


def resolve_episodes(root, args):
    n_episodes = root['meta/episode_ends'].shape[0]   # zarr v3 arrays have no len()
    if args.episodes:
        return args.episodes
    if args.object:
        meta = root['meta']
        if 'episode_object_id' not in meta or 'object_names' not in meta:
            raise SystemExit("--object needs a labeled dataset (meta/episode_object_id, meta/object_names)")
        names = [str(n) for n in meta['object_names'][:]]
        if args.object not in names:
            raise SystemExit(f"Unknown object {args.object!r}; available: {names}")
        ids = np.asarray(meta['episode_object_id'][:])
        eps = [int(e) for e in np.where(ids == names.index(args.object))[0]]
        return eps[:args.n]
    return list(range(min(args.n, n_episodes)))


def object_name(root, ep):
    meta = root['meta']
    if 'episode_object_id' in meta and 'object_names' in meta:
        return str(meta['object_names'][int(meta['episode_object_id'][ep])])
    return None


def plot_episode(root, ep, args, plt):
    start, end = episode_bounds(root, ep)
    data = root['data']
    flow = np.asarray(data['flowrate'][start:end], dtype=float)
    pwm = np.asarray(data['pwm_signals'][start:end], dtype=float)
    op = np.asarray(data['operation_mode'][start:end])
    frames = np.arange(end - start)
    if 'timestamp' in data and not args.frames_axis:
        t = np.asarray(data['timestamp'][start:end], dtype=float)
        x, xlabel = t - t[0], 'Time (s)'
    else:
        x, xlabel = frames.astype(float), 'Frame'
    dflow = np.gradient(flow, x, axis=0) if len(x) > 1 else np.zeros_like(flow)
    segs = phase_segments(op)

    # Thumbnails: evenly spread over the Flowbot phase (+ one frame before it).
    thumb_frames = []
    cam_key = args.camera if args.camera in data else ('camera_1' if 'camera_1' in data else 'camera_0')
    if args.thumbs > 0:
        fb = np.where((op[:, 0] == 0) & (op[:, 1] == 1))[0]
        lo, hi = (max(fb[0] - 1, 0), fb[-1]) if len(fb) else (0, len(frames) - 1)
        thumb_frames = sorted(set(np.linspace(lo, hi, args.thumbs).round().astype(int).tolist()))

    n_rows = 3 + (1 if thumb_frames else 0)
    fig = plt.figure(figsize=(12, 7.5 + (2.2 if thumb_frames else 0)))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[1, 1, 1] + ([1.1] if thumb_frames else []), hspace=0.35)
    axes = [fig.add_subplot(gs[0])]
    axes += [fig.add_subplot(gs[i], sharex=axes[0]) for i in (1, 2)]

    for i in range(flow.shape[1]):
        axes[0].plot(x, flow[:, i], label=f'Module {i + 1}', linewidth=1.3)
        axes[1].plot(x, dflow[:, i], linewidth=1.1)
    for i in range(pwm.shape[1]):
        axes[2].plot(x, pwm[:, i], label=f'PWM {i + 1}', linewidth=1.3)
    axes[0].set_ylabel('Flowrate')
    axes[1].set_ylabel('d Flowrate / dt')
    axes[2].set_ylabel('PWM')
    axes[2].set_xlabel(xlabel)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[2].legend(loc='upper right', fontsize=8)

    seen = set()
    for s, e, key in segs:
        label, colour = PHASES.get(key, (str(key), '#dddddd'))
        x_end = x[e] if e < len(x) else x[-1]
        for ax in axes:
            ax.axvspan(x[s], x_end, color=colour, alpha=0.6, zorder=0,
                       label=label if (ax is axes[1] and label not in seen) else None)
        seen.add(label)
    axes[1].legend(loc='upper right', fontsize=8, title='Phase')
    for f in thumb_frames:
        for ax in axes:
            ax.axvline(x[f], color='k', linestyle='--', linewidth=0.8, alpha=0.6)

    name = object_name(root, ep)
    fig.suptitle(f'Episode {ep}' + (f' ({name})' if name else '') + f'  —  {end - start} frames')

    if thumb_frames:
        sub = gs[3].subgridspec(1, len(thumb_frames), wspace=0.05)
        for j, f in enumerate(thumb_frames):
            ax = fig.add_subplot(sub[0, j])
            ax.imshow(np.asarray(data[cam_key][start + f]))
            key = tuple(int(v) for v in op[f])
            ax.set_title(f'frame {f} ({PHASES.get(key, (str(key),))[0]})', fontsize=8)
            ax.axis('off')

    # Text summary: phase spans and flow range inside each Flowbot segment.
    print(f'\nEpisode {ep}' + (f' ({name})' if name else ''))
    for s, e, key in segs:
        label = PHASES.get(key, (str(key),))[0]
        extra = ''
        if key == (0, 1):
            extra = '  flow min/max per module: ' + ', '.join(
                f'{flow[s:e, i].min():.3f}/{flow[s:e, i].max():.3f}' for i in range(flow.shape[1]))
        print(f'  {label:<8} frames {s:4d}-{e - 1:4d}{extra}')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='Path to dataset.zarr')
    parser.add_argument('--episodes', type=int, nargs='+', default=None, help='Episode indices to plot')
    parser.add_argument('--object', type=str, default=None, help='Plot episodes of this object (labeled datasets)')
    parser.add_argument('--n', type=int, default=3, help='How many episodes with --object / by default (default 3)')
    parser.add_argument('--thumbs', type=int, default=0, help='Camera thumbnails over the Flowbot phase (0 = none)')
    parser.add_argument('--camera', type=str, default='camera_1', help='Camera for thumbnails (default wrist, camera_1)')
    parser.add_argument('--frames_axis', action='store_true', help='Use frame index on the x axis instead of time')
    parser.add_argument('--out_dir', type=str, default=None, help='Save PNGs here instead of showing windows')
    args = parser.parse_args()

    if args.out_dir:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = zarr.open(args.dataset, mode='r')
    if 'flowrate' not in root['data']:
        raise SystemExit(f"{args.dataset} has no data/flowrate -- collected before flow recording was added.")

    for ep in resolve_episodes(root, args):
        fig = plot_episode(root, ep, args, plt)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            name = object_name(root, ep)
            path = os.path.join(args.out_dir, f'episode_{ep:04d}' + (f'_{name}' if name else '') + '.png')
            fig.savefig(path, dpi=130, bbox_inches='tight')
            plt.close(fig)
            print(f'  saved {path}')
    if not args.out_dir:
        plt.show()


if __name__ == '__main__':
    main()
