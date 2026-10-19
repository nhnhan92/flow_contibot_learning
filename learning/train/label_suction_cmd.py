#!/usr/bin/env python3
"""
label_suction_cmd.py — add an explicit suction-on command to an EXISTING
Flowbot dataset, for the decoupled-Flowbot baseline (suction as its own
learned action instead of coming implicitly with the flow).

No recollection: the command is derived from the recorded operation modes, so
the baseline is trained on exactly the same demonstrations as Flowbot.

Rule, per episode:
    t      = first flowbot -> UR5 transition (op_mode [0,1] -> [1,0]): the
             flowbot has finished bending, and the UR5 now brings the cup onto
             the object.
    onset  = t + offset + jitter_e   (jitter_e: random integer in [-jitter, +jitter],
                                      drawn once per episode, seeded)
    off    = first release frame (op_mode [1,1]) after the transition, else the
             episode end
    suction_cmd = 1 for onset <= frame < off, 0 elsewhere

The per-episode jitter emulates an operator who doesn't press the suction button
at exactly the same moment every time. Without it, the command is locked to an
op_mode switch the policy already predicts, so the extra action is trivial to
learn and the baseline tests nothing. Train one baseline per jitter level.

Written into the SAME zarr (existing arrays untouched):
    data/<name>          (T, 1) uint8 per-frame command
    meta/<name>_onset    (n_episodes,) int64 onset frame within each episode,
                         -1 for episodes without a flowbot->UR5 transition
<name> defaults to suction_cmd_k<offset>_j<jitter>, so several jitter levels can
sit side by side; point the training config's suction_key at the one to use.

Usage:
    # 1) look at the timing first (writes nothing): approach-segment lengths
    #    after the transition, per object -- choose offset/jitter from this
    python learning/train/label_suction_cmd.py --dataset <dataset.zarr> --stats

    # 2) check, then write, one label set per jitter level
    python learning/train/label_suction_cmd.py --dataset <dataset.zarr> --offset 3 --jitter 2 --dry_run
    python learning/train/label_suction_cmd.py --dataset <dataset.zarr> --offset 3 --jitter 0
    python learning/train/label_suction_cmd.py --dataset <dataset.zarr> --offset 3 --jitter 2
"""
import argparse
from collections import defaultdict

import numpy as np
import zarr

FB, UR5, RELEASE = (0, 1), (1, 0), (1, 1)


def episode_ranges(root):
    ends = np.asarray(root['meta/episode_ends'][:], dtype=np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return list(zip(starts.tolist(), ends.tolist()))


def episode_objects(root, n_episodes):
    meta = root['meta']
    if 'episode_object_id' in meta and 'object_names' in meta:
        names = [str(n) for n in meta['object_names'][:]]
        ids = np.asarray(meta['episode_object_id'][:])
        return [names[int(i)] for i in ids]
    return ['all'] * n_episodes


def analyze_episode(op):
    """op: (L, 2) op_modes of one episode. Returns None if there's no
    flowbot->UR5 transition, else frame indices (episode-local):
        transition   first UR5 frame after a flowbot frame
        approach_len UR5 frames from the transition until the mode next changes
        release      first release frame after the transition, or None"""
    keys = [tuple(int(v) for v in row) for row in op]
    L = len(keys)
    t = next((i for i in range(1, L) if keys[i - 1] == FB and keys[i] == UR5), None)
    if t is None:
        return None
    a_end = t
    while a_end < L and keys[a_end] == UR5:
        a_end += 1
    release = next((i for i in range(t, L) if keys[i] == RELEASE), None)
    return {'transition': t, 'approach_len': a_end - t, 'release': release, 'length': L}


def _write_array(group, name, values, attrs=None):
    """Create/replace an array (zarr v3 create_array, or v2 create_dataset)."""
    if name in group:
        del group[name]
    make = group.create_array if hasattr(group, 'create_array') else group.create_dataset
    arr = make(name, shape=values.shape, dtype=values.dtype)
    arr[:] = values
    if attrs:
        arr.attrs.update(attrs)
    return arr


def _range_str(vals):
    if not vals:
        return '-'
    return f'{min(vals)}/{int(np.median(vals))}/{max(vals)}'


def print_stats(root):
    ranges = episode_ranges(root)
    op_all = np.asarray(root['data/operation_mode'][:])
    objects = episode_objects(root, len(ranges))
    per_obj = defaultdict(lambda: {'approach': [], 'to_release': [], 'missing': [], 'no_release': []})
    for ep, (s, e) in enumerate(ranges):
        info = analyze_episode(op_all[s:e])
        g = per_obj[objects[ep]]
        if info is None:
            g['missing'].append(ep)
            continue
        g['approach'].append(info['approach_len'])
        if info['release'] is None:
            g['no_release'].append(ep)
        else:
            g['to_release'].append(info['release'] - info['transition'])

    print(f"{'object':<10} {'eps':>4}  {'UR5 segment after transition':>30}  "
          f"{'transition->release':>20}  (min/median/max frames)")
    all_approach = []
    for obj, g in sorted(per_obj.items()):
        n = len(g['approach']) + len(g['missing'])
        print(f"{obj:<10} {n:>4}  {_range_str(g['approach']):>30}  {_range_str(g['to_release']):>20}")
        all_approach += g['approach']
    for obj, g in sorted(per_obj.items()):
        if g['missing']:
            print(f"  {obj}: no flowbot->UR5 transition in episode(s) {g['missing']} -- "
                  f"no suction label possible; consider exclude_episodes")
        if g['no_release']:
            print(f"  {obj}: no release after the transition in episode(s) {g['no_release']} -- "
                  f"suction would stay on until the episode end")
    if all_approach:
        print(f"\n'UR5 segment' = UR5 frames from the transition until the op_mode next changes. "
              f"If the approach and the transport are both UR5 with no mode switch between them, "
              f"it includes the transport too -- then choose offset from how long the approach "
              f"alone takes (e.g. scrub a few episodes in visualize_dataset.py).")
        print(f"Shortest UR5 segment: {min(all_approach)} frames, so offset + jitter must be "
              f"<= {min(all_approach) - 1} for every onset to fall inside it.")


def compute_labels(root, offset, jitter, seed):
    ranges = episode_ranges(root)
    op_all = np.asarray(root['data/operation_mode'][:])
    T = ranges[-1][1] if ranges else 0
    cmd = np.zeros((T, 1), dtype=np.uint8)
    onsets = np.full(len(ranges), -1, dtype=np.int64)
    rng = np.random.RandomState(seed)
    report = {'missing': [], 'clamped': [], 'after_approach': [], 'no_release': []}

    for ep, (s, e) in enumerate(ranges):
        j = int(rng.randint(-jitter, jitter + 1)) if jitter > 0 else 0   # drawn for every episode -> stable
        info = analyze_episode(op_all[s:e])
        if info is None:
            report['missing'].append(ep)
            continue
        off = info['release'] if info['release'] is not None else info['length']
        if info['release'] is None:
            report['no_release'].append(ep)
        onset = info['transition'] + offset + j
        if not 0 <= onset < off:
            onset = int(np.clip(onset, 0, off - 1))
            report['clamped'].append(ep)
        if onset >= info['transition'] + info['approach_len']:
            report['after_approach'].append(ep)
        cmd[s + onset:s + off, 0] = 1
        onsets[ep] = onset
    return cmd, onsets, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='Path to the existing dataset.zarr')
    parser.add_argument('--stats', action='store_true', help='Print transition/approach timing per object and exit')
    parser.add_argument('--offset', type=int, default=None, help='Frames after the flowbot->UR5 transition')
    parser.add_argument('--jitter', type=int, default=0, help='Per-episode random onset shift in [-jitter, +jitter]')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default=None,
                        help='Array name under data/ (default suction_cmd_k<offset>_j<jitter>)')
    parser.add_argument('--dry_run', action='store_true', help='Compute and report, write nothing')
    args = parser.parse_args()

    if args.stats:
        print_stats(zarr.open(args.dataset, mode='r'))
        return
    if args.offset is None:
        parser.error('--offset is required (run --stats first to choose it)')
    if args.jitter < 0:
        parser.error('--jitter must be >= 0')

    name = args.name or f'suction_cmd_k{args.offset}_j{args.jitter}'
    root = zarr.open(args.dataset, mode='r' if args.dry_run else 'a')
    cmd, onsets, report = compute_labels(root, args.offset, args.jitter, args.seed)

    n_ep = len(onsets)
    print(f"{name}: offset={args.offset}, jitter=+/-{args.jitter}, seed={args.seed}")
    print(f"  labeled {int((onsets >= 0).sum())}/{n_ep} episodes, suction on in "
          f"{int(cmd.sum())}/{len(cmd)} frames")
    if report['missing']:
        print(f"  no flowbot->UR5 transition (left all-off): episodes {report['missing']}")
    if report['no_release']:
        print(f"  no release after the transition (on until episode end): episodes {report['no_release']}")
    if report['clamped']:
        print(f"  onset clamped into [episode start, release): episodes {report['clamped']}")
    if report['after_approach']:
        print(f"  onset after the UR5 segment ended (suction very late): "
              f"{len(report['after_approach'])} episodes {report['after_approach'][:20]}"
              f"{' ...' if len(report['after_approach']) > 20 else ''}")

    if args.dry_run:
        print('--dry_run: nothing written.')
        return
    params = {'offset': args.offset, 'jitter': args.jitter, 'seed': args.seed}
    _write_array(root['data'], name, cmd, attrs=params)
    _write_array(root['meta'], f'{name}_onset', onsets, attrs=params)
    print(f"Wrote data/{name} and meta/{name}_onset -- set suction_key: {name} in the decoupled config.")


if __name__ == '__main__':
    main()
