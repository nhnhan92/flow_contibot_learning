"""
deployment_logger.py — shared episode logger for deploy_flowbot_w_policy.py and
deploy_baseline_w_policy.py, so Flowbot and baseline runs are saved in ONE
identical format and a single analysis script works for every system.

Per episode, in log_dir:
    episode_<idx>_<timestamp>.npz   per-step / per-plan arrays + episode outcome
    episode_<idx>_<timestamp>.mp4   wrist-camera video, paced to real time
                                    (only if a wrist camera is available)
    episodes_summary.csv            one row appended per episode (outcome,
                                    times) -- the comparison table

Per step (T = executed steps):
    timestamps        (T,)       wall-clock time of each executed step
    tcp_poses         (T, d)     TCP pose from the state vector
    pwm_actual        (T, 3)     PWM from the state vector -- the PREVIOUS step's
                                 command, not a sensor measurement
    executed_actions  (T, A)     full denormalized action executed
    pwm_commanded     (T, 3)     integer PWM sent after clamping
    op_mode           (T, 2)     executed op_mode [ur5_active, flowbot_active]
    suction_cmd_raw   (T,)       policy's raw suction output (NaN when the
                                 policy has no suction action, e.g. Flowbot)
    suction_on        (T,)       suction state ('sf' sent and not yet released)
    pressure_kpa      (T,)       vacuum pressure (kPa, negative = vacuum)
    sealed            (T,)       pressure_kpa <= seal_threshold_kpa (all False if
                                 no threshold was given -- recompute later)
Per plan (N = inference calls):
    predicted_horizons (N, P, A), plan_times_ms (N,), plan_step_indices (N,)
Per episode:
    system, checkpoint_path, true_object, predicted_object, total_steps,
    duration_s, completion_time_s (first release), time_to_suction_s,
    time_to_seal_s, seal_threshold_kpa, video_path, and the operator's report:
    outcome_reported, success, wrong_object, wrong_object_target,
    premature_suction, late_suction_no_seal, dropped_or_placing_error,
    collision, timeout, other, other_note
Missing values are NaN (numbers) or '' (text), so files load without pickle.
"""
import csv
import datetime
import time
from pathlib import Path

import cv2
import numpy as np

FAILURE_KEYS = [
    ('wrong_object', 'Wrong object targeted?'),
    ('premature_suction', 'Premature suction?'),
    ('late_suction_no_seal', 'Late suction / no seal?'),
    ('dropped_or_placing_error', 'Dropped during transport or placing error?'),
    ('collision', 'Collision?'),
    ('timeout', 'Timeout?'),
    ('other', 'Other failure?'),
]
SUMMARY_FIELDS = (['episode', 'file', 'system', 'checkpoint', 'true_object', 'predicted_object',
                   'outcome_reported', 'success']
                  + [k for k, _ in FAILURE_KEYS]
                  + ['wrong_object_target', 'other_note', 'completion_time_s', 'duration_s',
                     'time_to_suction_s', 'time_to_seal_s', 'total_steps'])


def _ask_yes_no(prompt):
    while True:
        ans = input(f"  {prompt} [y/n]: ").strip().lower()
        if ans in ('y', 'yes', 'n', 'no'):
            return ans.startswith('y')


def _ask_choice(prompt, names):
    """Pick one of names (by number or exact name); free text if names is empty."""
    if not names:
        return input(f"  {prompt}: ").strip()
    options = '  '.join(f'[{i}] {n}' for i, n in enumerate(names))
    while True:
        ans = input(f"  {prompt} {options}: ").strip()
        if ans.isdigit() and int(ans) < len(names):
            return names[int(ans)]
        if ans in names:
            return ans


def _first_time(mask, timestamps, t0):
    idx = np.flatnonzero(mask)
    return float(timestamps[idx[0]] - t0) if len(idx) and t0 is not None else float('nan')


class DeploymentLogger:
    """See module docstring. Call order per episode:
        start_episode(idx, t0) -> log_plan()/log_step() ... -> ask_outcome() -> save(idx, ...)"""

    def __init__(self, log_dir: str, checkpoint_path: str, tcp_dims: int = 3,
                 state_keys=('tcp', 'pwm', 'op_mode'), num_objects: int = None,
                 system: str = 'flowbot', seal_threshold_kpa: float = None,
                 record_wrist_video: bool = True, video_fps: float = 10.0):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = str(checkpoint_path)
        self.tcp_dims = tcp_dims
        self.system = system
        self.seal_threshold_kpa = seal_threshold_kpa
        self.record_wrist_video = record_wrist_video
        self.video_fps = video_fps
        # Locate 'tcp'/'pwm' within the concatenated state vector (see
        # RobotDeployment.state_keys / dataset.py's identical scheme) so
        # log_step can pull them out for the tcp_poses/pwm_actual log
        # columns regardless of composition/order -- state_raw is no longer
        # guaranteed to be [tcp, pwm, ...] now that state_keys is configurable.
        # num_objects is required (and only needed) if 'object_id' is in
        # state_keys, to correctly advance past its (dataset-specific) width.
        _widths = {'pwm': 3, 'flowrate': 3, 'op_mode': 2, 'object_id': num_objects}
        self._tcp_offset = self._pwm_offset = None
        offset = 0
        for key in state_keys:
            width = tcp_dims if key == 'tcp' else _widths[key]
            if width is None:
                raise ValueError(
                    f"state_keys includes 'object_id' but num_objects wasn't given to "
                    f"DeploymentLogger -- can't compute its width."
                )
            if key == 'tcp':
                self._tcp_offset = offset
            elif key == 'pwm':
                self._pwm_offset = offset
            offset += width
        if self._tcp_offset is None:
            print("[DeploymentLogger] 'tcp' not in state_keys -- tcp_poses log will be all zeros.")
        if self._pwm_offset is None:
            print("[DeploymentLogger] 'pwm' not in state_keys -- pwm_actual log will be all zeros.")
        if seal_threshold_kpa is None:
            print("[DeploymentLogger] no seal threshold given -- pressure is logged, 'sealed' left "
                  "False (recompute later from pressure_kpa).")
        self._video = None
        self._reset()

    # ── per-episode lifecycle ─────────────────────────────────────────────────

    def _reset(self):
        self._timestamps = []
        self._tcp_poses = []
        self._pwm_actual = []
        self._executed_actions = []
        self._pwm_commanded = []
        self._op_mode = []
        self._suction_cmd_raw = []
        self._suction_on = []
        self._pressure_kpa = []
        self._predicted_horizons = []
        self._plan_times_ms = []
        self._plan_step_indices = []
        self._episode_idx = None
        self._episode_t0 = None
        self._tag = None
        self._predicted_object = ''
        self._outcome = None
        self._close_video()

    def start_episode(self, episode_idx: int, t0: float, predicted_object: str = ''):
        """Call once when the episode starts (before the first plan)."""
        self._reset()
        self._episode_idx = episode_idx
        self._episode_t0 = t0
        self._tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._predicted_object = predicted_object or ''

    def _file_stem(self, episode_idx):
        if self._tag is None:   # start_episode() wasn't called -- name by first use
            self._tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        idx = f'{episode_idx:03d}' if isinstance(episode_idx, int) else 'xxx'
        return f"episode_{idx}_{self._tag}"

    def log_plan(self, step_idx: int, predicted_actions: np.ndarray, plan_time_s: float):
        """Call once per DDIM inference (before executing the action horizon)."""
        self._predicted_horizons.append(predicted_actions.copy())
        self._plan_times_ms.append(plan_time_s * 1000.0)
        self._plan_step_indices.append(step_idx)

    def log_step(self, state_raw: np.ndarray, action: np.ndarray, pwm_commanded: np.ndarray,
                 op_mode=None, suction_cmd_raw: float = float('nan'), suction_on: bool = False,
                 pressure_kpa: float = float('nan'), wrist_frame: np.ndarray = None):
        """Call once per executed step (after _update_obs_buffer)."""
        d = self.tcp_dims
        now = time.time()
        self._timestamps.append(now)
        tcp = state_raw[self._tcp_offset:self._tcp_offset + d].copy() \
            if self._tcp_offset is not None else np.zeros(d, dtype=np.float32)
        pwm = state_raw[self._pwm_offset:self._pwm_offset + 3].copy() \
            if self._pwm_offset is not None else np.zeros(3, dtype=np.float32)
        self._tcp_poses.append(tcp)
        self._pwm_actual.append(pwm)
        self._executed_actions.append(action.copy())
        self._pwm_commanded.append(np.asarray(pwm_commanded).copy())
        self._op_mode.append(np.asarray(op_mode if op_mode is not None else [0, 0], dtype=np.int8))
        self._suction_cmd_raw.append(float(suction_cmd_raw))
        self._suction_on.append(bool(suction_on))
        self._pressure_kpa.append(float(pressure_kpa))
        if self.record_wrist_video and wrist_frame is not None:
            self._write_video_frame(wrist_frame, now)

    # ── wrist video (paced to real time by repeating frames) ──────────────────

    def _write_video_frame(self, frame_rgb, t):
        if self._video is None:
            h, w = frame_rgb.shape[:2]
            path = self.log_dir / f"{self._file_stem(self._episode_idx)}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (w, h))
            if not writer.isOpened():
                print(f"[DeploymentLogger] could not open video writer for {path} -- video disabled.")
                self.record_wrist_video = False
                return
            self._video = {'writer': writer, 'path': path, 't0': t, 'written': 0}
        v = self._video
        # Steps aren't evenly spaced (flowbot steps are slower), so pace by wall
        # clock: write however many frames bring the video up to the elapsed
        # time (repeat a frame across a slow step, skip it within a fast one).
        target = int(round((t - v['t0']) * self.video_fps)) + 1
        n = target - v['written']
        if n > 0:
            bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
            for _ in range(n):
                v['writer'].write(bgr)
            v['written'] = target

    def _close_video(self):
        if getattr(self, '_video', None) is not None:
            self._video['writer'].release()
            self._video = None

    # ── operator report ───────────────────────────────────────────────────────

    def ask_outcome(self, object_names=()):
        """Interactive end-of-episode report. Ctrl+C / EOF skips it (recorded as
        outcome_reported=False)."""
        names = list(object_names or [])
        out = {'outcome_reported': False, 'true_object': '', 'success': False,
               'wrong_object_target': '', 'other_note': '', **{k: False for k, _ in FAILURE_KEYS}}
        print("\n--- Episode report ---")
        try:
            out['true_object'] = _ask_choice('True object?', names)
            out['success'] = _ask_yes_no('Task successful?')
            if not out['success']:
                for key, prompt in FAILURE_KEYS:
                    out[key] = _ask_yes_no(prompt)
                    if key == 'wrong_object' and out[key]:
                        out['wrong_object_target'] = _ask_choice('Which object was targeted?', names)
                    if key == 'other' and out[key]:
                        out['other_note'] = input('  Describe: ').strip()
                if not any(out[k] for k, _ in FAILURE_KEYS):
                    print("  (no failure category selected)")
            out['outcome_reported'] = True
        except (KeyboardInterrupt, EOFError):
            print("\n  Report skipped -- saved with outcome_reported=False.")
        self._outcome = out
        return out

    # ── save ──────────────────────────────────────────────────────────────────

    def save(self, episode_idx: int, total_steps: int, duration_s: float) -> Path:
        """Save collected data for one episode, append the summary row, reset."""
        out_path = self.log_dir / f"{self._file_stem(episode_idx)}.npz"

        video_path = ''
        if self._video is not None:
            # The writer was opened before the index was known -- rename to match.
            tmp = self._video['path']
            self._close_video()
            final = out_path.with_suffix('.mp4')
            if tmp != final:
                tmp.replace(final)
            video_path = str(final)

        ts = np.array(self._timestamps)
        op = np.array(self._op_mode, dtype=np.int8).reshape(-1, 2)
        pressure = np.array(self._pressure_kpa, dtype=np.float64)
        suction_on = np.array(self._suction_on, dtype=bool)
        threshold = float('nan') if self.seal_threshold_kpa is None else float(self.seal_threshold_kpa)
        sealed = (pressure <= threshold) if self.seal_threshold_kpa is not None \
            else np.zeros(len(pressure), dtype=bool)
        t0 = self._episode_t0
        times = {
            'completion_time_s': _first_time((op[:, 0] == 1) & (op[:, 1] == 1), ts, t0),
            'time_to_suction_s': _first_time(suction_on, ts, t0),
            'time_to_seal_s': _first_time(sealed, ts, t0),
        }
        outcome = self._outcome or {'outcome_reported': False, 'true_object': '', 'success': False,
                                    'wrong_object_target': '', 'other_note': '',
                                    **{k: False for k, _ in FAILURE_KEYS}}

        np.savez_compressed(
            out_path,
            timestamps=ts,
            tcp_poses=np.array(self._tcp_poses),
            pwm_actual=np.array(self._pwm_actual),
            executed_actions=np.array(self._executed_actions),
            pwm_commanded=np.array(self._pwm_commanded),
            op_mode=op,
            suction_cmd_raw=np.array(self._suction_cmd_raw, dtype=np.float64),
            suction_on=suction_on,
            pressure_kpa=pressure,
            sealed=sealed,
            predicted_horizons=np.array(self._predicted_horizons),
            plan_times_ms=np.array(self._plan_times_ms),
            plan_step_indices=np.array(self._plan_step_indices),
            total_steps=total_steps,
            duration_s=duration_s,
            episode_start_time=float('nan') if t0 is None else t0,
            checkpoint_path=self.checkpoint_path,
            system=self.system,
            predicted_object=self._predicted_object,
            seal_threshold_kpa=threshold,
            video_path=video_path,
            **times,
            **outcome,
        )
        print(f"  Log saved: {out_path}" + (f"  (+ video {Path(video_path).name})" if video_path else ''))

        summary = self.log_dir / 'episodes_summary.csv'
        new_file = not summary.exists()
        with open(summary, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({'episode': episode_idx, 'file': out_path.name, 'system': self.system,
                        'checkpoint': self.checkpoint_path, 'predicted_object': self._predicted_object,
                        'duration_s': round(duration_s, 2), 'total_steps': total_steps,
                        **{k: (round(v, 2) if v == v else '') for k, v in times.items()},
                        **{k: outcome[k] for k in outcome if k in SUMMARY_FIELDS}})
        self._reset()
        return out_path
