"""
propri_pid.py — Task-space PID feedback using the proprioception model.

Design
------
After the IK drives the robot near a waypoint (MOVE phase), this controller
runs during the HOLD phase:

    pred_pos = propri_model(flow, pwm)               # (3,) mm — actual tip
    error    = target_mm − pred_pos                  # task-space error
    u        = Kp·e + Ki·∫e + Kd·ė                  # correction (mm)
    virtual  = target_mm + u                         # biased IK target
    fb.step(direction toward virtual)                # IK → PWM → Arduino

The proprioception model replaces OptiTrack as the position sensor, so the
loop closes on measured flow rather than a motion-capture ground truth.

Requirements
------------
Before creating this controller, stop fb's internal serial reader so this
controller's SerialReader can read from fb.ser exclusively:

    fb.stop_flag["stop"] = True
    time.sleep(0.15)
    fb.ser.reset_input_buffer()
    sr  = SerialReader(fb.ser)
    pid = ProprioceptionPIDController(ckpt_dir="...", reader=sr, Kp=0.4)

Call stop() on sr when finished.

Gain tuning guide
-----------------
Start with Ki=Kd=0, Kp=0.3.  Increase Kp until error halves per hold cycle,
back off 20%.  Add Ki (0.01–0.05) to eliminate residual steady-state error.
Kd is rarely needed; flow noise amplifies the derivative term.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

_PROG_DIR = Path(__file__).resolve().parents[2]
if str(_PROG_DIR) not in sys.path:
    sys.path.insert(0, str(_PROG_DIR))

from flowbot.proprioception_model.dataset import (
    StandardScaler,
    _compute_features,
    parse_feature_groups,
    FEATURE_NAMES,
)
from flowbot.proprioception_model.model import PropMLP, PlainMLP


# ── Serial reader (background thread) ─────────────────────────────────────────

class SerialReader:
    """
    Background thread that parses Arduino CSV lines from an open serial.Serial.

    Arduino format (columns):
      t_ms, rawFlow1, proc_flow1, rawFlow2, proc_flow2, rawFlow3, proc_flow3, ...
    """

    def __init__(self, ser, maxlen: int = 200):
        self._ser    = ser
        self._buf: deque = deque(maxlen=maxlen)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                break
            if not line or not line[0].isdigit():
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                row = {
                    "proc_flow1": float(parts[2]),
                    "proc_flow2": float(parts[4]),
                    "proc_flow3": float(parts[6]),
                }
            except ValueError:
                continue
            with self._lock:
                self._buf.append(row)

    def latest(self) -> Optional[dict]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def drain(self):
        with self._lock:
            self._buf.clear()

    def stop(self):
        self._stop.set()


# ── PID ───────────────────────────────────────────────────────────────────────

class _TaskSpacePID:
    """Independent 3-axis PID. Inputs and outputs are in mm."""

    def __init__(self, Kp: float, Ki: float, Kd: float, integral_limit: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._iclamp     = integral_limit
        self._integral   = np.zeros(3, dtype=float)
        self._prev_error = np.zeros(3, dtype=float)
        self._last_t: Optional[float] = None

    def reset(self):
        self._integral[:]   = 0.0
        self._prev_error[:] = 0.0
        self._last_t        = None

    def step(self, error: np.ndarray) -> np.ndarray:
        now = time.perf_counter()
        dt  = (now - self._last_t) if self._last_t is not None else 0.05
        dt  = max(dt, 1e-6)
        self._last_t = now

        self._integral  += error * dt
        self._integral   = np.clip(self._integral, -self._iclamp, self._iclamp)
        derivative       = (error - self._prev_error) / dt
        self._prev_error = error.copy()

        return self.Kp * error + self.Ki * self._integral + self.Kd * derivative


# ── Main controller ───────────────────────────────────────────────────────────

class ProprioceptionPIDController:
    """
    Task-space PID controller backed by the trained proprioception model.

    Parameters
    ----------
    ckpt_dir       : Checkpoint directory (best_model.pt, scaler_x/y.pkl,
                     train_config.yaml).
    reader         : SerialReader providing latest() → dict with proc_flow1/2/3.
    Kp, Ki, Kd    : PID gains (mm error → mm correction).
    integral_limit : Anti-windup clamp on the integral term (mm).
    device         : Torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        ckpt_dir:       str,
        reader:         SerialReader,
        Kp:             float = 0.4,
        Ki:             float = 0.01,
        Kd:             float = 0.0,
        integral_limit: float = 5.0,
        device:         str   = "cpu",
        pred_signs:     tuple = (1.0, 1.0, 1.0),
    ):
        self._reader     = reader
        self._pid        = _TaskSpacePID(Kp, Ki, Kd, integral_limit)
        self._device     = device
        self._pred_signs = np.array(pred_signs, dtype=float)

        # ── Load checkpoint ──────────────────────────────────────────────
        ckpt = Path(ckpt_dir)
        with open(ckpt / "train_config.yaml") as f:
            cfg = yaml.safe_load(f)

        feature_indices = cfg.get("feature_indices")
        if feature_indices is None:
            feature_indices = parse_feature_groups(cfg.get("features", "flow,K,diff"))
        self._feature_indices = feature_indices
        input_size = len(feature_indices)
        feat_names = [FEATURE_NAMES[i] for i in feature_indices]

        arch   = cfg.get("arch",       "resmlp")
        hidden = cfg.get("hidden",     128)
        blocks = cfg.get("num_blocks", 3)

        if arch == "plainmlp":
            model = PlainMLP(input_size=input_size, hidden_size=hidden,
                             num_layers=blocks, dropout=0.0, output_size=3)
        else:
            model = PropMLP(input_size=input_size, hidden_size=hidden,
                            num_blocks=blocks, dropout=0.0, output_size=3)

        model.load_state_dict(torch.load(ckpt / "best_model.pt", map_location=device))
        model.eval()
        self._model    = model
        self._x_scaler = StandardScaler.load(ckpt / "scaler_x.pkl")
        self._y_scaler = StandardScaler.load(ckpt / "scaler_y.pkl")

        print(f"[propri_pid] Checkpoint  : {ckpt_dir}")
        print(f"[propri_pid] Features    : {feat_names}  ({input_size}-dim)")
        print(f"[propri_pid] Gains       : Kp={Kp}  Ki={Ki}  Kd={Kd}  iclamp={integral_limit}")
        print(f"[propri_pid] pred_signs  : {list(pred_signs)}")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer(self, pwm: np.ndarray, flow: dict) -> np.ndarray:
        """Proprioception model → predicted tip position (x, y, z) mm."""
        f = np.array([flow["proc_flow1"], flow["proc_flow2"], flow["proc_flow3"]],
                     dtype=np.float32)
        p = pwm.astype(np.float32)
        X   = _compute_features(p, f)[self._feature_indices]                 # (n_feat,)
        X_s = self._x_scaler.transform(X[None])                              # (1, n_feat)
        y_s = self._model(torch.tensor(X_s, dtype=torch.float32)).numpy()    # (1, 3)
        return self._y_scaler.inverse_transform(y_s)[0]                       # (3,) mm

    def predict_pos(self, fb) -> Optional[np.ndarray]:
        """Return proprioception estimate in IK frame (mm), or None if no flow data."""
        flow = self._reader.latest()
        if flow is None:
            return None
        return self._infer(fb.last_pwm, flow) * self._pred_signs

    # ── Control ───────────────────────────────────────────────────────────────

    def reset(self):
        """Clear integrator and derivative state. Call before each new hold phase."""
        self._pid.reset()

    def correct(self, fb, target_mm: np.ndarray):
        """
        One PID correction step during the hold phase.

        Returns (pwm, pred_ik_mm): the sent PWM array and the proprioception
        estimate used for this correction (both in sync — same last_pwm).

        Algorithm:
          pred_pos       = propri_model(flow, fb.last_pwm)
          error          = target_mm − pred_pos
          u (mm)         = PID(error)
          virtual_target = target_mm + u          ← biased IK goal
          IK(virtual_target) → PWM → Arduino

        Direct IK on virtual_target (not fb.step) so the full correction is
        applied immediately. fb.step's one-step limit would otherwise prevent
        the PWM from changing when fb.pc is reset between calls.
        """
        flow = self._reader.latest()
        if flow is None:
            return fb.last_pwm.copy(), None

        pred_ik = self._infer(fb.last_pwm, flow) * self._pred_signs

        # target_mm is in IK absolute frame (Z ≈ 95–120 mm).
        # Subtract fb.pc_init to convert to IK relative frame (Z ≈ 0–25 mm).
        target_rel = np.asarray(target_mm, dtype=float) - np.asarray(fb.pc_init, dtype=float)
        error = target_rel - pred_ik
        u     = self._pid.step(error)
        # print(f"[propri_pid] target={target_mm}  pred={pred_ik}  error={error}  u={u}")
        virtual_target = np.asarray(target_mm, dtype=float) + u
        # Clamp to workspace so IK never sees an out-of-range position.
        virtual_safe = fb.apply_workspace_constraint(fb.pc, virtual_target, "backtrack")

        try:
            ik  = fb.flowbot.inverse_pressures_from_position(virtual_safe)
            pwm = np.asarray(ik["pwm"], dtype=int).reshape(3,)
        except Exception:
            pwm = fb.last_pwm.copy()

        fb.serial_sending(pwm)
        fb.last_pwm = pwm

        # Keep fb.pc at the nominal target so the display dot stays on the waypoint
        # and the next MOVE phase starts from the correct nominal position.
        fb.pc[:] = np.asarray(target_mm, dtype=float)

        return pwm, pred_ik
