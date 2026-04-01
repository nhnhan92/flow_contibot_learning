"""
compensator.py  –  Rolling-buffer ResGRU error compensator.

Two compensation methods
------------------------
"simple"
    Predict position error, scale by alpha, subtract from cmd_pc.
    Returns: position correction (mm) to SUBTRACT from cmd_pc.

    correction = comp.step(features)
    if correction is not None:
        cmd_pc -= correction

"mpc"
    One-step MPC: solve argmin_deltaU ||F(x,u+deltaU,h)||²_Q + ||deltaU||²_R
    via Adam gradient descent through the differentiable ResGRU.
    Returns: PWM delta (raw counts) to ADD to the nominal PWM command.

    delta_u = comp.step(features)
    if delta_u is not None:
        pwm_to_send = pwm_nominal + delta_u

Trigger conditions (shared by both methods)
-------------------------------------------
  1. Dead zone      : ||predicted_error|| > dead_zone_mm
  2. Displacement   : ||cmd_pc - last_correction_pos|| > min_displacement_mm

Usage
-----
    comp = ErrorCompensator.from_checkpoint(
        "flowbot/residual_model/checkpoints",
        method="mpc",          # or "simple"
    )
    comp.reset()

    features = [cmd_pc[0], cmd_pc[1], cmd_pc[2], pwm[0], pwm[1], pwm[2]]
    result   = comp.step(features)          # or comp.step(features, opti_pos_world=xyz)
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import math
import yaml
from collections import deque
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch

from flowbot.residual_model.dataset import Scaler
from flowbot.residual_model.model import ResGRU


def _opti_to_manip_mm(
    pos_world: np.ndarray,
    opti_origin: np.ndarray,
    alpha_rad: float,
) -> np.ndarray:
    """Transform OptiTrack world-frame position (mm) → manipulator frame (mm)."""
    p = pos_world - opti_origin
    c, s = math.cos(alpha_rad), math.sin(alpha_rad)
    return np.array([c*p[0]+s*p[1], -s*p[0]+c*p[1], p[2]], dtype=np.float32)


class ErrorCompensator:
    """
    Rolling-buffer error compensator wrapping a trained ResGRU.

    Parameters
    ----------
    model                : trained ResGRU (eval mode)
    scaler               : fitted Scaler
    seq_len              : rolling buffer length (must match training)
    method               : "simple" or "mpc"
    alpha                : [simple] correction gain in (0,1]
    dead_zone_mm         : minimum ||predicted_error|| to trigger
    min_displacement_mm  : minimum displacement since last correction
    use_optitrack        : append OptiTrack position to features (input_size=9)
    opti_origin          : OptiTrack base origin in world frame (mm)
    alpha_rad            : manipulator mounting angle (rad)
    mpc_Q                : [mpc] position error weight
    mpc_R                : [mpc] control effort weight
    mpc_n_iter           : [mpc] number of Adam optimisation steps per tick
    mpc_lr               : [mpc] Adam learning rate
    mpc_max_delta_pwm    : [mpc] clamp |deltaU| to this value (raw PWM counts)
    device               : torch device
    """

    def __init__(
        self,
        model:               ResGRU,
        scaler:              Scaler,
        seq_len:             int   = 20,
        method:              Literal["simple", "mpc"] = "simple",
        # simple params
        alpha:               float = 0.5,
        # shared trigger params
        dead_zone_mm:        float = 0.5,
        min_displacement_mm: float = 5.0,
        # optitrack
        use_optitrack:       bool  = False,
        opti_origin:         np.ndarray | None = None,
        alpha_rad:           float = 0.0,
        # mpc params
        mpc_Q:               float = 1.0,
        mpc_R:               float = 0.01,
        mpc_n_iter:          int   = 20,
        mpc_lr:              float = 0.1,
        mpc_max_delta_pwm:   float = 10.0,
        device:              torch.device | None = None,
    ):
        self.model               = model
        self.scaler              = scaler
        self.seq_len             = seq_len
        self.method              = method
        self.alpha               = alpha
        self.dead_zone_mm        = dead_zone_mm
        self.min_displacement_mm = min_displacement_mm
        self.use_optitrack       = use_optitrack
        self.opti_origin         = opti_origin if opti_origin is not None else np.zeros(3, dtype=np.float32)
        self.alpha_rad           = alpha_rad
        self.mpc_Q               = mpc_Q
        self.mpc_R               = mpc_R
        self.mpc_n_iter          = mpc_n_iter
        self.mpc_lr              = mpc_lr
        self.mpc_max_delta_pwm   = mpc_max_delta_pwm
        self.device              = device or torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        # ── state ─────────────────────────────────────────────────────────────
        self._buf:                  deque                = deque(maxlen=seq_len)
        self._last_feat_raw:        Optional[np.ndarray] = None   # unnorm, for MPC
        self._last_correction_pos:  Optional[np.ndarray] = None
        self._last_pred:            Optional[np.ndarray] = None   # for inspection

        # cached torch tensors for scaler (MPC needs differentiable ops)
        self._y_mean = torch.tensor(scaler.y_mean, dtype=torch.float32, device=self.device)
        self._y_std  = torch.tensor(scaler.y_std,  dtype=torch.float32, device=self.device)
        self._x_std  = torch.tensor(scaler.x_std,  dtype=torch.float32, device=self.device)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir:            str | Path,
        method:              Literal["simple", "mpc"] = "simple",
        alpha:               float = 0.5,
        dead_zone_mm:        float = 0.5,
        min_displacement_mm: float = 5.0,
        mpc_Q:               float = 1.0,
        mpc_R:               float = 0.01,
        mpc_n_iter:          int   = 20,
        mpc_lr:              float = 0.1,
        mpc_max_delta_pwm:   float = 10.0,
        device:              torch.device | None = None,
    ) -> "ErrorCompensator":
        ckpt_dir = Path(ckpt_dir)
        cfg_path = ckpt_dir / "train_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"train_config.yaml not found in {ckpt_dir}")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        dev           = device or torch.device("cpu")
        use_optitrack = cfg.get("use_optitrack", False)
        input_size    = cfg.get("input_size", 9 if use_optitrack else 6)
        alpha_deg     = cfg.get("alpha_deg", 0.0)

        model = ResGRU(
            input_size=input_size,
            hidden_size=cfg["hidden"],
            num_layers=cfg["layers"],
            dropout=0.0,
            output_size=3,
        )
        model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=dev))
        scaler  = Scaler.load(ckpt_dir / "scaler.pkl")
        seq_len = cfg["seq_len"]

        print(
            f"[compensator] Loaded ResGRU  input_size={input_size}  seq_len={seq_len}  "
            f"method={method}  use_optitrack={use_optitrack}"
        )
        return cls(
            model, scaler,
            seq_len=seq_len, method=method,
            alpha=alpha, dead_zone_mm=dead_zone_mm,
            min_displacement_mm=min_displacement_mm,
            use_optitrack=use_optitrack,
            alpha_rad=math.radians(alpha_deg),
            mpc_Q=mpc_Q, mpc_R=mpc_R,
            mpc_n_iter=mpc_n_iter, mpc_lr=mpc_lr,
            mpc_max_delta_pwm=mpc_max_delta_pwm,
            device=dev,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self):
        """Clear all state (call before each session)."""
        self._buf.clear()
        self._last_feat_raw       = None
        self._last_correction_pos = None
        self._last_pred           = None

    def step(
        self,
        features:       List[float] | np.ndarray,
        opti_pos_world: np.ndarray | None = None,
    ) -> Optional[np.ndarray]:
        """
        Ingest one tick and return a correction (or None).

        method="simple" → returns position correction (mm); subtract from cmd_pc.
        method="mpc"    → returns PWM delta (counts);       add to pwm_nominal.

        Parameters
        ----------
        features       : [cmd_pc_x, cmd_pc_y, cmd_pc_z, pwm_1, pwm_2, pwm_3]
        opti_pos_world : (3,) OptiTrack position in world frame (mm), optional.
        """
        feat_raw = np.asarray(features, dtype=np.float32)
        cmd_pc   = feat_raw[:3].copy()

        # Optionally append OptiTrack position (manip frame)
        if self.use_optitrack:
            if (opti_pos_world is None
                    or np.any(np.isnan(opti_pos_world))
                    or (np.asarray(opti_pos_world) == 0.0).all()):
                opti_mm = cmd_pc.copy()
            else:
                opti_mm = _opti_to_manip_mm(
                    np.asarray(opti_pos_world, dtype=np.float32),
                    self.opti_origin, self.alpha_rad,
                )
            feat_raw = np.concatenate([feat_raw, opti_mm])   # (9,)

        # Store raw features for MPC (before normalisation)
        self._last_feat_raw = feat_raw.copy()

        # Push normalised features into rolling buffer
        feat_norm = self.scaler.transform_X(feat_raw[None, :]).squeeze(0)
        self._buf.append(feat_norm)

        if len(self._buf) < self.seq_len:
            return None

        # Get prediction for trigger check (no grad needed here)
        pred = self._predict_np()
        self._last_pred = pred

        # ── Trigger conditions ────────────────────────────────────────────────
        if np.linalg.norm(pred) < self.dead_zone_mm:
            return None
        if self._last_correction_pos is not None:
            if np.linalg.norm(cmd_pc - self._last_correction_pos) < self.min_displacement_mm:
                return None

        self._last_correction_pos = cmd_pc

        if self.method == "mpc":
            return self._step_mpc()
        else:
            return (self.alpha * pred).astype(np.float32)

    @property
    def last_prediction(self) -> Optional[np.ndarray]:
        """Most recent raw model prediction in mm (before any scaling)."""
        return self._last_pred

    # ── simple: numpy forward pass ────────────────────────────────────────────

    def _predict_np(self) -> np.ndarray:
        """Forward pass with no gradient. Returns error prediction in mm."""
        x = torch.from_numpy(
            np.stack(list(self._buf), axis=0)[None, ...]    # (1, T, F)
        ).to(self.device)
        with torch.no_grad():
            wl = self.seq_len // 2
            hidden       = self.model.warmup(x[:, :wl, :])
            pred_norm, _ = self.model(x[:, wl:, :], hidden)
        return self.scaler.inverse_y(
            pred_norm.cpu().numpy().squeeze(0)
        ).astype(np.float32)

    # ── mpc: gradient-based deltaU optimisation ───────────────────────────────

    def _step_mpc(self) -> np.ndarray:
        """
        Solve:  argmin_deltaU  ||ResGRU([..., u+deltaU], h)||²_Q  +  ||deltaU||²_R

        deltaU lives in raw PWM space (counts). Only the PWM part (indices 3:6)
        of the current tick is perturbed; history ticks are unchanged.

        Returns deltaU as np.ndarray(3,) in raw PWM counts.
        """
        # History: all ticks except the last (fixed, no grad)
        hist_np = np.stack(list(self._buf)[:-1], axis=0)          # (T-1, F)
        hist    = torch.from_numpy(hist_np).float().to(self.device)  # detached

        # Current tick (normalised) as base for perturbation
        last_norm = torch.from_numpy(self._buf[-1]).float().to(self.device)

        # PWM std for converting raw deltaU → normalised perturbation
        pwm_std = self._x_std[3:6]                                # (3,)

        # Optimisation variable: deltaU in raw PWM space
        delta_u = torch.zeros(3, dtype=torch.float32,
                              device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta_u], lr=self.mpc_lr)

        wl = self.seq_len // 2

        for _ in range(self.mpc_n_iter):
            # Perturb current tick's PWM features in normalised space
            delta_u_clamped = delta_u.clamp(-self.mpc_max_delta_pwm,
                                             self.mpc_max_delta_pwm)
            last_perturbed        = last_norm.clone()
            last_perturbed[3:6]   = last_perturbed[3:6] + delta_u_clamped / pwm_std

            # Build full sequence (1, T, F)
            seq = torch.cat(
                [hist, last_perturbed.unsqueeze(0)], dim=0
            ).unsqueeze(0)

            # Forward pass (differentiable)
            hidden       = self.model.warmup(seq[:, :wl, :])
            pred_norm, _ = self.model(seq[:, wl:, :], hidden)

            # Denormalise prediction to mm (differentiable linear op)
            pred_mm = pred_norm.squeeze(0) * self._y_std + self._y_mean

            # Loss: tracking + regularisation
            loss = (self.mpc_Q * pred_mm.pow(2).sum()
                  + self.mpc_R * delta_u_clamped.pow(2).sum())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return delta_u.detach().clamp(
            -self.mpc_max_delta_pwm, self.mpc_max_delta_pwm
        ).cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    USE_OPTI = False
    IN_SIZE  = 9 if USE_OPTI else 6

    class _MockScaler:
        def __init__(self, n):
            self.x_mean = np.zeros(n, dtype=np.float32)
            self.x_std  = np.ones(n,  dtype=np.float32)
            self.y_mean = np.zeros(3, dtype=np.float32)
            self.y_std  = np.ones(3,  dtype=np.float32)
        def transform_X(self, X): return (X - self.x_mean) / self.x_std
        def inverse_y(self, y):   return y * self.y_std + self.y_mean
        def save(self, _): pass

    rng = np.random.default_rng(0)
    cmd = np.array([0.0, 0.0, 100.0])

    for method in ("simple", "mpc"):
        print(f"\n{'='*60}")
        print(f"  Method: {method}")
        print(f"{'='*60}")
        model = ResGRU(input_size=IN_SIZE, hidden_size=32, num_layers=2, dropout=0.0)
        comp  = ErrorCompensator(
            model, _MockScaler(IN_SIZE),
            seq_len=10, method=method,
            alpha=0.5, dead_zone_mm=0.3, min_displacement_mm=3.0,
            mpc_Q=1.0, mpc_R=0.01, mpc_n_iter=10, mpc_lr=0.1,
            use_optitrack=USE_OPTI,
        )
        comp.reset()
        label = "pos_corr(mm)" if method == "simple" else "delta_pwm"
        print(f"  {'tick':>4}  {'cmd_pc':>30}  {label}")
        print(f"  {'-'*65}")
        for t in range(20):
            cmd = cmd + rng.uniform(-1, 1, 3) * np.array([0.8, 0.8, 0.2])
            feat = [*cmd, 10.0, 10.0, 10.0]
            result = comp.step(feat)
            r_str = str(np.round(result, 3)) if result is not None else "buffering/skip"
            print(f"  {t:4d}  cmd={np.round(cmd,2)}  {r_str}")
