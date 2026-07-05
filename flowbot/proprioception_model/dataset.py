"""
dataset.py — FlowTipDataset for proprioception model training.

CSV source: collect_flow_tip.py / collect_free_human.py
Columns used:
    pwm1_cmd, pwm2_cmd, pwm3_cmd        — actuator commands
    proc_flow1, proc_flow2, proc_flow3  — processed flow (L/min)
    opti_x_mm, opti_y_mm, opti_z_mm    — tip position labels (mm)

Full 12-dim canonical feature vector:
    index  group  name
    0      pwm    pwm1
    1      pwm    pwm2
    2      pwm    pwm3
    3      flow   flow1
    4      flow   flow2
    5      flow   flow3
    6      K      K1   = flow1 / (pwm1 + ε)    (valve efficiency)
    7      K      K2   = flow2 / (pwm2 + ε)
    8      K      K3   = flow3 / (pwm3 + ε)
    9      diff   diff12 = flow1 − flow2
    10     diff   diff23 = flow2 − flow3
    11     diff   diff13 = flow1 − flow3

Feature selection: --features is a comma-separated list of group names.
    e.g. "flow,K,diff"  → 9 features (no PWM)
         "pwm,diff"     → 6 features
         "pwm,flow,K,diff" → all 12
Selected features are always emitted in canonical index order.

Option C augmentation (train set only):
    α ~ Uniform(aug_alpha_min, 1.0) per sample
    flow_i *= α  →  K_i and diff_ij recomputed automatically (pwm unchanged)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Column names in the collect CSV
# ──────────────────────────────────────────────────────────────────────────────
_PWM_COLS   = ["pwm1_cmd", "pwm2_cmd", "pwm3_cmd"]
_FLOW_COLS  = ["proc_flow1", "proc_flow2", "proc_flow3"]
_LABEL_COLS = ["opti_x_mm", "opti_y_mm", "opti_z_mm"]

_EPS = np.float32(1e-3)

# ──────────────────────────────────────────────────────────────────────────────
# Feature groups — 12-dim canonical ordering
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "pwm1",  "pwm2",  "pwm3",    # 0–2   actuator commands
    "flow1", "flow2", "flow3",   # 3–5   flow readings (L/min)
    "K1",    "K2",    "K3",      # 6–8   valve efficiency
    "diff12","diff23","diff13",  # 9–11  pairwise flow differentials
]

FEATURE_GROUPS: dict = {
    "pwm":  [0, 1, 2],
    "flow": [3, 4, 5],
    "K":    [6, 7, 8],
    "diff": [9, 10, 11],
}


def parse_feature_groups(groups_str: str) -> List[int]:
    """
    Convert a comma-separated group string to a sorted list of feature indices.

    Examples
    --------
    >>> parse_feature_groups("flow,K,diff")
    [3, 4, 5, 6, 7, 8, 9, 10, 11]
    >>> parse_feature_groups("pwm,diff")
    [0, 1, 2, 9, 10, 11]
    """
    indices: set = set()
    for g in groups_str.split(","):
        g = g.strip()
        if g not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group '{g}'. Valid groups: {list(FEATURE_GROUPS)}"
            )
        indices.update(FEATURE_GROUPS[g])
    return sorted(indices)


def _compute_features(pwm: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Full 12-dim feature vector (caller selects a subset by index):
      [pwm1, pwm2, pwm3,  flow1, flow2, flow3,  K1, K2, K3,  diff12, diff23, diff13]
    """
    K    = flow / (pwm + _EPS)
    diff = np.array([flow[0] - flow[1], flow[1] - flow[2], flow[0] - flow[2]],
                    dtype=np.float32)
    return np.concatenate([pwm, flow, K, diff]).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Scaler — separate X and y scalers, both z-score
# ──────────────────────────────────────────────────────────────────────────────
class StandardScaler:
    """Z-score scaler for a single array type (X or y)."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0).clip(1e-8)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std + self.mean).astype(np.float32)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "StandardScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class FlowTipDataset(Dataset):
    """
    Parameters
    ----------
    samples         : list of {"pwm": (3,), "flow": (3,), "y": (3,)} dicts (raw, unscaled)
    augment         : if True apply Option C flow-scale augmentation each __getitem__
    aug_alpha_min   : lower bound of α ~ Uniform(aug_alpha_min, 1.0)
    x_scaler        : fitted StandardScaler for features; None → raw features returned
    y_scaler        : fitted StandardScaler for labels;   None → raw labels returned
    rng             : numpy Generator for augmentation; created internally if None
    feature_indices : indices into the 12-dim canonical vector to use as model input.
                      None → all 12. Build with parse_feature_groups().
    """

    def __init__(
        self,
        samples:         List[dict],
        augment:         bool = False,
        aug_alpha_min:   float = 0.8,
        x_scaler:        Optional[StandardScaler] = None,
        y_scaler:        Optional[StandardScaler] = None,
        rng:             Optional[np.random.Generator] = None,
        feature_indices: Optional[List[int]] = None,
    ):
        self._samples         = samples
        self._augment         = augment
        self._aug_alpha_min   = aug_alpha_min
        self._x_scaler        = x_scaler
        self._y_scaler        = y_scaler
        self._rng             = rng if rng is not None else np.random.default_rng()
        self._feature_indices = feature_indices   # None → all 12

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s    = self._samples[idx]
        pwm  = s["pwm"].copy()
        flow = s["flow"].copy()
        y    = s["y"].copy()

        if self._augment:
            alpha = self._rng.uniform(self._aug_alpha_min, 1.0)
            flow  = flow * alpha   # K and diff recomputed inside _compute_features

        X = _compute_features(pwm, flow)           # (12,)
        if self._feature_indices is not None:
            X = X[self._feature_indices]           # select subset

        if self._x_scaler is not None:
            X = self._x_scaler.transform(X[None])[0]
        if self._y_scaler is not None:
            y = self._y_scaler.transform(y[None])[0]

        return torch.from_numpy(X), torch.from_numpy(y)


# ──────────────────────────────────────────────────────────────────────────────
# CSV loader
# ──────────────────────────────────────────────────────────────────────────────
def _load_csv(path: Path, verbose: bool,
              states: Optional[set] = None) -> List[dict]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[dataset] WARNING: could not read {path.name}: {e}")
        return []

    required = _PWM_COLS + _FLOW_COLS + _LABEL_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[dataset] WARNING: {path.name} missing columns {missing}, skipping.")
        return []

    if states is not None and "state" in df.columns:
        df = df[df["state"].isin(states)].reset_index(drop=True)

    df = df.dropna(subset=_LABEL_COLS).reset_index(drop=True)
    if len(df) == 0:
        if verbose:
            print(f"[dataset]   {path.name}: 0 valid rows (all labels NaN)")
        return []

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "pwm":  np.array([row[c] for c in _PWM_COLS],  dtype=np.float32),
            "flow": np.array([row[c] for c in _FLOW_COLS], dtype=np.float32),
            "y":    np.array([row[c] for c in _LABEL_COLS], dtype=np.float32),
        })

    if verbose:
        print(f"[dataset]   {path.name}: {len(samples)} samples")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ──────────────────────────────────────────────────────────────────────────────
def build_datasets(
    data_dir:      str | Path,
    val_fraction:  float = 0.2,
    aug_alpha_min: float = 0.8,
    seed:          int   = 42,
    verbose:       bool  = True,
    states:        Optional[set] = None,
    features:      str   = "flow,K,diff",
) -> Tuple[FlowTipDataset, FlowTipDataset, StandardScaler, StandardScaler]:
    """
    Load all CSVs under data_dir, split train / val, fit scalers on training data.

    Parameters
    ----------
    features : comma-separated group names.
               "pwm,flow,K,diff" → all 12 features
               "flow,K,diff"     → 9 features (no PWM, default)
               "pwm,diff"        → 6 features (actuator + differentials)
               "flow"            → 3 features (raw flow only)

    Returns
    -------
    (train_ds, val_ds, x_scaler, y_scaler)
    """
    data_dir        = Path(data_dir)
    feature_indices = parse_feature_groups(features)
    input_size      = len(feature_indices)

    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    if verbose:
        state_label = ", ".join(sorted(states)) if states is not None else "all"
        feat_names  = [FEATURE_NAMES[i] for i in feature_indices]
        print(f"[dataset] Found {len(csvs)} CSV files in {data_dir}  (states={state_label})")
        print(f"[dataset] Features ({input_size}): {feat_names}")

    per_file: List[List[dict]] = []
    for p in csvs:
        s = _load_csv(p, verbose, states=states)
        if s:
            per_file.append(s)

    if not per_file:
        raise ValueError("No valid samples found. Check CSV paths and OptiTrack columns.")

    rng           = np.random.default_rng(seed)
    train_samples: List[dict] = []
    val_samples:   List[dict] = []
    for file_samples in per_file:
        idxs    = rng.permutation(len(file_samples))
        n_val_f = max(1, int(len(file_samples) * val_fraction))
        val_samples.extend(  [file_samples[i] for i in idxs[:n_val_f]])
        train_samples.extend([file_samples[i] for i in idxs[n_val_f:]])

    if verbose:
        print(f"[dataset] Total samples: {len(train_samples) + len(val_samples)}"
              f"  from {len(per_file)} files (stratified split)")
        print(f"[dataset] Train: {len(train_samples)}  Val: {len(val_samples)}")

    if not train_samples:
        raise ValueError("Training set is empty (val_fraction too high?).")

    X_train = np.stack([
        _compute_features(s["pwm"], s["flow"])[feature_indices] for s in train_samples
    ])
    y_train = np.stack([s["y"] for s in train_samples])

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    if verbose:
        print(f"[dataset] X mean : {np.round(x_scaler.mean, 3)}")
        print(f"[dataset] X std  : {np.round(x_scaler.std,  3)}")
        print(f"[dataset] y mean : {np.round(y_scaler.mean, 3)} mm")
        print(f"[dataset] y std  : {np.round(y_scaler.std,  3)} mm")

    train_ds = FlowTipDataset(
        train_samples, augment=True, aug_alpha_min=aug_alpha_min,
        x_scaler=x_scaler, y_scaler=y_scaler,
        rng=np.random.default_rng(seed),
        feature_indices=feature_indices,
    )
    val_ds = FlowTipDataset(
        val_samples, augment=False,
        x_scaler=x_scaler, y_scaler=y_scaler,
        feature_indices=feature_indices,
    )
    return train_ds, val_ds, x_scaler, y_scaler


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root     = sys.argv[1] if len(sys.argv) > 1 else "data/flow_tip"
    features = sys.argv[2] if len(sys.argv) > 2 else "flow,K,diff"
    train_ds, val_ds, xs, ys = build_datasets(root, features=features, verbose=True)
    X, y = train_ds[0]
    print(f"\nSample 0  X: {X.shape} {X.dtype}  y: {y.shape} {y.dtype}")
    print(f"X (scaled): {X.numpy()}")
    print(f"y (scaled): {y.numpy()}")
    print(f"y (mm)    : {ys.inverse_transform(y.numpy()[None])[0]}")
