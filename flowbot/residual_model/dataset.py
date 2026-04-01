"""
dataset.py  –  FlowbotErrorDataset

Loads task-log CSVs, reconstructs the OptiTrack → manipulator-frame transform,
extracts hold-phase endpoints, and builds fixed-length input sequences.

Each training sample:
  X : float32 tensor (seq_len, 6)   ← [cmd_pc_x/y/z, pwm_1/2/3] for the
                                        seq_len ticks ending at the hold endpoint
  y : float32 tensor (3,)           ← [err_x, err_y, err_z] (mm) at hold end

Usage
-----
    from learning.error_model.dataset import FlowbotErrorDataset, build_datasets

    train_ds, val_ds, scaler = build_datasets(
        log_root="data/task_logs",
        seq_len=40,
        val_fraction=0.2,
    )
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
# Feature / target columns
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS      = ["cmd_pc_x", "cmd_pc_y", "cmd_pc_z", "pwm_1", "pwm_2", "pwm_3"]
FEATURE_COLS_OPTI = FEATURE_COLS + ["meas_x", "meas_y", "meas_z"]  # +optitrack (9 total)
TARGET_COLS       = ["err_x", "err_y", "err_z"]

# ──────────────────────────────────────────────────────────────────────────────
# OptiTrack → manipulator-frame transform
# (mirrors online_optitrack.OptiTrack.opti_to_manip + axis flips in execute_task)
# ──────────────────────────────────────────────────────────────────────────────
_R_MW = np.array([[0.0,  0.0,  1.0],
                  [-1.0, 0.0,  0.0],
                  [0.0, -1.0,  0.0]])


def _Rz(alpha: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]])


def opti_to_manip_mm(pos_W_m: np.ndarray,
                     origin_W_m: np.ndarray,
                     alpha_rad: float) -> np.ndarray:
    p_rel = np.asarray(pos_W_m, dtype=float) - np.asarray(origin_W_m, dtype=float)
    pM = _Rz(alpha_rad) @ (_R_MW @ p_rel)
    pM *= 1000.0
    pM[0] = -pM[0]
    pM[1] = -pM[1]
    return pM


# ──────────────────────────────────────────────────────────────────────────────
# Hold-phase endpoint detection (same logic as analyze_task.py)
# ──────────────────────────────────────────────────────────────────────────────
def extract_hold_phases(df: pd.DataFrame, tol: float = 0.01) -> List[tuple]:
    """
    Return (start_idx, end_idx) pairs for each hold phase.

    start_idx : first tick where cmd_pc becomes constant  ← window ends here
    end_idx   : last tick of the hold phase               ← error label taken here

    Using start_idx for the window means the training sequence contains only
    travel ticks (+ 1 hold tick), matching the rolling-buffer deployment scenario
    where the window is entirely travel ticks.
    """
    cmd   = df[["cmd_pc_x", "cmd_pc_y", "cmd_pc_z"]].values
    delta = np.linalg.norm(np.diff(cmd, axis=0), axis=1)

    phases: List[tuple] = []
    in_hold    = False
    hold_start = 0

    for i, d in enumerate(delta):
        if d <= tol:
            if not in_hold:
                in_hold    = True
                hold_start = i          # first tick where cmd is constant
        else:
            if in_hold:
                phases.append((hold_start, i))   # i = last hold tick
                in_hold = False

    if in_hold:
        phases.append((hold_start, len(df) - 1))

    return phases


def extract_hold_endpoints(df: pd.DataFrame, tol: float = 0.01) -> List[int]:
    """Return row indices of the last tick in each hold phase (legacy)."""
    return [end for _, end in extract_hold_phases(df, tol)]


# ──────────────────────────────────────────────────────────────────────────────
# Per-feature normaliser (fit on training set, applied to val/test)
# ──────────────────────────────────────────────────────────────────────────────
class Scaler:
    """Z-score normaliser for X (features) and optionally y (targets)."""

    def __init__(self):
        self.x_mean: Optional[np.ndarray] = None
        self.x_std:  Optional[np.ndarray] = None
        self.y_mean: Optional[np.ndarray] = None
        self.y_std:  Optional[np.ndarray] = None

    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        """Fit from lists of (seq_len, n_feat) arrays and (3,) arrays."""
        X_all = np.vstack([x.reshape(-1, x.shape[-1]) for x in X_list])
        y_all = np.vstack(y_list)
        self.x_mean = X_all.mean(0)
        self.x_std  = X_all.std(0).clip(1e-8)
        self.y_mean = y_all.mean(0)
        self.y_std  = y_all.std(0).clip(1e-8)
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "Scaler":
        with open(path, "rb") as f:
            return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Core dataset
# ──────────────────────────────────────────────────────────────────────────────
class FlowbotErrorDataset(Dataset):
    """
    Parameters
    ----------
    csv_paths   : list of CSV file paths to load
    seq_len     : number of ticks in each input sequence
    l0, lu      : robot geometry (mm) for opti_origin reconstruction
    alpha_deg   : OptiTrack rotation angle (degrees)
    hold_tol      : cmd_pc stability tolerance for hold-phase detection (mm)
    home_xy_tol   : waypoints with |x|, |y| < this are treated as home and skipped
    home_z_tol    : waypoints with |z - (l0+lu)| < this are treated as home Z (mm)
    use_optitrack : if True, append measured position (opti→manip frame) to features
                    → input_size becomes 9 instead of 6
    scaler        : fitted Scaler; if None the dataset stores raw (un-normalised) data
    verbose       : print per-file summary
    """

    def __init__(
        self,
        csv_paths: List[str | Path],
        seq_len:       int   = 20,
        l0:            float = 82.0,
        lu:            float = 13.5,
        alpha_deg:     float = -30.0,
        hold_tol:      float = 0.01,
        home_xy_tol:   float = 0.1,
        home_z_tol:    float = 1.0,
        use_optitrack: bool  = False,
        scaler:        Optional[Scaler] = None,
        verbose:       bool  = True,
    ):
        self.seq_len      = seq_len
        self.alpha_rad    = np.deg2rad(alpha_deg)
        self.l0_lu_m      = (l0 + lu) / 1000.0
        self.l0_lu_mm     = l0 + lu
        self.hold_tol     = hold_tol
        self.home_xy_tol  = home_xy_tol
        self.home_z_tol   = home_z_tol
        self.use_optitrack = use_optitrack
        self.scaler      = scaler

        # Raw (unscaled) arrays accumulated across all files
        self._X_raw: List[np.ndarray] = []   # each (seq_len, 6)
        self._y_raw: List[np.ndarray] = []   # each (3,)
        self._meta:  List[dict]       = []   # source file, waypoint index

        for p in csv_paths:
            self._load_csv(Path(p), verbose)

        if verbose:
            print(f"[dataset] Total samples: {len(self._X_raw)}")

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_csv(self, path: Path, verbose: bool):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[dataset] WARNING: could not read {path.name}: {e}")
            return

        # Check required columns
        required = FEATURE_COLS + ["opti_x", "opti_y", "opti_z"]
        if not all(c in df.columns for c in required):
            print(f"[dataset] WARNING: missing columns in {path.name}, skipping.")
            return

        # Drop rows with NaN in feature or opti columns
        df = df.dropna(subset=FEATURE_COLS + ["opti_x", "opti_y", "opti_z"]).reset_index(drop=True)
        if len(df) < self.seq_len + 1:
            print(f"[dataset] WARNING: {path.name} too short ({len(df)} rows), skipping.")
            return

        # Reconstruct OptiTrack origin from first valid (non-zero) reading
        valid_mask = (df["opti_x"] != 0.0)
        if not valid_mask.any():
            print(f"[dataset] WARNING: no valid opti readings in {path.name}, skipping.")
            return

        first = df[valid_mask].iloc[0]
        opti_origin = np.array([
            first["opti_x"],
            first["opti_y"] + self.l0_lu_m,
            first["opti_z"],
        ], dtype=float)

        # Build feature matrix (n_rows, 6 or 9)
        feat = df[FEATURE_COLS].values.astype(np.float32)

        if self.use_optitrack:
            # Transform each optitrack reading to manipulator frame (mm)
            opti_raw = df[["opti_x", "opti_y", "opti_z"]].values  # (N, 3) metres
            opti_mm  = np.zeros((len(df), 3), dtype=np.float32)
            for i, pos_W in enumerate(opti_raw):
                if np.any(np.isnan(pos_W)) or (pos_W == 0.0).all():
                    opti_mm[i] = feat[i, :3]   # fall back to cmd_pc
                else:
                    opti_mm[i] = opti_to_manip_mm(pos_W, opti_origin, self.alpha_rad)
            feat = np.concatenate([feat, opti_mm], axis=1)   # (N, 9)

        # Pre-compute home mask: rows where cmd_pc is at home (XY ≈ 0, Z ≈ l0+lu)
        at_home = ((df["cmd_pc_x"].abs() < self.home_xy_tol) &
                   (df["cmd_pc_y"].abs() < self.home_xy_tol) &
                   ((df["cmd_pc_z"] - self.l0_lu_mm).abs() < self.home_z_tol)).values

        # Detect hold phases: (start_idx, end_idx) pairs
        hold_phases = extract_hold_phases(df, tol=self.hold_tol)

        n_added = 0
        for wp_i, (win_end_idx, label_idx) in enumerate(hold_phases):
            label_row = df.iloc[label_idx]

            # Skip home position label
            if at_home[label_idx]:
                continue

            # Window ends at hold_start (win_end_idx) — all travel ticks
            start = win_end_idx - self.seq_len + 1
            if start < 0:
                continue   # not enough travel history

            # Skip if any tick in the window passes through home
            if at_home[start : win_end_idx + 1].any():
                continue

            # Build sequence: seq_len ticks ending at first hold tick
            X = feat[start : win_end_idx + 1]      # (seq_len, 6)
            n_feat = feat.shape[1]
            assert X.shape == (self.seq_len, n_feat), f"Shape mismatch: {X.shape}"

            # Compute error from last hold tick (robot settled, OptiTrack stable)
            pos_W    = np.array([label_row["opti_x"], label_row["opti_y"], label_row["opti_z"]])
            measured = opti_to_manip_mm(pos_W, opti_origin, self.alpha_rad)
            target   = np.array([label_row["cmd_pc_x"], label_row["cmd_pc_y"], label_row["cmd_pc_z"]])
            err      = (measured - target).astype(np.float32)

            self._X_raw.append(X)
            self._y_raw.append(err)
            self._meta.append({"file": path.name, "wp": wp_i, "row": label_idx})
            n_added += 1

        if verbose:
            print(f"[dataset]   {path.name}: {n_added} samples")

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._X_raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self._X_raw[idx].copy()
        y = self._y_raw[idx].copy()

        if self.scaler is not None:
            X = self.scaler.transform_X(X).astype(np.float32)
            y = self.scaler.transform_y(y).astype(np.float32)

        return torch.from_numpy(X), torch.from_numpy(y)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def X_raw(self) -> np.ndarray:
        """All raw (unscaled) feature arrays stacked: (N, seq_len, 6)."""
        return np.stack(self._X_raw)

    @property
    def y_raw(self) -> np.ndarray:
        """All raw error arrays stacked: (N, 3)."""
        return np.stack(self._y_raw)

    def get_meta(self, idx: int) -> dict:
        return self._meta[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience builder: train / val split + scaler fitting
# ──────────────────────────────────────────────────────────────────────────────
def build_datasets(
    log_root:      str | Path = "data/task_logs",
    seq_len:       int   = 40,
    val_fraction:  float = 0.2,
    seed:          int   = 42,
    l0:            float = 82.0,
    lu:            float = 13.5,
    alpha_deg:     float = -30.0,
    hold_tol:      float = 0.01,
    use_optitrack: bool  = False,
    verbose:       bool  = True,
) -> Tuple[FlowbotErrorDataset, FlowbotErrorDataset, Scaler]:
    """
    Discover all CSVs under `log_root`, split by file into train/val,
    fit a Scaler on the training set, and return (train_ds, val_ds, scaler).

    Splitting by file (not by sample) prevents data leakage across runs.
    """
    log_root  = Path(log_root)
    if log_root.is_dir():
        all_csvs  = sorted(log_root.rglob("*.csv"))
    else:
        all_csvs  = [log_root]

    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found under {log_root}")

    if verbose:
        print(f"[dataset] Found {len(all_csvs)} CSV files in {log_root}")

    # Shuffle file list and split
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(all_csvs))       
    n_val = max(1, int(len(all_csvs) * val_fraction))
    val_files   = [all_csvs[i] for i in idxs[:n_val]]
    train_files = [all_csvs[i] for i in idxs[n_val:]]

    if verbose:
        print(f"[dataset] Train files: {len(train_files)}, Val files: {len(val_files)}")

    _ds_kwargs = dict(seq_len=seq_len, l0=l0, lu=lu, alpha_deg=alpha_deg,
                      hold_tol=hold_tol, use_optitrack=use_optitrack)

    if not train_files:
        # eval-only mode (val_fraction=1.0): scaler must be supplied externally
        scaler   = Scaler()          # placeholder – caller replaces via val_ds.scaler
        train_ds = FlowbotErrorDataset([], **_ds_kwargs, scaler=scaler, verbose=False)
        val_ds   = FlowbotErrorDataset(val_files, **_ds_kwargs, scaler=scaler, verbose=verbose)
    else:
        # Build raw (unscaled) training dataset first to fit the scaler
        train_ds_raw = FlowbotErrorDataset(
            train_files, **_ds_kwargs, scaler=None, verbose=verbose,
        )

        if len(train_ds_raw) == 0:
            raise ValueError("Training dataset is empty. Check CSV paths and column names.")

        scaler = Scaler().fit(train_ds_raw._X_raw, train_ds_raw._y_raw)

        # Re-wrap with scaler applied
        train_ds = FlowbotErrorDataset(
            train_files, **_ds_kwargs, scaler=scaler, verbose=False,
        )
        val_ds = FlowbotErrorDataset(
            val_files, **_ds_kwargs, scaler=scaler, verbose=verbose,
        )

    if verbose:
        print(f"[dataset] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        if scaler.x_mean is not None:
            print(f"[dataset] Feature means : {np.round(scaler.x_mean, 3)}")
            print(f"[dataset] Feature stds  : {np.round(scaler.x_std,  3)}")
            print(f"[dataset] Target means  : {np.round(scaler.y_mean, 3)} mm")
            print(f"[dataset] Target stds   : {np.round(scaler.y_std,  3)} mm")

    return train_ds, val_ds, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/task_logs"
    train_ds, val_ds, scaler = build_datasets(log_root=root, seq_len=40, verbose=True)

    X, y = train_ds[0]
    print(f"\nSample 0  X: {X.shape} {X.dtype}  y: {y.shape} {y.dtype}")
    print(f"y (normalised): {y.numpy()}")
    print(f"y (mm)        : {scaler.inverse_y(y.numpy())}")
    print(f"X (raw)       : {train_ds.__getitem__(0)}")
