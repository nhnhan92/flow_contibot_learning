"""
evaluate.py  –  Load a trained ResGRU and evaluate on the validation set.

Prints MAE and RMSE per axis (mm) and overall distance error.
Plots:
  • Predicted vs True error for X, Y, Z (scatter)
  • Per-sample distance error histogram
  • Loss curve (if train_history.npy is present)

Usage
-----
    python -m learning.error_model.evaluate \
        --checkpoint  learning/checkpoints \
        --log_root    data/task_logs \
        --seq_len     40 \
        --warmup_len  20
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path

import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from flowbot.residual_model.dataset import Scaler, FlowbotErrorDataset, build_datasets
from flowbot.residual_model.model import ResGRU


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: Path, device: torch.device) -> ResGRU:
    cfg_path = Path(ckpt_dir) / "train_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"train_config.yaml not found in {ckpt_dir}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Backwards-compat: older checkpoints may not have input_size saved
    input_size = cfg.get("input_size", 9 if cfg.get("use_optitrack", False) else 6)

    model = ResGRU(
        input_size=input_size,
        hidden_size=cfg["hidden"],
        num_layers=cfg["layers"],
        dropout=0.0,          # no dropout during eval
        output_size=3,
    ).to(device)

    weights = ckpt_dir / "best_model.pt"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    print(f"[evaluate] Loaded model from {weights}  "
          f"input_size={input_size}  ({model.n_params():,} params)")
    return model, cfg


@torch.no_grad()
def predict_all(
    model: ResGRU,
    loader: DataLoader,
    warmup_len: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (preds, targets) both (N, 3) in normalised space."""
    preds_list, targets_list = [], []
    for X, y in loader:
        X = X.to(device)
        if warmup_len > 0 and warmup_len < X.shape[1]:
            hidden = model.warmup(X[:, :warmup_len, :])
        else:
            hidden = None
        x_roll = X[:, warmup_len:, :] if warmup_len > 0 else X
        pred, _ = model(x_roll, hidden)
        preds_list.append(pred.cpu().numpy())
        targets_list.append(y.numpy())
    return np.vstack(preds_list), np.vstack(targets_list)


def print_metrics(pred_mm: np.ndarray, true_mm: np.ndarray):
    """pred_mm, true_mm: (N, 3) in mm."""
    axes = ["X", "Y", "Z"]
    print("\n── Per-axis metrics (mm) ─────────────────────────────────")
    for i, ax in enumerate(axes):
        err = pred_mm[:, i] - true_mm[:, i]
        mae  = np.abs(err).mean()
        rmse = np.sqrt((err ** 2).mean())
        bias = err.mean()
        print(f"  {ax}: MAE={mae:.3f}  RMSE={rmse:.3f}  bias={bias:+.3f}")

    dist_err = np.linalg.norm(pred_mm - true_mm, axis=1)
    print(f"\n── Distance error (mm) ───────────────────────────────────")
    print(f"  Mean  = {dist_err.mean():.3f}")
    print(f"  Median= {np.median(dist_err):.3f}")
    print(f"  95th  = {np.percentile(dist_err, 95):.3f}")
    print(f"  Max   = {dist_err.max():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_scatter(pred_mm: np.ndarray, true_mm: np.ndarray, out_path: Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["tab:red", "tab:orange", "tab:green"]
    labels = ["X (mm)", "Y (mm)", "Z (mm)"]

    for i, (ax, col, lab) in enumerate(zip(axes, colors, labels)):
        lo = min(true_mm[:, i].min(), pred_mm[:, i].min()) - 1
        hi = max(true_mm[:, i].max(), pred_mm[:, i].max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ideal")
        ax.scatter(true_mm[:, i], pred_mm[:, i], s=15, alpha=0.5, color=col, label="samples")
        ax.set_xlabel(f"True err {lab}")
        ax.set_ylabel(f"Pred err {lab}")
        ax.set_title(lab)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Predicted vs True position error (mm)", fontsize=12)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[evaluate] Scatter plot saved → {out_path}")
    plt.show()


def plot_distance_hist(pred_mm: np.ndarray, true_mm: np.ndarray, out_path: Path | None = None):
    dist_err = np.linalg.norm(pred_mm - true_mm, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dist_err, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(dist_err.mean(),   color="red",    linestyle="--", label=f"mean={dist_err.mean():.2f} mm")
    ax.axvline(np.median(dist_err), color="orange", linestyle="--", label=f"median={np.median(dist_err):.2f} mm")
    ax.set_xlabel("Distance error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction distance error distribution")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[evaluate] Histogram saved → {out_path}")
    plt.show()


def plot_loss_curve(history_path: Path, out_path: Path | None = None):
    if not history_path.exists():
        return
    history = np.load(history_path, allow_pickle=True).item()
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue")
    ax.plot(epochs, history["val_loss"],   label="val",   color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (normalised)")
    ax.set_title("Training loss curve")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"[evaluate] Loss curve saved → {out_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint)

    model, cfg = load_model(ckpt_dir, device)

    # Use seq_len / warmup_len from checkpoint config if not overridden
    seq_len    = args.seq_len    if args.seq_len    is not None else cfg["seq_len"]
    warmup_len = args.warmup_len if args.warmup_len is not None else cfg["warmup_len"]
    warmup_len = min(warmup_len, seq_len - 1)

    scaler        = Scaler.load(ckpt_dir / "scaler.pkl")
    use_optitrack = cfg.get("use_optitrack", False)
    ds_kwargs     = dict(
        seq_len      = seq_len,
        l0           = cfg.get("l0", 60.0),
        lu           = cfg.get("lu", 35.0),
        alpha_deg    = cfg.get("alpha_deg", 0.0),
        use_optitrack= use_optitrack,
        scaler       = scaler,
    )

    if args.csv is not None:
        # ── Single CSV file ───────────────────────────────────────
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        val_ds = FlowbotErrorDataset([csv_path], **ds_kwargs, verbose=True)
        print(f"[evaluate] Evaluating on single file: {csv_path}  "
              f"({len(val_ds)} samples)")
    else:
        # ── Directory of CSV files (original behaviour) ───────────
        val_fraction = args.val_frac if args.val_frac is not None else cfg.get("val_fraction", 0.2)
        _, val_ds, _ = build_datasets(
            log_root     = args.log_root,
            seq_len      = seq_len,
            val_fraction = val_fraction,
            seed         = cfg.get("seed", 42),
            use_optitrack= use_optitrack,
            l0           = cfg.get("l0", 60.0),
            lu           = cfg.get("lu", 35.0),
            alpha_deg    = cfg.get("alpha_deg", 0.0),
            verbose      = True,
        )
        val_ds.scaler = scaler   # replace with checkpoint scaler

    if use_optitrack:
        print("[evaluate] use_optitrack=True  (input_size=9)")

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Predict
    pred_norm, true_norm = predict_all(model, val_loader, warmup_len, device)

    # Convert to mm
    pred_mm = scaler.inverse_y(pred_norm)
    true_mm = scaler.inverse_y(true_norm)

    print_metrics(pred_mm, true_mm)

    # Plots
    plot_scatter(pred_mm, true_mm, out_path=ckpt_dir / "scatter.png")
    plot_distance_hist(pred_mm, true_mm, out_path=ckpt_dir / "dist_hist.png")
    plot_loss_curve(ckpt_dir / "train_history.npy", out_path=ckpt_dir / "loss_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ResGRU error model")
    parser.add_argument("--checkpoint",  default="flowbot/residual_model/checkpoints",
                        help="Directory containing best_model.pt, scaler.pkl")
    parser.add_argument("--csv",         default=None,
                        help="Evaluate on a single CSV file (overrides --log_root).")
    parser.add_argument("--log_root",    default="data/task_logs")
    parser.add_argument("--seq_len",     type=int, default=None,
                        help="Override seq_len (default: use value from train_config.yaml)")
    parser.add_argument("--warmup_len",   type=int,   default=None,
                        help="Override warmup_len (default: use value from train_config.yaml)")
    parser.add_argument("--val_frac", type=float, default=None,
                        help="Fraction of log_root files used as val set. "
                             "Set to 1.0 to evaluate on all files in log_root "
                             "(useful for a new run not seen during training).")
    args = parser.parse_args()
    main(args)
