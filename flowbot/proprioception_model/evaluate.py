"""
evaluate.py — Evaluate a trained PropMLP / PlainMLP proprioception model.

Loads the checkpoint, runs inference on the validation split (or a separate
test directory), prints per-axis metrics, and saves scatter + error plots.

Usage
-----
    # Evaluate on the val split from training data (must match --seed used in train.py)
    python flowbot/proprioception_model/evaluate.py \
        --ckpt_dir flowbot/proprioception_model/checkpoints \
        --data_dir data/flow_tip

    # Evaluate on a completely separate test directory
    python flowbot/proprioception_model/evaluate.py \
        --ckpt_dir flowbot/proprioception_model/checkpoints \
        --test_dir data/flow_tip_test
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from flowbot.proprioception_model.dataset import (
    build_datasets, FlowTipDataset, StandardScaler, _load_csv,
    parse_feature_groups, FEATURE_NAMES,
)
from flowbot.proprioception_model.model import PropMLP, PlainMLP


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(ckpt_dir: Path, device: torch.device,
               model_file: str = "best_model.pt"):
    cfg_path = ckpt_dir / "train_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No train_config.yaml in {ckpt_dir}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    arch       = cfg.get("arch", "resmlp")
    input_size = cfg.get("input_size", cfg.get("input_size", 9))  # default to 9 if not in cfg

    if arch == "plainmlp":
        model = PlainMLP(
            input_size  = input_size,
            hidden_size = cfg["hidden"],
            num_layers  = cfg["num_blocks"],
            dropout     = cfg.get("dropout", 0.1),
            output_size = 3,
        )
    else:
        model = PropMLP(
            input_size  = input_size,
            hidden_size = cfg["hidden"],
            num_blocks  = cfg["num_blocks"],
            dropout     = cfg.get("dropout", 0.1),
            output_size = 3,
        )

    weights = torch.load(ckpt_dir / model_file, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.to(device).eval()
    return model, cfg


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device, y_scaler: StandardScaler):
    preds, trues = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).cpu().numpy())
        trues.append(y.numpy())
    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)
    return y_scaler.inverse_transform(y_pred), y_scaler.inverse_transform(y_true)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_pred_mm: np.ndarray, y_true_mm: np.ndarray) -> dict:
    err  = y_pred_mm - y_true_mm          # (N, 3)
    eucl = np.linalg.norm(err, axis=1)   # (N,)

    m: dict = {"eucl": eucl}
    for i, ax in enumerate(["x", "y", "z"]):
        e = err[:, i]
        ss_res = np.sum(e ** 2)
        ss_tot = np.sum((y_true_mm[:, i] - y_true_mm[:, i].mean()) ** 2)
        m[f"mae_{ax}"]  = float(np.mean(np.abs(e)))
        m[f"rmse_{ax}"] = float(np.sqrt(np.mean(e ** 2)))
        m[f"max_{ax}"]  = float(np.max(np.abs(e)))
        m[f"r2_{ax}"]   = float(1 - ss_res / (ss_tot + 1e-12))

    m["mae_3d"]  = float(np.mean(eucl))
    m["rmse_3d"] = float(np.sqrt(np.mean(eucl ** 2)))
    m["max_3d"]  = float(np.max(eucl))
    return m


def print_metrics(m: dict, n: int):
    print(f"\n{'='*58}")
    print(f"  Evaluation Results  (N = {n})")
    print(f"{'='*58}")
    print(f"  {'Axis':<5}  {'RMSE':>9}  {'MAE':>9}  {'Max':>9}  {'R²':>8}")
    print(f"  {'-'*56}")
    for ax in ["x", "y", "z"]:
        print(f"  {ax:<5}  {m[f'rmse_{ax}']:>8.3f}mm  {m[f'mae_{ax}']:>8.3f}mm"
              f"  {m[f'max_{ax}']:>8.3f}mm  {m[f'r2_{ax}']:>8.4f}")
    print(f"  {'-'*56}")
    print(f"  {'3D':<5}  {m['rmse_3d']:>8.3f}mm  {m['mae_3d']:>8.3f}mm"
          f"  {m['max_3d']:>8.3f}mm")
    print(f"{'='*58}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_results(y_pred_mm: np.ndarray, y_true_mm: np.ndarray,
                 m: dict, save_path: Path):
    err = y_pred_mm - y_true_mm
    axis_labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    ax_keys     = ["x", "y", "z"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"PropMLP Evaluation — 3D RMSE={m['rmse_3d']:.2f} mm  "
        f"MAE={m['mae_3d']:.2f} mm",
        fontsize=12,
    )

    # ── Row 0: Predicted vs True scatter ──────────────────────────────────────
    for i, (lbl, key) in enumerate(zip(axis_labels, ax_keys)):
        ax = axes[0, i]
        lo = min(y_true_mm[:, i].min(), y_pred_mm[:, i].min())
        hi = max(y_true_mm[:, i].max(), y_pred_mm[:, i].max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=1, alpha=0.5, label="ideal")
        ax.scatter(y_true_mm[:, i], y_pred_mm[:, i],
                   s=8, alpha=0.35, color="tab:blue", linewidths=0)
        ax.set_xlabel(f"True {lbl}")
        ax.set_ylabel(f"Predicted {lbl}")
        ax.set_title(f"{lbl}  R²={m[f'r2_{key}']:.3f}  "
                     f"RMSE={m[f'rmse_{key}']:.2f} mm")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7)

    # ── Row 1: Per-axis error histograms + 3D Euclidean ───────────────────────
    for i, (lbl, key) in enumerate(zip(axis_labels[:2], ax_keys[:2])):
        ax = axes[1, i]
        ax.hist(err[:, i], bins=40, color="tab:orange",
                alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="k", lw=1, linestyle="--")
        ax.set_xlabel(f"Error {lbl}")
        ax.set_ylabel("Count")
        ax.set_title(f"Error {lbl}  MAE={m[f'mae_{key}']:.2f} mm")

    # Z-axis error + 3D euclidean on the same right panel (twin axes)
    ax_z    = axes[1, 2]
    ax_eucl = ax_z.twinx()
    ax_z.hist(err[:, 2], bins=40, color="tab:purple",
              alpha=0.6, edgecolor="white", linewidth=0.3, label="Z error")
    ax_eucl.hist(m["eucl"], bins=40, color="tab:green",
                 alpha=0.45, edgecolor="white", linewidth=0.3, label="3D error")
    ax_z.axvline(0, color="k", lw=1, linestyle="--")
    ax_z.set_xlabel("Error (mm)")
    ax_z.set_ylabel("Count (Z)")
    ax_eucl.set_ylabel("Count (3D)")
    ax_z.set_title(f"Z error MAE={m['mae_z']:.2f} mm  |  "
                   f"3D MAE={m['mae_3d']:.2f} mm")
    lines1, labels1 = ax_z.get_legend_handles_labels()
    lines2, labels2 = ax_eucl.get_legend_handles_labels()
    ax_z.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[eval] Plot saved → {save_path}")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PropMLP proprioception model."
    )
    parser.add_argument("--ckpt_dir",   required=True,
                        help="Checkpoint directory (best_model.pt, train_config.yaml, scalers)")
    parser.add_argument("--data_dir",   default=None,
                        help="Training data directory — uses the val split (match --seed to train.py)")
    parser.add_argument("--test_dir",   default=None,
                        help="Separate test CSV directory — evaluates on ALL rows, no split")
    parser.add_argument("--model_file", default="best_model.pt",
                        choices=["best_model.pt", "last_model.pt"],
                        help="Which checkpoint to load (default: best_model.pt)")
    parser.add_argument("--batch",      type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed — must match the seed used in train.py to reproduce same val split")
    parser.add_argument("--states",
                        choices=["free_human", "with_human", "all"], default="all",
                        help="Filter rows by state (default: all)")
    parser.add_argument("--no_plot",    action="store_true",
                        help="Print metrics only, skip plots")
    args = parser.parse_args()

    if args.data_dir is None and args.test_dir is None:
        parser.error("Provide --data_dir (val split) or --test_dir (separate test set).")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)

    # ── Load model + scalers ──────────────────────────────────────────────────
    model, cfg = load_model(ckpt_dir, device, args.model_file)
    x_scaler    = StandardScaler.load(ckpt_dir / "scaler_x.pkl")
    y_scaler    = StandardScaler.load(ckpt_dir / "scaler_y.pkl")
    input_size  = cfg.get("input_size", 9)
    # feature_indices: new checkpoints store the list; fall back for old ones
    feature_indices = cfg.get("feature_indices")
    if feature_indices is None:
        feature_indices = parse_feature_groups(cfg.get("features", "flow,K,diff"))
    feat_names  = [FEATURE_NAMES[i] for i in feature_indices]
    n_params    = sum(p.numel() for p in model.parameters())
    print(f"[eval] {cfg.get('arch', 'resmlp').upper()}  ({n_params:,} params)"
          f"  features({input_size})={feat_names}"
          f"  from {ckpt_dir / args.model_file}")

    # ── Build eval dataset ────────────────────────────────────────────────────
    states = None if args.states == "all" else {args.states}
    if args.test_dir is not None:
        samples = []
        for p in sorted(Path(args.test_dir).rglob("*.csv")):
            samples.extend(_load_csv(p, verbose=True, states=states))
        if not samples:
            raise ValueError(f"No valid samples found in {args.test_dir}")
        eval_ds = FlowTipDataset(samples, augment=False,
                                 x_scaler=x_scaler, y_scaler=y_scaler,
                                 feature_indices=feature_indices)
        split_label = f"test set ({args.test_dir})"
    else:
        # Re-create the same val split using the training seed
        val_fraction    = cfg.get("val_fraction", 0.2)
        train_state_str = cfg.get("state", "all")
        train_state_set = None if train_state_str == "all" else {train_state_str}
        _, eval_ds, _, _ = build_datasets(
            data_dir     = args.data_dir,
            val_fraction = val_fraction,
            seed         = args.seed,
            verbose      = False,
            states       = states if states is not None else train_state_set,
            features     = cfg.get("features", "flow,K,diff"),
        )
        # Use checkpoint scalers to match training normalisation exactly
        eval_ds._x_scaler        = x_scaler
        eval_ds._y_scaler        = y_scaler
        eval_ds._feature_indices = feature_indices
        split_label = f"val split ({args.data_dir}, seed={args.seed})"

    print(f"[eval] Dataset: {len(eval_ds)} samples  [{split_label}]")

    loader = DataLoader(eval_ds, batch_size=args.batch, shuffle=False)

    # ── Inference + metrics ───────────────────────────────────────────────────
    y_pred_mm, y_true_mm = run_inference(model, loader, device, y_scaler)
    metrics = compute_metrics(y_pred_mm, y_true_mm)
    print_metrics(metrics, len(eval_ds))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        stem      = Path(args.model_file).stem
        save_path = ckpt_dir / f"eval_{stem}.png"
        plot_results(y_pred_mm, y_true_mm, metrics, save_path)


if __name__ == "__main__":
    main()
