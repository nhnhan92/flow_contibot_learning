"""
train.py — Training script for PropMLP proprioception model.

Feature groups (12-dim canonical vector, select any subset via --features):
    pwm   [0-2]  : pwm1, pwm2, pwm3
    flow  [3-5]  : flow1, flow2, flow3
    K     [6-8]  : K1=flow1/(pwm1+ε), K2, K3
    diff  [9-11] : diff12, diff23, diff13

Labels (3-dim): [opti_x_mm, opti_y_mm, opti_z_mm]
Augmentation  : flow *= α, α ~ Uniform(aug_alpha_min, 1.0)

Usage
-----
    python flowbot/proprioception_model/train.py --data_dir data/flow_tip
    python flowbot/proprioception_model/train.py \\
        --data_dir data/flow_tip --features flow,K,diff --hidden 256 --epochs 1000
    python flowbot/proprioception_model/train.py \\
        --data_dir data/flow_tip --features pwm,diff
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")          # works headless-safe; falls back silently
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from flowbot.proprioception_model.dataset import build_datasets, parse_feature_groups, FEATURE_NAMES
from flowbot.proprioception_model.model import PropMLP, PlainMLP


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval steps
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: PropMLP,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: PropMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total_loss += criterion(model(X), y).item() * X.size(0)
    return total_loss / len(loader.dataset)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    if args.load is not None:
        out_dir = Path(args.out_dir + f"/{args.state}"+f"/load{args.load}")
    else:
        out_dir = Path(args.out_dir + f"/{args.state}"+f"/freeload")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    states          = None if args.state == "all" else {args.state}
    feature_indices = parse_feature_groups(args.features)
    input_size      = len(feature_indices)
    feat_names      = [FEATURE_NAMES[i] for i in feature_indices]
    print(f"[train] Features ({input_size}): {feat_names}")

    train_ds, val_ds, x_scaler, y_scaler = build_datasets(
        data_dir=args.data_dir,
        val_fraction=args.val_fraction,
        aug_alpha_min=args.aug_alpha_min,
        seed=args.seed,
        verbose=True,
        states=states,
        features=args.features,
    )
    x_scaler.save(out_dir / "scaler_x.pkl")
    y_scaler.save(out_dir / "scaler_y.pkl")
    print(f"[train] Scalers saved → {out_dir}/scaler_x.pkl  scaler_y.pkl")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.arch == "plainmlp":
        model = PlainMLP(
            input_size=input_size,
            hidden_size=args.hidden,
            num_layers=args.num_blocks,
            dropout=args.dropout,
            output_size=3,
        ).to(device)
        print(f"[train] PlainMLP  input={input_size}  hidden={args.hidden}  layers={args.num_blocks}"
              f"  params: {model.n_params():,}")
    else:
        model = PropMLP(
            input_size=input_size,
            hidden_size=args.hidden,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            output_size=3,
        ).to(device)
        print(f"[train] PropMLP  input={input_size}  hidden={args.hidden}  blocks={args.num_blocks}"
              f"  params: {model.n_params():,}")

    # criterion = nn.HuberLoss(delta=1.0)  
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    # ── Live loss plot ────────────────────────────────────────────────────────
    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (log scale)")
        ax.set_yscale("log")
        line_train, = ax.plot([], [], label="train", color="tab:blue")
        line_val,   = ax.plot([], [], label="val",   color="tab:orange")
        ax.legend()
        fig.tight_layout()
        fig.canvas.manager.set_window_title("PropMLP — Loss Curve")
        plt.show(block=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val   = float("inf")
    es_counter = 0
    history    = {"train_loss": [], "val_loss": [], "lr": []}
    t0         = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimiser.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        if val_loss < best_val:
            best_val   = val_loss
            es_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            tag = " ← best"
        else:
            es_counter += 1
            tag = f" (no improve {es_counter}/{args.patience})"

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"[train] Epoch {epoch:4d}/{args.epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"lr={lr:.2e}  ({elapsed:.0f}s){tag}"
            )
            if not args.no_plot:
                epochs_so_far = range(1, len(history["train_loss"]) + 1)
                line_train.set_data(epochs_so_far, history["train_loss"])
                line_val.set_data(epochs_so_far, history["val_loss"])
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                plt.pause(0.001)

        if es_counter >= args.patience:
            print(f"\n[train] Early stopping at epoch {epoch}"
                  f"  (no improvement for {args.patience} epochs)")
            break

    # ── Save outputs ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), out_dir / "last_model.pt")

    cfg = vars(args).copy()
    cfg["feature_indices"] = feature_indices   # list of ints — used by evaluate/demo
    cfg["input_size"]      = input_size
    cfg["best_val_loss"]   = best_val
    cfg["n_params"]        = model.n_params()
    with open(out_dir / "train_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    np.save(out_dir / "train_history.npy", history)

    if not args.no_plot:
        plot_path = out_dir / "loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"[train] Loss curve saved → {plot_path}")
        plt.ioff()
        plt.show()

    print(f"\n[train] Done.  Best val loss: {best_val:.6f}")
    print(f"[train] Outputs saved to {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PropMLP proprioception model")

    # Data
    parser.add_argument("--data_dir",      required=True,
                        help="Folder containing collect_flow_tip CSVs")
    parser.add_argument("--out_dir",       default="flowbot/proprioception_model/checkpoints")
    parser.add_argument("--load", type=str, default=None,
                        help="Load at the End effector, None for no load")
    parser.add_argument("--val_fraction",  type=float, default=0.2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--state",
                        choices=["free_human", "with_human", "all"], default="free_human",
                        help="Which state rows to train on (default: all)")

    # Augmentation
    parser.add_argument("--aug_alpha_min", type=float, default=0.6,
                        help="Lower bound for flow-scale augmentation α (Option C)")

    # Features
    parser.add_argument("--features", default="pwm,flow,K",
                        help="Comma-separated feature groups to use. "
                             "Groups: pwm, flow, K, diff  (canonical 12-dim order preserved). "
                             "Examples: 'flow,K,diff'")

    # Model
    parser.add_argument("--arch",       choices=["resmlp", "plainmlp"], default="resmlp",
                        help="Model architecture: resmlp (default) or plainmlp")
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--num_blocks", type=int,   default=3,
                        help="Residual blocks (resmlp) or hidden layers (plainmlp)")
    parser.add_argument("--dropout",    type=float, default=0.05)

    # Optimiser
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--batch",         type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=100,
                        help="Early-stopping patience (epochs without val improvement)")
    parser.add_argument("--no_plot",       action="store_true",
                        help="Disable live loss plot (useful for headless/remote runs)")

    main(parser.parse_args())
