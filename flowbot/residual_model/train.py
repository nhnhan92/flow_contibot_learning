"""
train.py  –  Two-phase training for ResGRU error model.

Phase 1 (warm-up): feed first `warmup_len` steps through the model with
    no gradient to prime the hidden states.
Phase 2 (rollout): compute MSE loss on the last step prediction.

Usage
-----
    python flowbot/residual_model/train.py \
        --log_root      data/task_logs \
        --out_dir       flowbot/residual_model/checkpoints \
        --seq_len       10 \
        --warmup_len    5 \
        --hidden        32 \
        --layers        2 \
        --dropout       0.1 \
        --epochs        500 \
        --batch         32 \
        --lr            1e-3

    # With OptiTrack positions as extra input features (input_size becomes 9):
    python flowbot/residual_model/train.py \
        --use_optitrack \
        --l0 60.0 --lu 35.0 --alpha_deg 0.0
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import time

import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from flowbot.residual_model.dataset import build_datasets
from flowbot.residual_model.model import ResGRU


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ResGRU,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    warmup_len: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)          # (B, T, 6), (B, 3)

        # Phase 1 – warm-up (no gradient, prime hidden states)
        if warmup_len > 0 and warmup_len < X.shape[1]:
            hidden = model.warmup(X[:, :warmup_len, :])
        else:
            hidden = None

        # Phase 2 – rollout with gradient
        x_roll = X[:, warmup_len:, :] if warmup_len > 0 else X
        pred, _ = model(x_roll, hidden)             # (B, 3)

        loss = criterion(pred, y)
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: ResGRU,
    loader: DataLoader,
    criterion: nn.Module,
    warmup_len: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if warmup_len > 0 and warmup_len < X.shape[1]:
            hidden = model.warmup(X[:, :warmup_len, :])
        else:
            hidden = None
        x_roll = X[:, warmup_len:, :] if warmup_len > 0 else X
        pred, _ = model(x_roll, hidden)
        total_loss += criterion(pred, y).item() * X.size(0)
    return total_loss / len(loader.dataset)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, scaler = build_datasets(
        log_root=args.log_root,
        seq_len=args.seq_len,
        val_fraction=args.val_fraction,
        use_optitrack=args.use_optitrack,
        l0=args.l0,
        lu=args.lu,
        alpha_deg=args.alpha_deg,
        verbose=True,
    )
    scaler.save(out_dir / "scaler.pkl")
    print(f"[train] Scaler saved → {out_dir / 'scaler.pkl'}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    input_size = 9 if args.use_optitrack else 6
    model = ResGRU(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        output_size=3,
    ).to(device)
    print(f"[train] ResGRU  input_size={input_size}  params: {model.n_params():,}")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val       = float("inf")
    es_counter     = 0                  # epochs without improvement
    history        = {"train_loss": [], "val_loss": [], "lr": []}
    t0             = time.time()

    warmup_len = min(args.warmup_len, args.seq_len - 1)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, warmup_len, device)
        val_loss   = evaluate(model, val_loader, criterion, warmup_len, device)
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

        if es_counter >= args.patience:
            print(f"\n[train] Early stopping at epoch {epoch}  (no improvement for {args.patience} epochs)")
            break

    # Save final model and training config
    torch.save(model.state_dict(), out_dir / "last_model.pt")
    cfg = vars(args).copy()
    cfg["input_size"]     = input_size
    cfg["best_val_loss"]  = best_val
    cfg["n_params"]       = model.n_params()
    with open(out_dir / "train_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    np.save(out_dir / "train_history.npy", history)

    print(f"\n[train] Done.  Best val loss: {best_val:.6f}")
    print(f"[train] Outputs saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResGRU error model")
    # ── Data / paths ──────────────────────────────────────────────────────────
    parser.add_argument("--log_root",     default="data/task_logs")
    parser.add_argument("--out_dir",      default="flowbot/residual_model/checkpoints")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed",         type=int,   default=42)
    # ── Sequence ──────────────────────────────────────────────────────────────
    parser.add_argument("--seq_len",      type=int,   default=10)
    parser.add_argument("--warmup_len",   type=int,   default=5,
                        help="Ticks used for hidden-state warm-up (no gradient)")
    # ── OptiTrack input features ───────────────────────────────────────────────
    parser.add_argument("--use_optitrack", action="store_true", default=False,
                        help="Append OptiTrack measured position as 3 extra input features")
    parser.add_argument("--l0",           type=float, default=82.0,
                        help="Flowbot l0 (mm) – used for optitrack frame transform")
    parser.add_argument("--lu",           type=float, default=13.5,
                        help="Flowbot lu (mm) – used for optitrack frame transform")
    parser.add_argument("--alpha_deg",    type=float, default=-30.0,
                        help="Manipulator mounting angle (deg) – optitrack frame transform")
    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--hidden",       type=int,   default=32)
    parser.add_argument("--layers",       type=int,   default=2)
    parser.add_argument("--dropout",      type=float, default=0.1)
    # ── Optimiser ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--patience",     type=int,   default=50,
                        help="Early-stopping: stop after this many epochs with no val-loss improvement")
    args = parser.parse_args()
    main(args)
