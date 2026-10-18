#!/usr/bin/env python3
"""
train_object_classifier.py — train the standalone object-identity classifier
(see object_classifier.py) on a zarr dataset labeled by label_object_id.py
or concat_datasets.py's --object_names.

Fully decoupled from the diffusion policy: this is a plain supervised
image-classification problem (N known, visually-distinct objects), trained
on the SAME camera_0 frames the policy itself trains on, using the labels
you already implicitly have from collecting each object's episodes
separately. Reuses the SAME image_size/crop_scale/crop_x/crop_y as whatever
policy training config you point --config at, so the classifier sees
exactly the same crop the deployed policy (and camera) will show it.

One frame per training sample, drawn from the first --frames_per_episode
frames of each episode (classification should be easy/early, before the arm
has moved far -- see the class docstring in object_classifier.py for why
object-identity is computed ONCE near the start of a deploy episode, not
every control step). Split by EPISODE (not frame) and stratified per object,
so validation accuracy isn't inflated by near-duplicate frames leaking
between train and val.

Usage:
    python learning/train/train_object_classifier.py \\
        --dataset data/demo_data/Task3/dataset.zarr \\
        --config config/config_train_flowbot_ur5.yaml \\
        --output_dir checkpoints/object_classifier \\
        --epochs 30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import zarr
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.object_classifier import ObjectClassifier, preprocess_image


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class ObjectIdFrameDataset(Dataset):
    """One sample = one early frame of one episode, labeled with that
    episode's object_id. augment=True applies light train-time augmentation
    (random crop jitter, color jitter, small rotation -- NOT horizontal
    flip, which could corrupt real spatial cues; see train_object_classifier.py docstring)."""

    def __init__(self, zarr_root, episode_indices, episode_ends, episode_object_id,
                image_size, crop_scale, crop_x, crop_y, frames_per_episode, augment):
        self.zarr_root = zarr_root
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.augment = augment

        self.samples = []  # (frame_idx, label)
        for ep_idx in episode_indices:
            start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
            end = int(episode_ends[ep_idx])
            ep_len = end - start
            n_frames = min(frames_per_episode, ep_len)
            label = int(episode_object_id[ep_idx])
            for i in range(n_frames):
                self.samples.append((start + i, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_idx, label = self.samples[idx]
        img_raw = np.array(self.zarr_root['data/camera_0'][frame_idx])  # (H,W,3) uint8

        crop_scale, crop_x, crop_y = self.crop_scale, self.crop_x, self.crop_y
        if self.augment:
            # Small random jitter around the configured crop (not a fixed
            # offset every time) -- cheap, effective augmentation for a
            # modest per-class episode count.
            crop_scale = crop_scale * np.random.uniform(0.95, 1.08)
            crop_x = np.clip(crop_x + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
            crop_y = np.clip(crop_y + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

        img = preprocess_image(img_raw, self.image_size, crop_scale, crop_x, crop_y)  # (3,H,W) float32 [-1,1]

        if self.augment:
            # Light color jitter (brightness/contrast) directly in [-1,1] space.
            img = img * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.1, 0.1)
            img = np.clip(img, -1.0, 1.0)
            if np.random.rand() < 0.5:
                # Small rotation (+/-10deg) via simple nearest-neighbor-free
                # approach: skip if no scipy/cv2 dependency desired here --
                # rotation is the one augmentation most worth adding later
                # with cv2.warpAffine if val accuracy needs the extra push;
                # omitted by default to keep this script dependency-light.
                pass

        return torch.from_numpy(img.copy()).float(), label


def _stratified_episode_split(episode_object_id, val_fraction, seed):
    """Hold out val_fraction of episodes per object (not overall), so rare
    objects aren't accidentally left with zero validation episodes."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for obj_id in np.unique(episode_object_id):
        eps = np.where(episode_object_id == obj_id)[0]
        rng.shuffle(eps)
        n_val = max(1, int(round(len(eps) * val_fraction))) if len(eps) > 1 else 0
        val_idx.extend(eps[:n_val].tolist())
        train_idx.extend(eps[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


# ──────────────────────────────────────────────────────────────────────────────
# Train / eval
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_objects):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    confusion = np.zeros((num_objects, num_objects), dtype=np.int64)  # rows=true, cols=pred
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(1)
        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        n += x.size(0)
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            confusion[t, p] += 1
    return total_loss / n, correct / n, confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='Path to the labeled dataset.zarr')
    parser.add_argument('--config', type=str, default=None,
                        help="Policy training YAML to reuse image_size/crop_scale/crop_x/crop_y from "
                             "(recommended, keeps the classifier's crop identical to the policy's). "
                             "Overridden by any of --image_size/--crop_scale/--crop_x/--crop_y given explicitly.")
    parser.add_argument('--image_size', type=int, nargs=2, default=None, metavar=('H', 'W'))
    parser.add_argument('--crop_scale', type=float, default=None)
    parser.add_argument('--crop_x', type=float, default=None)
    parser.add_argument('--crop_y', type=float, default=None)
    parser.add_argument('--output_dir', '-o', required=True)
    parser.add_argument('--frames_per_episode', type=int, default=5,
                        help='Early frames sampled per episode as training images (default 5)')
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # ── Resolve crop settings: --config defaults, explicit CLI args override ──
    image_size, crop_scale, crop_x, crop_y = (212, 228), 1.5, 0.5, 0.5
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        image_size = tuple(cfg.get('image_size', image_size))
        crop_scale = cfg.get('crop_scale', crop_scale)
        crop_x = cfg.get('crop_x', crop_x)
        crop_y = cfg.get('crop_y', crop_y)
    if args.image_size is not None:
        image_size = tuple(args.image_size)
    if args.crop_scale is not None:
        crop_scale = args.crop_scale
    if args.crop_x is not None:
        crop_x = args.crop_x
    if args.crop_y is not None:
        crop_y = args.crop_y
    print(f"Crop settings: image_size={image_size}, crop_scale={crop_scale}, "
          f"crop_x={crop_x}, crop_y={crop_y}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    zarr_root = zarr.open(args.dataset, mode='r')
    if 'meta' not in zarr_root or 'episode_object_id' not in zarr_root['meta'] \
            or 'object_names' not in zarr_root['meta']:
        raise ValueError(
            f"{args.dataset} has no meta/episode_object_id or meta/object_names -- "
            f"label it first with label_object_id.py (or concat_datasets.py's --object_names)."
        )
    episode_ends = zarr_root['meta/episode_ends'][:]
    episode_object_id = np.asarray(zarr_root['meta/episode_object_id'][:], dtype=np.int64)
    object_names = [str(n) for n in zarr_root['meta/object_names'][:]]
    num_objects = len(object_names)
    n_episodes = len(episode_ends)
    print(f"Dataset: {n_episodes} episodes, {num_objects} objects: {object_names}")

    train_ep, val_ep = _stratified_episode_split(episode_object_id, args.val_fraction, args.seed)
    print(f"Episode split: {len(train_ep)} train, {len(val_ep)} val (stratified per object)")

    train_ds = ObjectIdFrameDataset(zarr_root, train_ep, episode_ends, episode_object_id,
                                    image_size, crop_scale, crop_x, crop_y,
                                    args.frames_per_episode, augment=True)
    val_ds = ObjectIdFrameDataset(zarr_root, val_ep, episode_ends, episode_object_id,
                                  image_size, crop_scale, crop_x, crop_y,
                                  args.frames_per_episode, augment=False)
    print(f"Samples: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── Model ───────────────────────────────────────────────────────────────
    model = ObjectClassifier(num_objects, pretrained=True, dropout=args.dropout).to(device)
    print(f"Model params: {model.n_params():,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, confusion = evaluate(model, val_loader, criterion, device, num_objects)
        print(f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'object_names': object_names,
                'image_size': image_size,
                'crop_scale': crop_scale,
                'crop_x': crop_x,
                'crop_y': crop_y,
            }, out_dir / 'best_model.pt')
            print(f"  -> new best (val_acc={val_acc:.3f}), saved {out_dir / 'best_model.pt'}")

    # Final confusion matrix (best model would need reloading to print exactly
    # at best_val_acc; this prints the LAST epoch's, which is informative
    # regardless -- sorted by object for readability)
    print(f"\nFinal epoch confusion matrix (rows=true, cols=predicted):")
    header = "        " + " ".join(f"{n[:8]:>8s}" for n in object_names)
    print(header)
    for i, name in enumerate(object_names):
        row = " ".join(f"{confusion[i, j]:>8d}" for j in range(num_objects))
        print(f"{name[:8]:>8s} {row}")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()
