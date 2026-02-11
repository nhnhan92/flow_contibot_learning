#!/usr/bin/env python3
"""
Test if model uses visual information or just robot state

Usage:
    python train/test_visual_importance.py --checkpoint train/checkpoints/best_model.pt
"""

import sys
import os

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np

# Import after path is set
from train.eval import DiffusionPolicyInference
from train.dataset import PickPlaceDataset
from torch.utils.data import DataLoader

def test_visual_importance(checkpoint_path):
    """Test if model predictions change with different images"""

    print("="*60)
    print("   TESTING VISUAL IMPORTANCE")
    print("="*60)

    # Load model
    policy = DiffusionPolicyInference(checkpoint_path)

    # Load dataset
    dataset = PickPlaceDataset(
        dataset_path=policy.config['dataset_path'],
        obs_horizon=policy.config['obs_horizon'],
        pred_horizon=policy.config['pred_horizon'],
        action_horizon=policy.config['action_horizon'],
        image_size=tuple(policy.config['image_size']),
    )

    # Get a sample
    sample = dataset[100]

    print(f"\nSample keys: {sample.keys()}")

    obs_images = sample['obs_image'].unsqueeze(0).cuda()  # (1, obs_horizon, C, H, W)
    obs_states = sample['obs_state'].unsqueeze(0).cuda()  # (1, obs_horizon, state_dim)

    print(f"\nOriginal sample:")
    print(f"  obs_images shape: {obs_images.shape}")
    print(f"  obs_states shape: {obs_states.shape}")

    # Predict with original images
    with torch.no_grad():
        actions_original = policy.predict(obs_states, obs_images)

    # Predict with random noise images (same state)
    obs_images_noise = torch.randn_like(obs_images)
    with torch.no_grad():
        actions_noise = policy.predict(obs_states, obs_images_noise)

    # Predict with black images (same state)
    obs_images_black = torch.zeros_like(obs_images)
    with torch.no_grad():
        actions_black = policy.predict(obs_states, obs_images_black)

    # Calculate differences
    diff_noise = torch.mean(torch.abs(actions_original - actions_noise)).item()
    diff_black = torch.mean(torch.abs(actions_original - actions_black)).item()

    print(f"\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Prediction difference with noise images: {diff_noise:.6f}")
    print(f"Prediction difference with black images: {diff_black:.6f}")

    print(f"\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)

    if diff_noise < 0.01 and diff_black < 0.01:
        print("⚠️  Model is NOT using visual information!")
        print("    Predictions barely change with different images")
        print("    → Model relies only on robot state")
        print("\nPossible causes:")
        print("  - Vision encoder too weak")
        print("  - State information too dominant")
        print("  - Vision features not properly fused")
    else:
        print("✅ Model IS using visual information")
        print(f"   Predictions change significantly with different images")
        print(f"   Visual dependency: {max(diff_noise, diff_black):.6f}")

    return diff_noise, diff_black


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    test_visual_importance(args.checkpoint)
