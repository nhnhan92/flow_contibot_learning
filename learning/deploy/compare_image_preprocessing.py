#!/usr/bin/env python3
"""
Compare image preprocessing between collection, training, and deployment

This helps verify that images are processed the same way throughout the pipeline.

Usage:
    python deploy/compare_image_preprocessing.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import zarr
import matplotlib.pyplot as plt

# Image size from config
IMAGE_SIZE = (128, 128)


def show_collection_image():
    """Show an image from the collected dataset (as saved during collection)"""
    print("\n" + "="*60)
    print("1. COLLECTION STAGE")
    print("="*60)

    # Load from zarr
    zarr_path = 'data/real_data/dataset.zarr'
    root = zarr.open(zarr_path, 'r')

    # Get a sample image from first episode
    sample_idx = 100  # Random sample
    raw_image = root['data/camera_0'][sample_idx]

    print(f"Raw image shape: {raw_image.shape}")
    print(f"Raw image dtype: {raw_image.dtype}")
    print(f"Raw image range: [{raw_image.min()}, {raw_image.max()}]")

    return raw_image


def preprocess_for_training(raw_image):
    """Preprocess image as done during training (dataset.py)"""
    print("\n" + "="*60)
    print("2. TRAINING PREPROCESSING (dataset.py)")
    print("="*60)

    # Resize (dataset.py line 185)
    img_resized = cv2.resize(raw_image, IMAGE_SIZE)
    print(f"After resize: shape={img_resized.shape}, range=[{img_resized.min()}, {img_resized.max()}]")

    # Normalize to [-1, 1] (dataset.py line 187)
    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
    print(f"After normalize: range=[{img_normalized.min():.3f}, {img_normalized.max():.3f}]")

    # Transpose to CHW (dataset.py line 192)
    img_chw = img_normalized.transpose(2, 0, 1)
    print(f"After transpose: shape={img_chw.shape} (CHW format)")

    return img_resized, img_normalized, img_chw


def preprocess_for_deployment(raw_image):
    """Preprocess image as done during deployment (deploy_real_robot.py)"""
    print("\n" + "="*60)
    print("3. DEPLOYMENT PREPROCESSING (deploy_real_robot.py)")
    print("="*60)

    # Resize (line 148)
    image = cv2.resize(raw_image, IMAGE_SIZE)
    print(f"After resize: shape={image.shape}, range=[{image.min()}, {image.max()}]")

    # Normalize to [-1, 1] (lines 155-156)
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    print(f"After normalize: range=[{image.min():.3f}, {image.max():.3f}]")

    # Transpose to CHW (line 157)
    image = np.transpose(image, (2, 0, 1))
    print(f"After transpose: shape={image.shape} (CHW format)")

    return image


def visualize_comparison():
    """Visualize all stages"""
    print("\n" + "="*60)
    print("COMPARING IMAGE PREPROCESSING PIPELINE")
    print("="*60)

    # 1. Collection
    raw_image = show_collection_image()

    # 2. Training preprocessing
    train_resized, train_normalized, train_chw = preprocess_for_training(raw_image)

    # 3. Deployment preprocessing
    deploy_chw = preprocess_for_deployment(raw_image)

    # Compare final outputs
    print("\n" + "="*60)
    print("4. COMPARISON")
    print("="*60)

    # Convert CHW back to HWC for comparison
    train_final = train_chw.transpose(1, 2, 0)
    deploy_final = deploy_chw.transpose(1, 2, 0)

    difference = np.abs(train_final - deploy_final)
    max_diff = difference.max()
    mean_diff = difference.mean()

    print(f"\nTraining output shape: {train_chw.shape}")
    print(f"Deployment output shape: {deploy_chw.shape}")
    print(f"\nDifference statistics:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-5:
        print(f"\n✅ IDENTICAL! Training and deployment preprocessing are the same!")
    else:
        print(f"\n⚠️  DIFFERENT! Training and deployment preprocessing differ!")
        print(f"   This could cause model performance issues!")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Raw, Training preprocessed, Deployment preprocessed
    axes[0, 0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'1. Collection\nOriginal {raw_image.shape}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(train_resized, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'2. Training\nResized {train_resized.shape}')
    axes[0, 1].axis('off')

    # Show deployment resized (same as training resized)
    deploy_resized = cv2.resize(raw_image, IMAGE_SIZE)
    axes[0, 2].imshow(cv2.cvtColor(deploy_resized, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'3. Deployment\nResized {deploy_resized.shape}')
    axes[0, 2].axis('off')

    # Row 2: Normalized versions
    # Denormalize for visualization: (x + 1) * 127.5
    train_vis = ((train_normalized + 1.0) * 127.5).astype(np.uint8)
    deploy_vis = ((deploy_final + 1.0) * 127.5).astype(np.uint8)

    axes[1, 0].imshow(cv2.cvtColor(train_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'Training Normalized\n(denormalized for display)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(deploy_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Deployment Normalized\n(denormalized for display)')
    axes[1, 1].axis('off')

    # Show difference
    diff_vis = (difference * 255).astype(np.uint8)
    axes[1, 2].imshow(diff_vis)
    axes[1, 2].set_title(f'Absolute Difference\nMax: {max_diff:.6f}')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('image_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: image_preprocessing_comparison.png")

    # Print normalization formulas
    print("\n" + "="*60)
    print("NORMALIZATION FORMULAS")
    print("="*60)
    print("\nTraining (dataset.py line 187):")
    print("  normalized = (img / 127.5) - 1.0")
    print("  → Maps [0, 255] to [-1, 1]")

    print("\nDeployment (deploy_real_robot.py lines 155-156):")
    print("  normalized = (img / 255.0 - 0.5) / 0.5")
    print("  → = (img / 255.0) / 0.5 - 0.5 / 0.5")
    print("  → = img / 127.5 - 1.0")
    print("  → Maps [0, 255] to [-1, 1]")

    print("\n✅ Both formulas are mathematically IDENTICAL!")


if __name__ == '__main__':
    try:
        visualize_comparison()
        print("\n" + "="*60)
        print("DONE!")
        print("="*60)
        print("\nOpen 'image_preprocessing_comparison.png' to see the visual comparison.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
