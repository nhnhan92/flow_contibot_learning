#!/usr/bin/env python3
"""
Test script to verify the training pipeline for UR5e + Flowbot task.

State/Action space (9D):
    - UR5e TCP pose: x, y, z, rx, ry, rz  (6D)
    - Flowbot PWM signals: pwm1, pwm2, pwm3 (3D)

Usage:
    python system_verification/test_pipeline.py --config config/config_train_flowbot.yaml
"""

import os
import sys
import yaml
import torch

# Add parent directory to path
SYSVER_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.dirname(SYSVER_DIR)
sys.path.insert(0, LEARNING_DIR)

from train.dataset import PickPlaceDataset
from train.model import DiffusionPolicy


def test_dataset(config):
    """Test dataset loading"""
    print("\n" + "="*60)
    print("Testing Dataset...")
    print("="*60)

    try:
        dataset = PickPlaceDataset(
            dataset_path=config['dataset_path'],
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            action_horizon=config['action_horizon'],
            image_size=tuple(config['image_size']),
            exclude_episodes=config.get('exclude_episodes', []),
        )

        print(f"✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Number of episodes: {dataset.n_episodes}")

        # Get a sample
        sample = dataset[0]
        print(f"\n✅ Sample 0 loaded successfully!")
        print(f"   obs_state shape: {sample['obs_state'].shape}")
        print(f"   obs_image shape: {sample['obs_image'].shape}")
        print(f"   actions shape: {sample['actions'].shape}")

        # Verify dimensions
        assert sample['obs_state'].shape == (config['obs_horizon'], config['state_dim']), \
            f"Expected obs_state shape ({config['obs_horizon']}, {config['state_dim']}), got {sample['obs_state'].shape}"
        assert sample['obs_image'].shape == (config['obs_horizon'], 3, *config['image_size']), \
            f"Expected obs_image shape ({config['obs_horizon']}, 3, {config['image_size'][0]}, {config['image_size'][1]}), got {sample['obs_image'].shape}"
        assert sample['actions'].shape == (config['pred_horizon'], config['action_dim']), \
            f"Expected actions shape ({config['pred_horizon']}, {config['action_dim']}), got {sample['actions'].shape}"

        print(f"\n✅ All dimensions correct!")
        print(f"   state_dim:  {config['state_dim']} (UR5e TCP pose 6D + flowbot PWM 3D)")
        print(f"   action_dim: {config['action_dim']} (UR5e TCP pose 6D + flowbot PWM 3D)")

        # Check normalization stats
        print(f"\n✅ Normalization stats (Min-Max to [-1, 1]):")
        print(f"   State min  (XYZ): {dataset.state_min[:3]}")
        print(f"   State max  (XYZ): {dataset.state_max[:3]}")
        print(f"   State min  (PWM): {dataset.state_min[6:]}")
        print(f"   State max  (PWM): {dataset.state_max[6:]}")

        return True

    except Exception as e:
        print(f"\n❌ Dataset test failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model(config):
    """Test model creation and forward pass"""
    print("\n" + "="*60)
    print("Testing Model...")
    print("="*60)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Create model — mirror all params from config exactly as train.py does
        use_film = config.get('use_film_unet', True)
        use_ss   = config.get('use_spatial_softmax', None)
        model = DiffusionPolicy(
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            action_dim=config['action_dim'],
            state_dim=config['state_dim'],
            vision_feature_dim=config['vision_feature_dim'],
            state_feature_dim=config['state_feature_dim'],
            num_diffusion_iters=config['num_diffusion_iters'],
            num_inference_steps=config.get('num_inference_steps', 16),
            use_resnet=config.get('use_resnet', True),
            use_film_unet=use_film,
            film_step_embed_dim=config.get('film_step_embed_dim', 256),
            film_kernel_size=config.get('film_kernel_size', 5),
            use_spatial_softmax=use_ss,
            num_keypoints=config.get('num_keypoints', 32),
            crop_pad=config.get('random_crop_pad', 0),
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n✅ Model created successfully!")
        print(f"   Total parameters: {num_params:,}")

        # Test forward pass (training mode)
        batch_size = 4
        obs_state = torch.randn(batch_size, config['obs_horizon'], config['state_dim']).to(device)
        obs_image = torch.randn(batch_size, config['obs_horizon'], 3, *config['image_size']).to(device)
        actions = torch.randn(batch_size, config['pred_horizon'], config['action_dim']).to(device)

        print(f"\n✅ Testing training forward pass...")
        print(f"   Input shapes:")
        print(f"     obs_state: {obs_state.shape}  (B, obs_horizon, 9: pose6+pwm3)")
        print(f"     obs_image: {obs_image.shape}")
        print(f"     actions:   {actions.shape}  (B, pred_horizon, 9: pose6+pwm3)")

        loss = model(obs_state, obs_image, actions, train=True)

        print(f"\n✅ Training forward pass successful!")
        print(f"   Loss: {loss.item():.4f}")

        # Test backward pass
        print(f"\n✅ Testing backward pass...")
        loss.backward()
        print(f"   ✅ Backward pass successful!")

        # Test inference mode
        print(f"\n✅ Testing inference forward pass...")
        model.eval()
        with torch.no_grad():
            pred_actions = model(obs_state, obs_image, train=False)

        print(f"   ✅ Inference forward pass successful!")
        print(f"   Output shape: {pred_actions.shape}  (B, pred_horizon, 9: pose6+pwm3)")

        assert pred_actions.shape == (batch_size, config['pred_horizon'], config['action_dim']), \
            f"Expected output shape ({batch_size}, {config['pred_horizon']}, {config['action_dim']}), got {pred_actions.shape}"

        print(f"\n✅ All model tests passed!")

        return True

    except Exception as e:
        print(f"\n❌ Model test failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(config):
    """Test dataloader"""
    print("\n" + "="*60)
    print("Testing DataLoader...")
    print("="*60)

    try:
        from torch.utils.data import DataLoader

        dataset = PickPlaceDataset(
            dataset_path=config['dataset_path'],
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            action_horizon=config['action_horizon'],
            image_size=tuple(config['image_size']),
            exclude_episodes=config.get('exclude_episodes', []),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        print(f"✅ DataLoader created successfully!")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Total batches: {len(dataloader)}")

        print(f"\n✅ Loading one batch...")
        batch = next(iter(dataloader))

        print(f"   ✅ Batch loaded successfully!")
        print(f"   Batch shapes:")
        print(f"     obs_state: {batch['obs_state'].shape}")
        print(f"     obs_image: {batch['obs_image'].shape}")
        print(f"     actions:   {batch['actions'].shape}")

        return True

    except Exception as e:
        print(f"\n❌ DataLoader test failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline(config):
    """Test full training step"""
    print("\n" + "="*60)
    print("Testing Full Training Pipeline...")
    print("="*60)

    try:
        from torch.utils.data import DataLoader

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = PickPlaceDataset(
            dataset_path=config['dataset_path'],
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            action_horizon=config['action_horizon'],
            image_size=tuple(config['image_size']),
            exclude_episodes=config.get('exclude_episodes', []),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=min(config['batch_size'], len(dataset)),
            shuffle=True,
            num_workers=0,
        )

        use_film = config.get('use_film_unet', True)
        use_ss   = config.get('use_spatial_softmax', None)
        model = DiffusionPolicy(
            obs_horizon=config['obs_horizon'],
            pred_horizon=config['pred_horizon'],
            action_dim=config['action_dim'],
            state_dim=config['state_dim'],
            vision_feature_dim=config['vision_feature_dim'],
            state_feature_dim=config['state_feature_dim'],
            num_diffusion_iters=config['num_diffusion_iters'],
            num_inference_steps=config.get('num_inference_steps', 16),
            use_resnet=config.get('use_resnet', True),
            use_film_unet=use_film,
            film_step_embed_dim=config.get('film_step_embed_dim', 256),
            film_kernel_size=config.get('film_kernel_size', 5),
            use_spatial_softmax=use_ss,
            num_keypoints=config.get('num_keypoints', 32),
            crop_pad=config.get('random_crop_pad', 0),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay']),
        )

        print(f"✅ Pipeline components ready!")

        print(f"\n✅ Running one training step...")
        model.train()
        batch = next(iter(dataloader))

        obs_state = batch['obs_state'].to(device)
        obs_image = batch['obs_image'].to(device)
        actions = batch['actions'].to(device)

        optimizer.zero_grad()
        loss = model(obs_state, obs_image, actions, train=True)
        loss.backward()
        optimizer.step()

        print(f"   ✅ Training step successful!")
        print(f"   Loss: {loss.item():.4f}")

        print(f"\n✅ Running inference...")
        model.eval()
        with torch.no_grad():
            pred_actions = model(obs_state, obs_image, train=False)

        print(f"   ✅ Inference successful!")
        print(f"   Predicted actions shape: {pred_actions.shape}")
        print(f"   Pose (batch0, step0): {pred_actions[0, 0, :6]}")
        print(f"   PWM  (batch0, step0): {pred_actions[0, 0, 6:]}")

        return True

    except Exception as e:
        print(f"\n❌ Full pipeline test failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    import sys
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/config/config_train_flowbot.yaml', help='Config file')
    args = parser.parse_args()
    config_path = Path(LEARNING_DIR + args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*60)
    print("   TRAINING PIPELINE TEST - FLOWBOT")
    print("="*60)
    print(f"\nConfig: {args.config}")
    print(f"Dataset: {config['dataset_path']}")

    tests = [
        ("Dataset", test_dataset),
        ("Model", test_model),
        ("DataLoader", test_dataloader),
        ("Full Pipeline", test_full_pipeline),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func(config)

    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Ready to train!")
        print("\nRun training with:")
        print(f"  python train/train.py --config {args.config}")
    else:
        print("\n❌ Some tests failed. Please fix the issues before training.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
