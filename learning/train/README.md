# Diffusion Policy Training Pipeline

Complete training pipeline for Pick-Place task using Diffusion Policy.

## Overview

This directory contains:
- `dataset.py`: Dataset loader for zarr format
- `model.py`: Diffusion Policy architecture (Vision Encoder + State Encoder + Diffusion U-Net)
- `train.py`: Main training script
- `eval.py`: Evaluation and inference script
- `config.yaml`: Training configuration

## Quick Start

### 1. Collect Data

First, collect demonstration data using the data collection script:

```bash
# With camera (recommended for vision-based policy)
python scripts/collect_demos_with_camera.py \
    -o data/demos \
    --robot_ip 192.168.11.20 \
    --frequency 10

# Without camera (state-only)
python scripts/collect_demos.py \
    -o data/demos \
    --robot_ip 192.168.11.20
```

Controls:
- Press `C` to start recording an episode
- Perform the pick-place task with SpaceMouse
- Press `S` to stop and save the episode
- Repeat for 50-100 episodes

### 2. Train the Model

```bash
python train/train.py --config train/config.yaml
```

Optional arguments:
- `--resume <checkpoint>`: Resume training from checkpoint
- `--device cuda`: Use GPU (default) or `cpu`
- `--wandb_project <name>`: W&B project name
- `--wandb_run_name <name>`: W&B run name

### 3. Evaluate

```bash
# Evaluate on validation set
python train/eval.py \
    --checkpoint train/checkpoints/best_model.pt \
    --mode eval

# Visualize predictions
python train/eval.py \
    --checkpoint train/checkpoints/best_model.pt \
    --mode visualize \
    --sample_idx 0
```

## Configuration

Key parameters in `config.yaml`:

### Data Parameters
- `obs_horizon: 2` - Number of observation frames (temporal context)
- `pred_horizon: 16` - Number of action steps to predict (~1.6s at 10Hz)
- `action_horizon: 8` - Number of actions to execute before replanning
- `image_size: [96, 96]` - Image resolution (smaller = faster)

### Model Parameters
- `action_dim: 7` - Robot pose (x, y, z, rx, ry, rz) + gripper (1)
- `state_dim: 7` - Robot pose (6) + gripper (1)
- `num_diffusion_iters: 100` - Diffusion steps (higher = better quality, slower)

### Training Parameters
- `batch_size: 64` - Reduce if GPU memory limited
- `learning_rate: 1e-4` - AdamW learning rate
- `num_epochs: 1000` - Total training epochs
- `ema_decay: 0.999` - Exponential moving average

## Architecture

```
Observations → Encoders → Diffusion Model → Actions
    ↓              ↓              ↓              ↓
Camera (2 frames)  Vision CNN     1D U-Net    16 future actions
Robot State    →   State MLP   →  + Noise  →  (x,y,z,rx,ry,rz,gripper)
Gripper        →                   Scheduler
```

### Vision Encoder
- Input: (B, obs_horizon, 3, 96, 96)
- CNN with GroupNorm + Mish
- Output: (B, 256) visual features

### State Encoder
- Input: (B, obs_horizon * 7)
- 2-layer MLP
- Output: (B, 64) state features

### Diffusion U-Net
- Input: Noisy actions + timestep + conditioning
- 1D U-Net with skip connections
- Output: Predicted noise

## Data Format

The zarr dataset structure:
```
dataset.zarr/
├── data/
│   ├── timestamp        # (N,) float64
│   ├── robot_eef_pose   # (N, 6) float64 - TCP pose
│   ├── robot_joint      # (N, 6) float64 - Joint angles
│   ├── gripper_position # (N,) float64 - Gripper state [0, 1]
│   ├── action           # (N, 6) float64 - Target pose
│   └── camera_0         # (N, H, W, 3) uint8 - RGB images
└── meta/
    └── episode_ends     # (E,) int64 - Episode boundaries
```

## Training Tips

### For Limited Data (< 50 episodes)
- Reduce model size: `vision_feature_dim: 128`, `state_feature_dim: 32`
- Increase regularization: `weight_decay: 1e-5`
- Use data augmentation (TODO: add to dataset.py)

### For GPU Memory Issues
- Reduce `batch_size` (try 32 or 16)
- Reduce `image_size` (try [64, 64])
- Reduce `num_diffusion_iters` during training (50-100 is fine)

### For Better Performance
- Collect more diverse demonstrations (50-100 episodes)
- Increase `obs_horizon` for more temporal context (2-4)
- Use learning rate warmup and scheduling
- Enable W&B logging to monitor training

## Inference on Robot

To deploy the trained policy on the robot, create a deployment script:

```python
from train.eval import DiffusionPolicyInference
import torch

# Load policy
policy = DiffusionPolicyInference('train/checkpoints/best_model.pt')

# In control loop:
# obs_state = torch.tensor([...])  # (obs_horizon, 7)
# obs_image = torch.tensor([...])  # (obs_horizon, 3, 96, 96)

# Predict actions
actions = policy.predict(obs_state, obs_image)  # (pred_horizon, 7)

# Execute first action_horizon steps
for i in range(action_horizon):
    robot_pose = actions[i, :6]  # (x, y, z, rx, ry, rz)
    gripper_pos = actions[i, 6]   # gripper position [0, 1]
    robot.execute(robot_pose)
    gripper.set_position(gripper_pos)
```

## Troubleshooting

### Training Loss Not Decreasing
- Check data quality (use eval.py to visualize)
- Reduce learning rate
- Increase batch size
- Check data normalization

### Out of Memory
- Reduce batch_size
- Reduce image_size
- Reduce num_workers
- Use mixed precision training (TODO)

### Inference Too Slow
- Reduce num_diffusion_iters (use DDIM scheduler for faster inference)
- Use smaller image_size
- Optimize with TorchScript or ONNX

## References

- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [Original Implementation](https://github.com/real-stanford/diffusion_policy)
