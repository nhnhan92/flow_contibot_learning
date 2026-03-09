# Flowbot Diffusion Policy — Learning Module

Diffusion Policy training and deployment for the **UR5e + Flowbot** soft manipulator system.
The robot has 3 pneumatic valves controlled via PWM signals.

## System Overview

| Component | Details |
|-----------|---------|
| Robot arm | UR5e (6-DOF), controlled via RTDE |
| Soft gripper | Flowbot (3 pneumatic valves, PWM via Arduino) |
| Camera | Intel RealSense D455 |
| Input device | 3DConnexion SpaceMouse (for teleoperation & data collection) |

**State / Action space (9D):**
- UR5e TCP pose: `x, y, z, rx, ry, rz` (6D)
- Flowbot PWM signals: `pwm1, pwm2, pwm3` (3D)

---

## Folder Structure

```
learning/
├── config/
│   └── config_train_flowbot.yaml    # All training hyperparameters
│
├── hardware/                         # Hardware interface drivers
│   ├── ur5e_rtde.py                  #   UR5e RTDE control wrapper
│   ├── realsense_camera.py           #   RealSense D455 camera interface
│   ├── spacemouse.py                 #   3DConnexion SpaceMouse driver
│   ├── flowbot.py                    #   Flowbot PWM control via Arduino
│   └── __init__.py
│
├── scripts/                          # Data collection scripts
│   ├── collect_demos_with_camera.py  #   Main demo collection (camera + robot + flowbot)
│   ├── collect_demos.py              #   Demo collection without camera
│   └── debug_spacemouse_axes.py      #   Debug SpaceMouse axis mapping
│
├── train/                            # Training pipeline
│   ├── dataset.py                    #   Zarr dataset loader & normalization
│   ├── model.py                      #   Diffusion Policy (FiLM UNet + ResNet18)
│   ├── train.py                      #   Main training script
│   ├── eval.py                       #   Inference wrapper (DiffusionPolicyInference)
│   ├── EMA_model.py                  #   Exponential Moving Average (UNet-only)
│   ├── analyze_dataset.py            #   Dataset statistics analysis
│   ├── evaluate_model.py             #   Offline model evaluation
│   ├── visualize_dataset.py          #   Dataset visualization
│   └── __init__.py
│
├── deploy/                           # Real-robot deployment scripts
│   ├── deploy_real_robot.py          #   Main closed-loop control loop
│   ├── move_to_start.py              #   Move robot to start position
│   ├── debug_pwm_predictions.py      #   Debug PWM predictions from model
│   ├── debug_camera_view.py          #   Debug camera feed
│   ├── visualize_predictions.py      #   Visualize model action predictions
│   └── compare_image_preprocessing.py
│
└── system_verification/              # Hardware & pipeline tests
    ├── test_camera.py                #   RealSense camera stream test
    ├── test_robot.py                 #   UR5e RTDE connection test
    ├── test_spacemouse.py            #   SpaceMouse input test
    ├── test_teleop.py                #   Full teleoperation test
    ├── test_connections.py           #   All hardware connections test
    ├── test_pipeline.py              #   Training pipeline test
    ├── test_checkpoint_on_episode.py #   Verify checkpoint on training episode
    ├── test_visual_importance.py     #   Test whether model uses visual input
    └── __init__.py
```

---

## Quickstart

All commands should be run from the `learning/` directory.

### 1 — Verify Hardware

Before collecting data or deploying, verify all hardware connections:

```bash
cd ~/Desktop/flow_contibot_learning/learning

# Test camera
python system_verification/test_camera.py

# Test robot connection
python system_verification/test_robot.py --robot_ip 192.168.1.100

# Test SpaceMouse
python system_verification/test_spacemouse.py

# Test full teleoperation
python system_verification/test_teleop.py --robot_ip 192.168.1.100

# Test all connections at once
python system_verification/test_connections.py \
    --robot_ip 192.168.1.100 --flowbot_port /dev/ttyACM0
```

### 2 — Collect Demonstrations

```bash
python scripts/collect_demos_with_camera.py \
    --robot_ip 192.168.1.100 \
    --output_dir data/demo_data \
    --n_episodes 20
```

### 3 — Verify Training Pipeline

```bash
python system_verification/test_pipeline.py --config config/config_train_flowbot.yaml
```

Expected output:
```
✅ PASS - Dataset
✅ PASS - Model
✅ PASS - DataLoader
✅ PASS - Full Pipeline

🎉 All tests passed! Ready to train!
```

### 4 — Train

```bash
python train/train.py --config config/config_train_flowbot.yaml

# Resume from checkpoint
python train/train.py --config config/config_train_flowbot.yaml \
    --resume train/checkpoints/checkpoint_epoch_500.pt

# With W&B logging
python train/train.py --config config/config_train_flowbot.yaml \
    --wandb_project flowbot-diffusion
```

Checkpoints are saved to `train/checkpoints/` every `save_every` epochs.
The best model (by EMA validation loss) is saved as `train/checkpoints/best_model.pt`.

### 5 — Verify Checkpoint

```bash
python system_verification/test_checkpoint_on_episode.py \
    --checkpoint train/checkpoints/best_model.pt --episode 0

python system_verification/test_visual_importance.py \
    --checkpoint train/checkpoints/best_model.pt
```

### 6 — Deploy on Robot

```bash
# Move robot to start position first
python deploy/move_to_start.py --robot_ip 192.168.1.100

# Run closed-loop deployment
python deploy/deploy_real_robot.py \
    --checkpoint train/checkpoints/best_model.pt \
    --robot_ip 192.168.1.100 \
    --flowbot_port /dev/ttyACM0
```

---

## Model Architecture

```
Observation:
  Camera:      obs_horizon × RGB (216×288) → ResNet18 + SpatialSoftmax → 64D/frame
  Robot state: obs_horizon × 9D (pose 6D + PWM 3D) → MLP → 128D/frame

Concatenated observation → 384D context

Diffusion:
  ConditionalUNet1D (FiLM-conditioned, Stanford-style)
  DDPM training  (100 steps, squaredcos_cap_v2 schedule)
  DDIM inference (16 steps — ~6× faster than DDPM)

Output:
  pred_horizon=16 × 9D actions (pose 6D + PWM 3D)
  Execute first action_horizon=8 steps, then replan
```

---

## Key Config Options

Edit `config/config_train_flowbot.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `obs_horizon` | 2 | Observation frames fed to model |
| `pred_horizon` | 16 | Actions predicted per forward pass |
| `action_horizon` | 8 | Actions executed before replanning |
| `num_epochs` | 3000 | Total training epochs |
| `warmup_epochs` | 500 | Linear LR warmup epochs |
| `learning_rate` | 5.0e-5 | Peak learning rate (AdamW) |
| `batch_size` | 64 | Reduce to 32 if OOM |
| `random_crop_pad` | 20 | Vision augmentation (0 = disabled) |
| `early_stopping_patience` | 200 | Epochs without improvement before stopping |
| `use_wandb` | false | Enable Weights & Biases logging |

---

## Troubleshooting

**Training loss not decreasing**
- Verify dataset: `python train/analyze_dataset.py`
- Try reducing `learning_rate` to `1.0e-5`

**CUDA out of memory**
- Reduce `batch_size` (try 32 or 16)
- Reduce `num_workers` to 2 or 0

**Robot connection failed**
- Confirm robot IP with `ping <robot_ip>`
- Check that the robot is in Remote Control mode

**Flowbot not responding**
- Check Arduino is connected: `ls /dev/ttyACM*`
- Verify correct `--flowbot_port` argument

---

## References

- [Diffusion Policy (Columbia)](https://diffusion-policy.cs.columbia.edu/)
- [Stanford diffusion_policy repo](https://github.com/real-stanford/diffusion_policy)
- [UR5e RTDE documentation](https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/)
