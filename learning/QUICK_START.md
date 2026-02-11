# Quick Start Guide - Pick-Place với Diffusion Policy

Hướng dẫn nhanh để train và deploy Diffusion Policy cho task pick-place.

## 📋 Tổng quan

Bạn đã có:
- ✅ 11 episodes demonstration data (2713 samples)
- ✅ Dataset với camera, robot pose, gripper state
- ✅ Training pipeline hoàn chỉnh
- ✅ Visualization tools

## 🚀 Workflow

### Bước 1: Kiểm tra Data

Trước khi train, nên visualize data để đảm bảo chất lượng:

```bash
# Xem statistics
python visualize/compare_episodes.py --dataset data/camera_demos/dataset.zarr --stats_only

# Visualize episode đầu tiên
python visualize/quick_visualize.py --dataset data/camera_demos/dataset.zarr --episode 0

# So sánh tất cả episodes
python visualize/quick_visualize.py --dataset data/camera_demos/dataset.zarr --all
```

**Kết quả mong đợi:**
```
============================================================
DATASET STATISTICS
============================================================
Total episodes: 11
Total samples: 2713
Average episode length: 246.6 ± 43.0
Min/Max episode length: 182 / 305
============================================================

Workspace Range:
  X: [-0.179, 0.327] (range: 0.506m)
  Y: [-0.670, -0.263] (range: 0.407m)
  Z: [0.087, 0.542] (range: 0.455m)

Gripper Statistics:
  Range: [0.002, 1.000]
  Open ratio: 63.4%
```

### Bước 2: Test Pipeline

Kiểm tra xem training pipeline hoạt động đúng không:

```bash
python train/test_pipeline.py --config train/config.yaml
```

**Kết quả mong đợi:**
```
============================================================
   TEST SUMMARY
============================================================
✅ PASS - Dataset
✅ PASS - Model
✅ PASS - DataLoader
✅ PASS - Full Pipeline

🎉 All tests passed! Ready to train!
```

### Bước 3: Training

Bắt đầu training:

```bash
# Training cơ bản
python train/train.py --config train/config.yaml

# Training với W&B logging (nếu muốn)
python train/train.py --config train/config.yaml --wandb_project pickplace

# Training với GPU cụ thể
CUDA_VISIBLE_DEVICES=0 python train/train.py --config train/config.yaml
```

**Training sẽ:**
- Train cho 1000 epochs (có thể dừng sớm nếu thấy validation loss đã tốt)
- Save checkpoints mỗi 50 epochs
- Save best model dựa trên validation loss (với EMA)
- In ra train/val loss mỗi epoch

**Thời gian training:**
- ~5-10 phút/epoch với GPU (batch_size=64)
- ~30-60 phút/epoch với CPU

### Bước 4: Evaluation

Sau khi train xong, evaluate model:

```bash
# Evaluate trên validation set
python train/eval.py \
    --checkpoint train/checkpoints/best_model.pt \
    --mode eval

# Visualize predictions
python train/eval.py \
    --checkpoint train/checkpoints/best_model.pt \
    --mode visualize \
    --sample_idx 0
```

**Kết quả tốt:**
- L1 Error < 0.05 cho robot pose
- L2 Error < 0.08 cho robot pose
- Gripper predictions smooth và match với ground truth

## 📊 Model Architecture

```
Input:
  - Camera: 2 frames RGB (96×96)
  - Robot State: 2 frames × 7D (pose 6D + gripper 1D)

Encoders:
  - Vision CNN → 256D features
  - State MLP → 64D features

Diffusion Model:
  - 1D U-Net with skip connections
  - 100 diffusion steps (DDPM)

Output:
  - 16 future actions × 7D (pose 6D + gripper 1D)
  - Execute first 8 actions, then replan
```

## ⚙️ Hyperparameters

Các tham số chính trong [train/config.yaml](train/config.yaml):

```yaml
# Data
obs_horizon: 2        # Số frames quan sát
pred_horizon: 16      # Số actions predict
action_horizon: 8     # Số actions execute trước khi replan

# Model
action_dim: 7         # Robot pose (6D) + gripper (1D)
num_diffusion_iters: 100

# Training
batch_size: 64        # Giảm xuống 32 hoặc 16 nếu GPU nhỏ
learning_rate: 1e-4
num_epochs: 1000
```

## 🔧 Troubleshooting

### Training loss không giảm
- Kiểm tra data quality bằng visualize
- Thử giảm learning rate: `1e-5`
- Tăng batch size nếu GPU đủ mạnh

### Out of Memory
- Giảm `batch_size`: 32 hoặc 16
- Giảm `image_size`: [64, 64]
- Giảm `num_workers`: 2 hoặc 0

### Predictions không smooth
- Tăng `num_diffusion_iters` khi inference
- Sử dụng EMA weights (đã default)
- Thu thập thêm data smoother

### Model predict sai gripper
- Kiểm tra gripper range trong data (phải 0-1)
- Ensure normalization đúng
- Có thể cần weight riêng cho gripper loss

## 📁 Cấu trúc Project

```
my_pickplace/
├── data/
│   └── camera_demos/
│       └── dataset.zarr/          # Demonstration data
├── scripts/
│   ├── collect_demos_with_camera.py  # Data collection
│   └── test_*.py                  # Hardware tests
├── train/
│   ├── config.yaml                # Training config
│   ├── dataset.py                 # Dataset loader
│   ├── model.py                   # Diffusion Policy
│   ├── train.py                   # Training script
│   ├── eval.py                    # Evaluation
│   ├── test_pipeline.py           # Pipeline test
│   └── checkpoints/               # Saved models
├── visualize/
│   ├── quick_visualize.py         # Matplotlib viz
│   ├── compare_episodes.py        # Episode comparison
│   └── visualize_dataset.py       # Rerun viz (advanced)
└── custom/
    └── dynamixel_gripper.py       # Gripper control
```

## 🎯 Next Steps

### Sau khi có model tốt:

1. **Deploy trên robot:**
   - Tạo inference script (tham khảo [train/eval.py](train/eval.py))
   - Load model checkpoint
   - Chạy control loop với model predictions

2. **Fine-tune:**
   - Thu thập thêm data ở các situations khác nhau
   - Retrain hoặc fine-tune từ checkpoint
   - Thử các hyperparameters khác

3. **Cải thiện:**
   - Tăng số episodes (50-100 episodes tốt hơn)
   - Thử task khác nhau
   - Thử architecture khác (transformer, etc.)

## 📖 Tài liệu tham khảo

- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)
- [train/README.md](train/README.md) - Chi tiết training
- [visualize/README.md](visualize/README.md) - Chi tiết visualization
- [Original Diffusion Policy Repo](https://github.com/real-stanford/diffusion_policy)

## 💡 Tips

1. **Data quality > Quantity:** 10 episodes tốt > 50 episodes noisy
2. **Start simple:** Test với 1-2 episodes trước, đảm bảo pipeline works
3. **Monitor training:** Watch train/val loss, stop nếu overfit
4. **Use EMA weights:** Always evaluate với EMA (đã default)
5. **Visualize predictions:** Luôn visualize để hiểu model đang học gì

---

**Chúc bạn training thành công! 🚀**

Nếu có vấn đề, check:
1. train/README.md
2. visualize/README.md
3. GitHub issues của diffusion_policy

