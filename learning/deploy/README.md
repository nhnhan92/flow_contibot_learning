# Robot Deployment

Deploy trained Diffusion Policy model on real UR5e robot.

## Quick Start

### 1. Test Connections First

Before deploying, make sure all hardware is working:

```bash
# Test robot connection
python scripts/test_robot_connection.py

# Test camera
python scripts/test_camera.py

# Test gripper
python scripts/test_gripper.py
```

### 2. Deploy Model

```bash
# Deploy with best model
python deploy/deploy_real_robot.py --checkpoint train/checkpoints/best_model.pt

# Run multiple episodes
python deploy/deploy_real_robot.py \
    --checkpoint train/checkpoints/best_model.pt \
    --num_episodes 5 \
    --max_steps 300

# Custom robot IP and frequency
python deploy/deploy_real_robot.py \
    --checkpoint train/checkpoints/best_model.pt \
    --robot_ip 192.168.1.102 \
    --frequency 10.0
```

## Deployment Workflow

1. **Load trained model** from checkpoint
2. **Connect to robot** (UR5e via ROS)
3. **Initialize camera** (RealSense D455)
4. **Initialize gripper** (Dynamixel)
5. **Move to start position**
6. **Run inference loop**:
   - Get current observation (camera + robot state)
   - Predict action sequence (pred_horizon actions)
   - Execute first action_horizon actions
   - Replan when action queue is empty
7. **Repeat** for multiple episodes

## Control Flow

```
┌─────────────────────────────────────────────────────┐
│  Initialize                                         │
│  - Load policy checkpoint                           │
│  - Connect to robot, camera, gripper                │
│  - Move to start position                           │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Episode Loop                                       │
│  ┌───────────────────────────────────────────────┐ │
│  │ Control Loop (10 Hz)                          │ │
│  │                                                │ │
│  │ 1. Get Observation                            │ │
│  │    - Camera image (96×96 RGB)                 │ │
│  │    - Robot state (pose + gripper)             │ │
│  │                                                │ │
│  │ 2. Update Buffers                             │ │
│  │    - Store last obs_horizon observations      │ │
│  │                                                │ │
│  │ 3. Predict Actions (if queue empty)           │ │
│  │    - Run diffusion model                      │ │
│  │    - Get pred_horizon actions                 │ │
│  │    - Queue first action_horizon actions       │ │
│  │                                                │ │
│  │ 4. Execute Action                             │ │
│  │    - Pop action from queue                    │ │
│  │    - Send to robot & gripper                  │ │
│  │                                                │ │
│  │ 5. Wait for next control cycle (0.1s)         │ │
│  └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Shutdown                                           │
│  - Stop robot                                       │
│  - Close camera                                     │
│  - Disable gripper                                  │
└─────────────────────────────────────────────────────┘
```

## Configuration

### Start Position

Edit `move_to_start_position()` in `deploy_real_robot.py`:

```python
start_pose = [0.3, -0.4, 0.3, 3.14, 0, 0]  # [x, y, z, rx, ry, rz]
```

This should match the start position used during data collection.

### Control Frequency

Default: 10 Hz (0.1s per control loop)

```bash
python deploy/deploy_real_robot.py --checkpoint ... --frequency 10.0
```

Higher frequency = more responsive but may skip replanning cycles.

### Action Execution

- **pred_horizon**: 16 (predict 16 future actions ~1.6s)
- **action_horizon**: 8 (execute first 8 before replanning)

This means the model replans every 8 steps (0.8s at 10Hz).

## Safety

### Emergency Stop

- **Press Ctrl+C** to stop episode immediately
- Robot will stop movement
- All hardware will shutdown gracefully

### Safety Checks

Before deployment:
1. ✅ Ensure robot workspace is clear
2. ✅ Test with slow velocity first
3. ✅ Keep E-stop button accessible
4. ✅ Monitor first episode closely
5. ✅ Check gripper doesn't grip too hard

### Velocity Limits

Current settings in code:
```python
velocity=0.2       # 20% of max velocity
acceleration=0.5   # 50% of max acceleration
```

Increase gradually once confident model is safe.

## Troubleshooting

### Robot doesn't move

- Check robot IP address: `--robot_ip 192.168.1.102`
- Verify robot is in remote control mode
- Check ROS connection: `rostopic list`

### Camera not working

- List cameras: `rs-enumerate-devices`
- Specify serial: `--camera_serial <serial>`
- Check USB connection

### Gripper issues

- Check Dynamixel USB connection
- Verify port permissions: `sudo chmod 666 /dev/ttyUSB*`
- Test with `scripts/test_gripper.py`

### Model predictions are bad

- Check normalization stats match training data
- Visualize predictions with `train/eval.py`
- Try different checkpoint (e.g., `checkpoint_epoch_800.pt`)
- Consider retraining with more data

### Robot moves too fast/slow

- Adjust `--frequency` (default 10 Hz)
- Modify velocity/acceleration in `execute_action()`
- Check control loop timing in verbose output

## Advanced Usage

### Custom Observation Processing

Override `preprocess_observation()` if you need different normalization:

```python
def preprocess_observation(self, image, state):
    # Custom image processing
    image = your_custom_preprocessing(image)

    # Custom state processing
    state = your_custom_state_processing(state)

    return image, state
```

### Logging Deployment Data

Add logging to save deployment episodes for analysis:

```python
# In run_episode()
episode_data = {
    'observations': [],
    'actions': [],
    'timestamps': [],
}

# During episode
episode_data['observations'].append((image, state))
episode_data['actions'].append(action)
episode_data['timestamps'].append(time.time())

# After episode
np.save(f'deployment_episode_{episode}.npy', episode_data)
```

### Different Start Positions

Test robustness by varying start position:

```python
start_positions = [
    [0.3, -0.4, 0.3, 3.14, 0, 0],  # Center
    [0.25, -0.45, 0.3, 3.14, 0, 0],  # Left
    [0.35, -0.35, 0.3, 3.14, 0, 0],  # Right
]

for start_pose in start_positions:
    deployment.move_to_start_position(start_pose)
    deployment.run_episode()
```

## Performance Metrics

Expected performance at 10 Hz:
- **Inference time**: ~50-100ms (depends on GPU)
- **Action execution**: Asynchronous (non-blocking)
- **Observation capture**: ~10-20ms
- **Total loop time**: ~100ms (10 Hz)

Monitor with `--verbose` flag to see actual timing.

## Next Steps

After successful deployment:
1. **Collect deployment data** for analysis
2. **Compare with demonstration data** (consistency check)
3. **Fine-tune model** if needed with deployment data
4. **Test edge cases** (different objects, positions)
5. **Increase difficulty** gradually

## References

- Training: [train/README.md](../train/README.md)
- Data Collection: [QUICK_START.md](../QUICK_START.md)
- Model Architecture: [train/model.py](../train/model.py)
