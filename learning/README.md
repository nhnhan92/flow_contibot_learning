# Hướng dẫn xây dựng Pick and Place với Diffusion Policy

## Tổng quan

Hướng dẫn này sẽ giúp bạn xây dựng hệ thống thu thập dữ liệu và train Diffusion Policy cho task **Pick and Place** dựa trên codebase gốc của Columbia.

### Hardware yêu cầu

| Thiết bị | Model | Giao tiếp |
|----------|-------|-----------|
| Robot | UR5e | Ethernet (RTDE) |
| Gripper | Dynamixel | USB Serial |
| Wrist Camera | Intel RealSense D455 | USB 3.0 |
| Teleop | 3Dconnexion SpaceMouse | USB |

### Kiến trúc hệ thống

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ SpaceMouse  │     │  RealSense  │     │  Dynamixel  │
│  (teleop)   │     │    D455     │     │  Gripper    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │ spnav            │ pyrealsense2      │ dynamixel-sdk
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│                  Python Controller                   │
│              (demo_pickplace.py)                     │
└──────────────────────────┬──────────────────────────┘
                           │ ur-rtde
                           ▼
                    ┌─────────────┐
                    │   UR5e      │
                    │  (robot)    │
                    └─────────────┘
```

---

## Cấu trúc thư mục

```
Desktop/
├── diffusion_policy/                # Repo gốc (GIỮ NGUYÊN - không sửa)
│   ├── diffusion_policy/
│   │   ├── real_world/
│   │   │   ├── real_env.py
│   │   │   ├── spacemouse_shared_memory.py
│   │   │   └── ...
│   │   ├── config/
│   │   └── ...
│   ├── train.py
│   ├── demo_real_robot.py
│   └── ...
│
└── my_pickplace/                    # THƯ MỤC MỚI (tạo ở đây)
    ├── README.md                    # File này
    ├── config/
    │   └── real_pickplace_image.yaml
    ├── scripts/
    │   ├── demo_pickplace.py        # Script thu thập data
    │   ├── eval_pickplace.py        # Script evaluation
    │   └── test_hardware.py         # Test hardware
    ├── custom/
    │   └── dynamixel_gripper.py     # Gripper controller
    └── data/
        └── pickplace_demos/         # Data thu thập sẽ lưu ở đây
```

**Nguyên tắc**:
- `diffusion_policy/` = Repo gốc, giữ nguyên để dễ update
- `my_pickplace/` = Code custom của bạn, tách riêng

---

## Bước 1: Cài đặt Dependencies

Giả sử bạn đã có thư mục `diffusion_policy` với môi trường conda đã cài đặt.

```bash
# Activate environment
conda activate diffusion_policy

# Cài đặt thêm cho real robot
pip install ur-rtde==1.5.5          # UR5e control
pip install pyrealsense2            # RealSense D455
pip install dynamixel-sdk           # Dynamixel gripper

# SpaceMouse daemon (Ubuntu)
sudo apt update
sudo apt install spacenavd
sudo systemctl enable spacenavd
sudo systemctl start spacenavd

# SpaceMouse Python binding
pip install spnav
```

### Kiểm tra hardware

```bash
# Test RealSense
realsense-viewer

# Test SpaceMouse
systemctl status spacenavd

# Test UR5e (thay IP của bạn)
ping 192.168.1.100

# Test Dynamixel (xem port)
ls /dev/ttyUSB*
```

---

## Bước 2: Tạo cấu trúc thư mục my_pickplace

```bash
cd ~/Desktop
mkdir -p my_pickplace/{config,scripts,custom,data/pickplace_demos}
cd my_pickplace
```

---

## Bước 3: Tạo Dynamixel Gripper Controller

Tạo file `my_pickplace/custom/dynamixel_gripper.py`:

```python
"""
Dynamixel Gripper Controller
Compatible with XM/XL series, Protocol 2.0
"""

from dynamixel_sdk import *
import numpy as np
import time


class DynamixelGripper:
    """
    Controller for Dynamixel servo-based gripper

    Usage:
        gripper = DynamixelGripper(port='/dev/ttyUSB0', dxl_id=1)
        gripper.open()
        gripper.close()
        pos = gripper.get_position()  # 0.0 ~ 1.0
        gripper.disconnect()
    """

    # Control table addresses for Protocol 2.0
    ADDR_OPERATING_MODE = 11
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PROFILE_VELOCITY = 112

    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        baudrate: int = 57600,
        dxl_id: int = 1,
        position_open: int = 2048,      # Điều chỉnh theo gripper của bạn
        position_close: int = 3200,     # Điều chỉnh theo gripper của bạn
        protocol_version: float = 2.0,
        profile_velocity: int = 100     # Tốc độ di chuyển
    ):
        """
        Initialize Dynamixel gripper

        Args:
            port: USB port (e.g., '/dev/ttyUSB0')
            baudrate: Communication baudrate (default 57600)
            dxl_id: Dynamixel motor ID
            position_open: Motor position when fully open (0-4095)
            position_close: Motor position when fully closed (0-4095)
            protocol_version: Dynamixel protocol (1.0 or 2.0)
            profile_velocity: Movement speed (0-1023)
        """
        self.dxl_id = dxl_id
        self.position_open = position_open
        self.position_close = position_close

        # Initialize SDK handlers
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(protocol_version)

        # Open port
        if not self.port_handler.openPort():
            raise RuntimeError(f"Failed to open port {port}")
        print(f"Port {port} opened successfully")

        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")
        print(f"Baudrate set to {baudrate}")

        # Disable torque to change settings
        self._write_byte(self.ADDR_TORQUE_ENABLE, 0)

        # Set operating mode to Position Control (mode 3)
        self._write_byte(self.ADDR_OPERATING_MODE, 3)

        # Enable torque
        self._write_byte(self.ADDR_TORQUE_ENABLE, 1)

        # Set profile velocity
        self._write_dword(self.ADDR_PROFILE_VELOCITY, profile_velocity)

        print(f"Dynamixel Gripper initialized (ID: {dxl_id})")
        print(f"  Position range: {position_close} (close) ~ {position_open} (open)")

    def _write_byte(self, address: int, value: int) -> bool:
        """Write 1 byte to Dynamixel"""
        result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, self.dxl_id, address, value
        )
        if result != COMM_SUCCESS:
            print(f"Write error: {self.packet_handler.getTxRxResult(result)}")
            return False
        if error != 0:
            print(f"Dynamixel error: {self.packet_handler.getRxPacketError(error)}")
            return False
        return True

    def _write_dword(self, address: int, value: int) -> bool:
        """Write 4 bytes to Dynamixel"""
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, self.dxl_id, address, int(value)
        )
        if result != COMM_SUCCESS:
            print(f"Write error: {self.packet_handler.getTxRxResult(result)}")
            return False
        if error != 0:
            print(f"Dynamixel error: {self.packet_handler.getRxPacketError(error)}")
            return False
        return True

    def _read_dword(self, address: int) -> int:
        """Read 4 bytes from Dynamixel"""
        value, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.dxl_id, address
        )
        if result != COMM_SUCCESS:
            print(f"Read error: {self.packet_handler.getTxRxResult(result)}")
            return None
        if error != 0:
            print(f"Dynamixel error: {self.packet_handler.getRxPacketError(error)}")
            return None
        return value

    def set_position(self, position: float) -> None:
        """
        Set gripper position

        Args:
            position: Target position (0.0 = closed, 1.0 = open)
        """
        position = np.clip(position, 0.0, 1.0)
        dxl_pos = int(
            self.position_close +
            position * (self.position_open - self.position_close)
        )
        self._write_dword(self.ADDR_GOAL_POSITION, dxl_pos)

    def get_position(self) -> float:
        """
        Get current gripper position

        Returns:
            Current position (0.0 = closed, 1.0 = open)
        """
        dxl_pos = self._read_dword(self.ADDR_PRESENT_POSITION)
        if dxl_pos is None:
            return 0.5  # Return middle position on error

        position = (dxl_pos - self.position_close) / \
                   (self.position_open - self.position_close)
        return float(np.clip(position, 0.0, 1.0))

    def open(self) -> None:
        """Open gripper fully"""
        self.set_position(1.0)

    def close(self) -> None:
        """Close gripper fully"""
        self.set_position(0.0)

    def disconnect(self) -> None:
        """Disable torque and close port"""
        self._write_byte(self.ADDR_TORQUE_ENABLE, 0)
        self.port_handler.closePort()
        print("Gripper disconnected")


def test_gripper():
    """Test gripper functionality"""
    print("="*50)
    print("DYNAMIXEL GRIPPER TEST")
    print("="*50)

    # Điều chỉnh các thông số này theo setup của bạn
    gripper = DynamixelGripper(
        port='/dev/ttyUSB0',
        dxl_id=1,
        position_open=2048,
        position_close=3200
    )

    try:
        print("\n1. Opening gripper...")
        gripper.open()
        time.sleep(1.0)
        print(f"   Current position: {gripper.get_position():.2f}")

        print("\n2. Closing gripper...")
        gripper.close()
        time.sleep(1.0)
        print(f"   Current position: {gripper.get_position():.2f}")

        print("\n3. Moving to 50%...")
        gripper.set_position(0.5)
        time.sleep(1.0)
        print(f"   Current position: {gripper.get_position():.2f}")

        print("\n4. Opening gripper...")
        gripper.open()
        time.sleep(1.0)

        print("\nTest completed successfully!")

    finally:
        gripper.disconnect()


if __name__ == '__main__':
    test_gripper()
```

### Test gripper

```bash
cd ~/Desktop/my_pickplace
python custom/dynamixel_gripper.py
```

**Lưu ý**: Điều chỉnh `position_open` và `position_close` theo gripper của bạn.

---

## Bước 4: Tạo Task Configuration

Tạo file `my_pickplace/config/real_pickplace_image.yaml`:

```yaml
name: real_pickplace_image

# Đường dẫn đến data (relative to my_pickplace/)
dataset_path: data/pickplace_demos

# Image configuration (D455 output)
image_shape: [3, 240, 320]  # [C, H, W] - RGB

# Shape metadata
shape_meta:
  obs:
    # Wrist camera observation
    wrist_camera:
      shape: ${task.image_shape}
      type: rgb

    # Robot end-effector pose (full 6 DOF)
    robot_eef_pose:
      shape: [6]  # [x, y, z, rx, ry, rz]
      type: low_dim

    # Gripper state
    gripper_state:
      shape: [1]  # 0.0 (closed) ~ 1.0 (open)
      type: low_dim

  # Action space: 6D pose + gripper
  action:
    shape: [7]  # [x, y, z, rx, ry, rz, gripper]

# Hardware configuration - THAY ĐỔI THEO SETUP CỦA BẠN
env:
  # UR5e settings
  robot_ip: "192.168.1.100"       # ← Thay bằng IP robot của bạn

  # Gripper settings
  gripper_port: "/dev/ttyUSB0"    # ← Thay bằng port gripper
  gripper_id: 1
  gripper_position_open: 2048     # ← Calibrate theo gripper
  gripper_position_close: 3200    # ← Calibrate theo gripper

  # Control parameters
  frequency: 10                   # Data collection frequency (Hz)
  max_pos_speed: 0.25             # Max linear speed (m/s)
  max_rot_speed: 0.4              # Max angular speed (rad/s)

  # Camera settings
  camera_serial: null             # null = auto-detect first camera
  camera_resolution: [1280, 720]  # Capture resolution
  camera_fps: 30

  # Safety - workspace limits (meters)
  workspace_limits:
    x_min: -0.5
    x_max: 0.5
    y_min: -0.5
    y_max: 0.5
    z_min: 0.02    # Minimum 2cm above table
    z_max: 0.4

# Dataset parameters
n_obs_steps: 2
n_action_steps: 8
horizon: 16
```

---

## Bước 5: Tạo Hardware Test Script

Tạo file `my_pickplace/scripts/test_hardware.py`:

```python
#!/usr/bin/env python3
"""
Test all hardware components before data collection
"""

import sys
import os

# Add paths
PICKPLACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFUSION_POLICY_DIR = os.path.join(os.path.dirname(PICKPLACE_DIR), 'diffusion_policy')
sys.path.insert(0, PICKPLACE_DIR)
sys.path.insert(0, DIFFUSION_POLICY_DIR)

import time
import click


def test_gripper(port, dxl_id):
    """Test Dynamixel gripper"""
    print("\n" + "="*50)
    print("Testing DYNAMIXEL GRIPPER")
    print("="*50)

    try:
        from custom.dynamixel_gripper import DynamixelGripper

        gripper = DynamixelGripper(port=port, dxl_id=dxl_id)

        print("Opening...")
        gripper.open()
        time.sleep(1)
        print(f"Position: {gripper.get_position():.2f}")

        print("Closing...")
        gripper.close()
        time.sleep(1)
        print(f"Position: {gripper.get_position():.2f}")

        gripper.open()
        gripper.disconnect()
        print("GRIPPER TEST: PASSED")
        return True

    except Exception as e:
        print(f"GRIPPER TEST: FAILED - {e}")
        return False


def test_robot(robot_ip):
    """Test UR5e connection"""
    print("\n" + "="*50)
    print("Testing UR5e ROBOT")
    print("="*50)

    try:
        from rtde_receive import RTDEReceiveInterface

        rtde_r = RTDEReceiveInterface(robot_ip)
        pose = rtde_r.getActualTCPPose()
        joints = rtde_r.getActualQ()

        print(f"TCP Pose: {[f'{x:.3f}' for x in pose]}")
        print(f"Joints (deg): {[f'{np.degrees(x):.1f}' for x in joints]}")
        print("ROBOT TEST: PASSED")
        return True

    except Exception as e:
        print(f"ROBOT TEST: FAILED - {e}")
        return False


def test_camera():
    """Test RealSense camera"""
    print("\n" + "="*50)
    print("Testing REALSENSE CAMERA")
    print("="*50)

    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            print("No RealSense device found!")
            print("CAMERA TEST: FAILED")
            return False

        for i, dev in enumerate(devices):
            print(f"Device {i}: {dev.get_info(rs.camera_info.name)}")
            print(f"  Serial: {dev.get_info(rs.camera_info.serial_number)}")

        print("CAMERA TEST: PASSED")
        return True

    except Exception as e:
        print(f"CAMERA TEST: FAILED - {e}")
        return False


def test_spacemouse():
    """Test SpaceMouse"""
    print("\n" + "="*50)
    print("Testing SPACEMOUSE")
    print("="*50)

    try:
        from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse

        print("Move SpaceMouse to test (5 seconds)...")
        with Spacemouse(deadzone=0.1, max_value=500) as sm:
            for i in range(50):
                state = sm.get_motion_state_transformed()
                if i % 10 == 0:
                    print(f"  State: {[f'{x:.2f}' for x in state]}")
                time.sleep(0.1)

        print("SPACEMOUSE TEST: PASSED")
        return True

    except Exception as e:
        print(f"SPACEMOUSE TEST: FAILED - {e}")
        return False


@click.command()
@click.option('--robot_ip', default='192.168.1.100', help='UR5e IP')
@click.option('--gripper_port', default='/dev/ttyUSB0', help='Gripper port')
@click.option('--gripper_id', default=1, help='Dynamixel ID')
def main(robot_ip, gripper_port, gripper_id):
    import numpy as np

    print("\n" + "="*50)
    print("       HARDWARE TEST SUITE")
    print("="*50)

    results = {}

    # Test each component
    results['gripper'] = test_gripper(gripper_port, gripper_id)
    results['robot'] = test_robot(robot_ip)
    results['camera'] = test_camera()
    results['spacemouse'] = test_spacemouse()

    # Summary
    print("\n" + "="*50)
    print("       TEST SUMMARY")
    print("="*50)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name.upper():15} : {status}")
        if not passed:
            all_passed = False

    print("="*50)
    if all_passed:
        print("All tests passed! Ready for data collection.")
    else:
        print("Some tests failed. Please fix before proceeding.")


if __name__ == '__main__':
    main()
```

### Chạy test

```bash
cd ~/Desktop/my_pickplace
python scripts/test_hardware.py --robot_ip 192.168.1.100 --gripper_port /dev/ttyUSB0
```

---

## Bước 6: Tạo Data Collection Script

Tạo file `my_pickplace/scripts/demo_pickplace.py`:

```python
#!/usr/bin/env python3
"""
Pick and Place Data Collection Script

Usage:
    cd ~/Desktop/my_pickplace
    python scripts/demo_pickplace.py -o data/pickplace_demos --robot_ip 192.168.1.100

Controls:
    SpaceMouse      : Move robot XYZ
    Button LEFT     : Toggle gripper open/close
    Button RIGHT    : Hold for rotation mode

    Key 'C'         : Start recording
    Key 'S'         : Stop recording
    Key 'Q'         : Quit
    Backspace       : Delete last episode
"""

import sys
import os

# === PATH SETUP ===
# Get directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICKPLACE_DIR = os.path.dirname(SCRIPT_DIR)  # my_pickplace/
DIFFUSION_POLICY_DIR = os.path.join(os.path.dirname(PICKPLACE_DIR), 'diffusion_policy')

# Add to path
sys.path.insert(0, PICKPLACE_DIR)
sys.path.insert(0, DIFFUSION_POLICY_DIR)

import time
import click
import numpy as np
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager

# Import từ diffusion_policy gốc
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

# Import custom gripper
from custom.dynamixel_gripper import DynamixelGripper


def print_controls():
    """Print control instructions"""
    print("\n" + "="*60)
    print("       PICK AND PLACE DATA COLLECTION")
    print("="*60)
    print("\nSpaceMouse Controls:")
    print("  Move              : Control robot XYZ position")
    print("  Button LEFT (0)   : Toggle gripper (open/close)")
    print("  Button RIGHT (1)  : Hold for rotation mode")
    print("\nKeyboard Controls:")
    print("  'C'               : Start recording episode")
    print("  'S'               : Stop recording episode")
    print("  'Q'               : Quit program")
    print("  Backspace         : Delete last episode")
    print("="*60 + "\n")


@click.command()
@click.option('-o', '--output', required=True, help='Output directory for demos')
@click.option('-ri', '--robot_ip', required=True, help='UR5e IP address')
@click.option('-gp', '--gripper_port', default='/dev/ttyUSB0', help='Dynamixel port')
@click.option('-gi', '--gripper_id', default=1, type=int, help='Dynamixel ID')
@click.option('-f', '--frequency', default=10, type=float, help='Control frequency Hz')
@click.option('--max_pos_speed', default=0.25, type=float, help='Max position speed m/s')
@click.option('--max_rot_speed', default=0.4, type=float, help='Max rotation speed rad/s')
@click.option('--gripper_open_pos', default=2048, type=int, help='Gripper open position')
@click.option('--gripper_close_pos', default=3200, type=int, help='Gripper close position')
def main(
    output,
    robot_ip,
    gripper_port,
    gripper_id,
    frequency,
    max_pos_speed,
    max_rot_speed,
    gripper_open_pos,
    gripper_close_pos
):
    # Resolve output path relative to my_pickplace
    if not os.path.isabs(output):
        output = os.path.join(PICKPLACE_DIR, output)

    # =============================================
    # Initialize Hardware
    # =============================================

    print("Initializing Dynamixel gripper...")
    gripper = DynamixelGripper(
        port=gripper_port,
        dxl_id=gripper_id,
        position_open=gripper_open_pos,
        position_close=gripper_close_pos
    )
    gripper.open()
    gripper_is_open = True

    # Create output directory
    os.makedirs(output, exist_ok=True)
    print(f"Output directory: {output}")

    # =============================================
    # Main Control Loop
    # =============================================

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output,
            robot_ip=robot_ip,
            frequency=frequency,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            shm_manager=shm_manager,
            enable_multi_cam_vis=True,
            record_raw_video=True,
        ) as env:

            with Spacemouse(deadzone=0.3, max_value=500) as sm:
                with KeystrokeCounter() as key_counter:

                    print_controls()

                    # Get initial robot pose
                    robot_state = env.get_robot_state()
                    target_pose = robot_state['ActualTCPPose'].copy()
                    print(f"Initial TCP pose: {[f'{x:.4f}' for x in target_pose]}")

                    # State tracking
                    prev_button_0 = False
                    is_recording = False
                    episode_count = 0
                    iter_idx = 0

                    try:
                        while True:
                            # Timing
                            t_cycle_start = time.monotonic()
                            t_sample = t_cycle_start + (1/frequency) * 0.5
                            t_command_target = t_cycle_start + (1/frequency)
                            t_cycle_end = t_cycle_start + (1/frequency)

                            # =========================================
                            # Handle Keyboard
                            # =========================================
                            press_events = key_counter.get_press_events()
                            for key_stroke in press_events:
                                if key_stroke == Key.backspace:
                                    if click.confirm('Delete last episode?'):
                                        env.drop_episode()
                                        episode_count = max(0, episode_count - 1)
                                        print(f"Deleted. Episodes: {episode_count}")

                                elif key_stroke == KeyCode(char='c'):
                                    if not is_recording:
                                        env.start_episode()
                                        is_recording = True
                                        print("\n>>> RECORDING STARTED <<<")

                                elif key_stroke == KeyCode(char='s'):
                                    if is_recording:
                                        env.end_episode()
                                        is_recording = False
                                        episode_count += 1
                                        print(f">>> STOPPED <<< Total: {episode_count}\n")

                                elif key_stroke == KeyCode(char='q'):
                                    print("\nQuitting...")
                                    raise KeyboardInterrupt

                            # =========================================
                            # SpaceMouse Input
                            # =========================================
                            precise_wait(t_sample)

                            sm_state = sm.get_motion_state_transformed()
                            dpos = sm_state[:3] * (max_pos_speed / frequency)
                            drot_xyz = sm_state[3:] * (max_rot_speed / frequency)

                            # Button 1 (Right): Rotation mode
                            if not sm.is_button_pressed(1):
                                drot_xyz[:] = 0  # Translation only
                            else:
                                dpos[:] = 0  # Rotation only

                            # Button 0 (Left): Toggle gripper
                            button_0 = sm.is_button_pressed(0)
                            if button_0 and not prev_button_0:
                                gripper_is_open = not gripper_is_open
                                if gripper_is_open:
                                    gripper.open()
                                    print("  Gripper: OPEN")
                                else:
                                    gripper.close()
                                    print("  Gripper: CLOSE")
                            prev_button_0 = button_0

                            # =========================================
                            # Update Target Pose
                            # =========================================
                            target_pose[:3] += dpos

                            if np.any(drot_xyz != 0):
                                drot = st.Rotation.from_euler('xyz', drot_xyz)
                                current_rot = st.Rotation.from_rotvec(target_pose[3:])
                                new_rot = drot * current_rot
                                target_pose[3:] = new_rot.as_rotvec()

                            # =========================================
                            # Execute
                            # =========================================
                            gripper_state = gripper.get_position()

                            env.exec_actions(
                                actions=[target_pose],
                                timestamps=[t_command_target - time.monotonic() + time.time()],
                                stages=[0]
                            )

                            precise_wait(t_cycle_end)
                            iter_idx += 1

                            # Status update
                            if iter_idx % 100 == 0:
                                status = "REC" if is_recording else "---"
                                print(f"[{status}] iter={iter_idx} eps={episode_count} grip={gripper_state:.2f}")

                    except KeyboardInterrupt:
                        pass

                    finally:
                        print("\nCleaning up...")
                        gripper.open()
                        time.sleep(0.3)
                        gripper.disconnect()
                        print(f"Done! Episodes collected: {episode_count}")


if __name__ == '__main__':
    main()
```

---

## Bước 7: Thu thập dữ liệu

### 7.1 Chạy script

```bash
cd ~/Desktop/my_pickplace

python scripts/demo_pickplace.py \
    -o data/pickplace_demos \
    --robot_ip 192.168.1.100 \
    --gripper_port /dev/ttyUSB0 \
    --gripper_id 1
```

### 7.2 Quy trình mỗi episode

1. Đặt object vào vị trí ngẫu nhiên trên bàn
2. Di chuyển robot về vị trí bắt đầu
3. Nhấn **'C'** để bắt đầu ghi
4. Thực hiện pick and place:
   - Di chuyển đến object
   - **Button trái** để đóng gripper (pick)
   - Di chuyển đến vị trí đặt
   - **Button trái** để mở gripper (place)
5. Nhấn **'S'** để dừng ghi
6. Lặp lại (50-100 episodes)

### 7.3 Tips

| Tip | Lý do |
|-----|-------|
| Di chuyển chậm | Smooth trajectories dễ học hơn |
| Đa dạng vị trí | Model generalize tốt hơn |
| Xóa episode lỗi ngay | Không để data xấu vào training |
| 50+ episodes | Minimum để train được |
| Consistent pattern | Cùng cách tiếp cận mỗi episode |

---

## Bước 8: Training

```bash
cd ~/Desktop/diffusion_policy

python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task.dataset_path=../my_pickplace/data/pickplace_demos \
    training.num_epochs=3000 \
    training.batch_size=64
```

**Lưu ý**: Cần sửa config nếu action shape khác (7 thay vì 2).

---

## Bước 9: Evaluation

Tạo file `my_pickplace/scripts/eval_pickplace.py` tương tự `demo_pickplace.py` nhưng:
- Load checkpoint thay vì dùng SpaceMouse
- Chạy policy inference
- Gửi action đến robot

---

## Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| `ModuleNotFoundError` | Check sys.path có đúng không |
| Gripper không phản hồi | Check port: `ls /dev/ttyUSB*` |
| Camera không detect | Dùng USB 3.0, chạy `realsense-viewer` |
| Robot không kết nối | Check IP, RTDE enabled trên robot |
| SpaceMouse lag | `sudo systemctl restart spacenavd` |

---

## Checklist

- [ ] Dependencies đã cài đặt
- [ ] Tạo thư mục `my_pickplace/` với cấu trúc đúng
- [ ] `dynamixel_gripper.py` hoạt động
- [ ] `test_hardware.py` pass tất cả tests
- [ ] `demo_pickplace.py` chạy được
- [ ] Thu thập 50+ episodes
- [ ] Train model
- [ ] Evaluation

---

## Files tạo mới

```
my_pickplace/
├── README.md                           # File này
├── config/
│   └── real_pickplace_image.yaml       # Task config
├── scripts/
│   ├── test_hardware.py                # Hardware test
│   ├── demo_pickplace.py               # Data collection
│   └── eval_pickplace.py               # Evaluation (tự tạo)
├── custom/
│   └── dynamixel_gripper.py            # Gripper controller
└── data/
    └── pickplace_demos/                # Data output
```
