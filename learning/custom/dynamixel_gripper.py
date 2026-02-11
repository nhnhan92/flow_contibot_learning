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
        dxl_id: int = 7,
        position_open: int = 10,      # Điều chỉnh theo gripper của bạn
        position_close: int = 900,     # Điều chỉnh theo gripper của bạn
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
        dxl_id=7,
        position_open=10,
        position_close=900
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
