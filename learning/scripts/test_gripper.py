#!/usr/bin/env python3
"""
Test Dynamixel Gripper

Dynamixel ID: 7
Open: 10 degrees
Close: 90 degrees
Port: /dev/ttyUSB0

Usage:
    python scripts/test_gripper.py
    python scripts/test_gripper.py --port /dev/ttyUSB1
"""

import sys
import time
import click

# Dynamixel SDK
from dynamixel_sdk import *

# Control table addresses (Protocol 2.0 - XM/XL series)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11
ADDR_DRIVE_MODE = 10

# Protocol version
PROTOCOL_VERSION = 2.0

# Default settings
DXL_ID = 7
BAUDRATE = 57600

# Position conversion (0.088 degrees per unit for XM series)
DEGREES_PER_UNIT = 0.088

# Gripper limits
OPEN_DEGREES = 10
CLOSE_DEGREES = 90


def degrees_to_position(degrees):
    """Convert degrees to Dynamixel position units"""
    return int(degrees / DEGREES_PER_UNIT)


def position_to_degrees(position):
    """Convert Dynamixel position units to degrees"""
    return position * DEGREES_PER_UNIT


@click.command()
@click.option('--port', default='/dev/ttyUSB0', help='Serial port')
@click.option('--baudrate', default=57600, help='Baudrate')
@click.option('--dxl_id', default=7, help='Dynamixel motor ID')
def main(port, baudrate, dxl_id):
    print("="*50)
    print("       DYNAMIXEL GRIPPER TEST")
    print("="*50)
    print(f"\nPort: {port}")
    print(f"Baudrate: {baudrate}")
    print(f"Motor ID: {dxl_id}")
    print(f"Open position: {OPEN_DEGREES}°")
    print(f"Close position: {CLOSE_DEGREES}°")

    # Initialize PortHandler
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if not port_handler.openPort():
        print(f"\n❌ Failed to open port {port}")
        print("   Check: ls -la /dev/ttyUSB*")
        print("   May need: sudo chmod 666 /dev/ttyUSB0")
        return

    print(f"\n✅ Port opened: {port}")

    # Set baudrate
    if not port_handler.setBaudRate(baudrate):
        print(f"❌ Failed to set baudrate {baudrate}")
        port_handler.closePort()
        return

    print(f"✅ Baudrate set: {baudrate}")

    # Ping motor
    print(f"\nPinging motor ID {dxl_id}...")
    dxl_model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)

    if dxl_comm_result != COMM_SUCCESS:
        print(f"❌ Failed to ping: {packet_handler.getTxRxResult(dxl_comm_result)}")
        port_handler.closePort()
        return
    elif dxl_error != 0:
        print(f"❌ Dynamixel error: {packet_handler.getRxPacketError(dxl_error)}")
        port_handler.closePort()
        return

    print(f"✅ Motor found! Model number: {dxl_model_number}")

    # Read current position
    current_pos, _, _ = packet_handler.read4ByteTxRx(port_handler, dxl_id, ADDR_PRESENT_POSITION)
    current_degrees = position_to_degrees(current_pos)
    print(f"\nCurrent position: {current_degrees:.1f}° (raw: {current_pos})")

    # Enable torque
    print("\nEnabling torque...")
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
        port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1
    )
    if dxl_comm_result != COMM_SUCCESS:
        print(f"❌ Failed: {packet_handler.getTxRxResult(dxl_comm_result)}")
        port_handler.closePort()
        return

    print("✅ Torque enabled")

    def move_to_degrees(target_degrees, description):
        """Move gripper to target position in degrees"""
        target_pos = degrees_to_position(target_degrees)
        print(f"\n{description}: {target_degrees}° (raw: {target_pos})")

        # Write goal position
        dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(
            port_handler, dxl_id, ADDR_GOAL_POSITION, target_pos
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"❌ Failed: {packet_handler.getTxRxResult(dxl_comm_result)}")
            return False

        # Wait for motion to complete
        print("Moving", end="", flush=True)
        for _ in range(50):  # Max 5 seconds
            time.sleep(0.1)
            present_pos, _, _ = packet_handler.read4ByteTxRx(
                port_handler, dxl_id, ADDR_PRESENT_POSITION
            )
            print(".", end="", flush=True)

            # Check if reached target (within 5 units tolerance)
            if abs(present_pos - target_pos) < 20:
                break

        present_degrees = position_to_degrees(present_pos)
        print(f" Done! Position: {present_degrees:.1f}°")
        return True

    try:
        # Test sequence
        print("\n" + "="*50)
        print("Starting gripper test sequence...")
        print("="*50)

        # 1. Open gripper
        input("\nPress Enter to OPEN gripper...")
        move_to_degrees(OPEN_DEGREES, "Opening gripper")

        # 2. Close gripper
        input("\nPress Enter to CLOSE gripper...")
        move_to_degrees(CLOSE_DEGREES, "Closing gripper")

        # 3. Open again
        input("\nPress Enter to OPEN gripper again...")
        move_to_degrees(OPEN_DEGREES, "Opening gripper")

        # Interactive control
        print("\n" + "="*50)
        print("Interactive control:")
        print("  'o' = Open (10°)")
        print("  'c' = Close (90°)")
        print("  'h' = Half (50°)")
        print("  '0-9' = Set position (0=0°, 9=90°)")
        print("  'q' = Quit")
        print("="*50)

        while True:
            cmd = input("\nCommand: ").strip().lower()

            if cmd == 'q':
                break
            elif cmd == 'o':
                move_to_degrees(OPEN_DEGREES, "Opening")
            elif cmd == 'c':
                move_to_degrees(CLOSE_DEGREES, "Closing")
            elif cmd == 'h':
                move_to_degrees(50, "Half open")
            elif cmd.isdigit():
                target = int(cmd) * 10
                move_to_degrees(target, f"Moving to {target}°")
            else:
                print("Unknown command. Use o/c/h/0-9/q")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Return to open position
        print("\nReturning to open position...")
        target_pos = degrees_to_position(OPEN_DEGREES)
        packet_handler.write4ByteTxRx(port_handler, dxl_id, ADDR_GOAL_POSITION, target_pos)
        time.sleep(1)

        # Disable torque
        print("Disabling torque...")
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)

        # Close port
        port_handler.closePort()
        print("\n✅ Gripper test completed!")


if __name__ == '__main__':
    main()
