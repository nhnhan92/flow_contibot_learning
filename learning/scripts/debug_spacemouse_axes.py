#!/usr/bin/env python3
"""
Debug SpaceMouse Axes - Xác định mapping giữa SpaceMouse và Robot

Chạy script này và di chuyển từng trục một để xem raw values
"""

import sys
import os
import time
import ctypes
from ctypes import c_int, c_uint, Structure, POINTER, byref

# spnav event types
SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2

class SpnavMotionEvent(Structure):
    _fields_ = [
        ("type", c_int),
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
        ("rx", c_int),
        ("ry", c_int),
        ("rz", c_int),
        ("period", c_uint),
        ("data", ctypes.c_void_p),
    ]

class SpnavButtonEvent(Structure):
    _fields_ = [
        ("type", c_int),
        ("press", c_int),
        ("bnum", c_int),
    ]

class SpnavEvent(ctypes.Union):
    _fields_ = [
        ("type", c_int),
        ("motion", SpnavMotionEvent),
        ("button", SpnavButtonEvent),
    ]


def main():
    print("="*60)
    print("     DEBUG SPACEMOUSE AXES")
    print("="*60)

    # Load libspnav
    try:
        lib = ctypes.CDLL("libspnav.so")
    except OSError:
        lib = ctypes.CDLL("libspnav.so.0")

    lib.spnav_open.restype = c_int
    lib.spnav_poll_event.argtypes = [POINTER(SpnavEvent)]
    lib.spnav_poll_event.restype = c_int

    if lib.spnav_open() == -1:
        print("Cannot connect to spacenavd!")
        return

    print("\nSpaceMouse connected!")
    print("\n" + "="*60)
    print("Di chuyển SpaceMouse từng trục một và quan sát giá trị RAW:")
    print("="*60)
    print("\nLinear:")
    print("  - Đẩy TRÁI/PHẢI      → xem raw X thay đổi")
    print("  - NHẤC LÊN/HẠ XUỐNG  → xem raw Y thay đổi")
    print("  - Đẩy TỚI/LUI        → xem raw Z thay đổi")
    print("\nRotation:")
    print("  - NGHIÊNG TRÁI/PHẢI  → xem raw RX thay đổi")
    print("  - NGHIÊNG TỚI/LUI    → xem raw RY thay đổi")
    print("  - XOAY (twist)       → xem raw RZ thay đổi")
    print("\nPress Ctrl+C to quit")
    print("="*60 + "\n")

    # State
    x = y = z = rx = ry = rz = 0

    try:
        while True:
            event = SpnavEvent()

            while lib.spnav_poll_event(byref(event)) != 0:
                if event.type == SPNAV_EVENT_MOTION:
                    x = event.motion.x
                    y = event.motion.y
                    z = event.motion.z
                    rx = event.motion.rx
                    ry = event.motion.ry
                    rz = event.motion.rz

            # Find which axis has largest value
            linear_vals = {'X': x, 'Y': y, 'Z': z}
            rot_vals = {'RX': rx, 'RY': ry, 'RZ': rz}

            max_linear = max(linear_vals.keys(), key=lambda k: abs(linear_vals[k]))
            max_rot = max(rot_vals.keys(), key=lambda k: abs(rot_vals[k]))

            # Print raw values
            print(f"RAW Linear: X={x:+5d}  Y={y:+5d}  Z={z:+5d}  | "
                  f"RAW Rotation: RX={rx:+5d}  RY={ry:+5d}  RZ={rz:+5d}  | "
                  f"Active: {max_linear if abs(linear_vals[max_linear]) > 20 else '-':>2} "
                  f"{max_rot if abs(rot_vals[max_rot]) > 20 else '-':>2}",
                  end='\r')

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        lib.spnav_close()


if __name__ == '__main__':
    main()
