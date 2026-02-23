#!/usr/bin/env python3
"""
Test SpaceMouse using spacenavd directly via ctypes

Usage:
    cd ~/Desktop/my_pickplace
    python scripts/test_spacemouse.py
"""

import time
import click
import ctypes
import os
import sys
FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)

from ctypes import c_int, c_uint, Structure, POINTER, byref
from flowbot.plot_helper import plot_helper

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


class SpaceMouse:
    """Simple SpaceMouse reader using libspnav"""

    def __init__(self):
        self.lib = None
        self.x = self.y = self.z = 0
        self.rx = self.ry = self.rz = 0
        self.buttons = [False, False]

        # Load libspnav
        try:
            self.lib = ctypes.CDLL("libspnav.so")
        except OSError:
            try:
                self.lib = ctypes.CDLL("libspnav.so.0")
            except OSError:
                raise RuntimeError("Cannot load libspnav. Install: sudo apt install libspnav-dev")

        # Setup function signatures
        self.lib.spnav_open.restype = c_int
        self.lib.spnav_close.restype = c_int
        self.lib.spnav_poll_event.argtypes = [POINTER(SpnavEvent)]
        self.lib.spnav_poll_event.restype = c_int

        # Open connection
        if self.lib.spnav_open() == -1:
            raise RuntimeError("Cannot connect to spacenavd. Check: systemctl status spacenavd")

    def poll(self):
        """Poll for events and update state"""
        event = SpnavEvent()

        while self.lib.spnav_poll_event(byref(event)) != 0:
            if event.type == SPNAV_EVENT_MOTION:
                self.x = event.motion.x
                self.y = event.motion.y
                self.z = event.motion.z
                self.rx = event.motion.rx
                self.ry = event.motion.ry
                self.rz = event.motion.rz
            elif event.type == SPNAV_EVENT_BUTTON:
                if event.button.bnum < 2:
                    self.buttons[event.button.bnum] = bool(event.button.press)

    def get_state(self):
        """Get current state as normalized values

        SpaceMouse raw axes mapping:
            raw x = left/right
            raw y = up/down (nhấc lên/đặt xuống)
            raw z = forward/backward (đẩy tới/lui)

        Remapped to diffusion_policy convention:
            X = forward/backward (raw -z)
            Y = left/right (raw x)
            Z = up/down (raw y)
        """
        self.poll()
        scale = 350.0  # Normalization factor
        return {
            'x': -self.z / scale,          # forward/backward -> X
            'y': self.x / scale,           # left/right -> Y
            'z': self.y / scale,           # up/down -> Z
            'rx': -self.rz / scale,
            'ry': self.rx / scale,
            'rz': self.ry / scale,
            'btn_left': self.buttons[0],
            'btn_right': self.buttons[1],
        }

    def close(self):
        if self.lib:
            self.lib.spnav_close()


@click.command()
@click.option('--duration', default=30, help='Test duration in seconds')
def main(duration):
    print("="*60)
    print("         SPACEMOUSE TEST (libspnav)")
    print("="*60)
    print(f"\nDuration: {duration} seconds")
    print("\nControls (diffusion_policy convention):")
    print("  Push forward/back  -> X")
    print("  Push left/right    -> Y")
    print("  Lift up/down       -> Z")
    print("  Tilt SpaceMouse    -> Rotation (rx, ry, rz)")
    print("  Press buttons      -> Button states (L/R)")
    print("  Ctrl+C             -> Quit")
    print("="*60)

    sm = None
    try:
        sm = SpaceMouse()
        print("\nSpaceMouse connected! Start moving...\n")

        start_time = time.time()
        iter_count = 0

        while time.time() - start_time < duration:
            state = sm.get_state()

            # Format output
            xyz = f"XYZ:[{state['x']:+6.2f},{state['y']:+6.2f},{state['z']:+6.2f}]"
            rot = f"Rot:[{state['rx']:+6.2f},{state['ry']:+6.2f},{state['rz']:+6.2f}]"
            btn = f"Btn: L={int(state['btn_left'])} R={int(state['btn_right'])}"

            # Print every 5 iterations
            if iter_count % 5 == 0:
                print(f"{xyz}  {rot}  {btn}    ", end='\r')

            iter_count += 1
            time.sleep(0.02)  # 50Hz

        print("\n\nTest completed!")

    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Install libspnav: sudo apt install libspnav-dev")
        print("  2. Check spacenavd: systemctl status spacenavd")
        print("  3. Restart: sudo systemctl restart spacenavd")
        print("  4. Check USB connection")
    finally:
        if sm:
            sm.close()


if __name__ == '__main__':
    main()
