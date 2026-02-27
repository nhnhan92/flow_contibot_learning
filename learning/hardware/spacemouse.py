"""
Custom SpaceMouse Controller using libspnav directly

Axis Mapping (SpaceMouse Raw → Robot):
    Linear:
        SpaceMouse X → Robot X (left/right)
        SpaceMouse Z → Robot Y (forward/backward)
        SpaceMouse Y → Robot Z (up/down)

    Rotation:
        SpaceMouse RX → Robot RX (roll)
        SpaceMouse RZ → Robot RY (pitch)
        SpaceMouse RY → Robot RZ (yaw/twist)

Note: RX và RY hiện tại đang bị disable, chỉ có RZ (yaw) hoạt động.
"""

import ctypes
from ctypes import c_int, c_uint, Structure, POINTER, byref
import numpy as np
import threading
from typing import Optional
import os
import sys
FILE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.insert(0, PARENT_DIR)


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

# -------------------------
# SpaceMouse adapter (Linux vs Windows)
# -------------------------
import platform

class _BaseSpaceMouse:
    """Threaded SpaceMouse reader with cached latest xyz."""
    def __init__(self, device_index: int):
        self.device_index = device_index
        self._lock = threading.Lock()
        self._latest_xyz = np.zeros(3, dtype=float)
        self._button_status = [False,False]
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def get_latest_xyz(self) -> np.ndarray:
        with self._lock:
            return self._latest_xyz.copy()

    def get_button_status(self):
        with self._lock:
            return self._button_status.copy()
    # subclass must implement this
    def _read_xyz_once(self) -> np.ndarray:
        raise NotImplementedError

    def _run(self):
        # choose a polling rate. 200 Hz is usually enough.
        dt = 1.0 / 200.0
        while self._running.is_set():
            try:
                xyz,button_status = self._read_xyz_once()
                xyz = np.asarray(xyz, dtype=float).reshape(3,)
                with self._lock:
                    self._latest_xyz[:] = xyz
                    self._button_status = button_status
            except Exception:
                # don’t kill thread on transient read errors
                pass
            # tiny sleep to avoid 100% CPU
            threading.Event().wait(dt)


def _build_spacemouse(device_index: int = 1,os_name: str = 'linux',deadzone : float = 0.1, max_value: float = 350.0) -> _BaseSpaceMouse:
    os_name = platform.system().lower()

    # Linux: libspnav spacemouse.py
    if "linux" in os_name:

        class _LinuxSpaceMouse(_BaseSpaceMouse):
            def __init__(self, device_index: int):
                super().__init__(device_index)
                self._sm = SpaceMouse(deadzone=deadzone, max_value=max_value)

            def _read_xyz_once(self) -> np.ndarray:
                state6 = self._sm.get_motion_state_transformed()  # [x,y,z,rx,ry,rz]
                xyz = np.asarray(state6[:3], dtype=float)
                # axis/sign convention (edit if needed)
                button_status = self._sm.get_all_buttons()
                return np.array([xyz[0], xyz[1], xyz[2]], dtype=float),button_status

            def stop(self):
                super().stop()
                try:
                    self._sm.close()
                except Exception:
                    pass

        sm = _LinuxSpaceMouse(device_index)
        sm.start()
        return sm

    # Windows: pyspacemouse spacemouse_control.py
    if "windows" in os_name:
        from flowbot.spacemouse_control import SpaceMouseReader as _SpaceMouseReaderWin

        class _WindowsSpaceMouse(_BaseSpaceMouse):
            def __init__(self, device_index: int):
                super().__init__(device_index)
                self._reader = _SpaceMouseReaderWin(device_index=device_index)
                self._reader.start()

            def _read_xyz_once(self) -> np.ndarray:
                xyz = np.asarray(self._reader.get_latest_xyz(), dtype=float)
                return np.array([xyz[0], xyz[1], -xyz[2]], dtype=float)

            def stop(self):
                super().stop()
                try:
                    self._reader.stop()
                except Exception:
                    pass

        sm = _WindowsSpaceMouse(device_index)
        sm.start()
        return sm

    raise RuntimeError(f"Unsupported OS: {platform.system()}")


class SpaceMouse:
    """
    SpaceMouse controller using libspnav directly.

    Axis mapping đã được calibrate cho robot UR5e:
        Linear:
            X = left/right (SpaceMouse X)
            Y = forward/backward (SpaceMouse Z)
            Z = up/down (SpaceMouse Y)
        Rotation:
            RX = roll (SpaceMouse RX) - disabled
            RY = pitch (SpaceMouse RZ) - disabled
            RZ = yaw/twist (SpaceMouse RY)

    Usage:
        with SpaceMouse() as sm:
            state = sm.get_motion_state_transformed()
            # state = [x, y, z, rx, ry, rz] normalized to [-1, 1]

            if sm.is_button_pressed(0):  # Left button
                print("Left button pressed")
    """

    def __init__(self, deadzone: float = 0.1, max_value: float = 350.0):
        """
        Initialize SpaceMouse

        Args:
            deadzone: Values below this threshold are set to 0
            max_value: Raw value for normalization (output will be [-1, 1])
        """
        self.deadzone = deadzone
        self.max_value = max_value
        self.lib = None

        # Raw state from SpaceMouse
        self._x = self._y = self._z = 0
        self._rx = self._ry = self._rz = 0
        self._buttons = [False, False]

        # Load libspnav
        try:
            self.lib = ctypes.CDLL("libspnav.so")
        except OSError:
            try:
                self.lib = ctypes.CDLL("libspnav.so.0")
            except OSError:
                raise RuntimeError(
                    "Cannot load libspnav. Install: sudo apt install libspnav-dev"
                )

        # Setup function signatures
        self.lib.spnav_open.restype = c_int
        self.lib.spnav_close.restype = c_int
        self.lib.spnav_poll_event.argtypes = [POINTER(SpnavEvent)]
        self.lib.spnav_poll_event.restype = c_int

        # Open connection
        if self.lib.spnav_open() == -1:
            raise RuntimeError(
                "Cannot connect to spacenavd. "
                "Check: systemctl status spacenavd"
            )

        print("SpaceMouse connected (libspnav)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _poll(self):
        """Poll for events and update internal state"""
        event = SpnavEvent()

        while self.lib.spnav_poll_event(byref(event)) != 0:
            if event.type == SPNAV_EVENT_MOTION:
                self._x = event.motion.x
                self._y = event.motion.y
                self._z = event.motion.z
                self._rx = event.motion.rx
                self._ry = event.motion.ry
                self._rz = event.motion.rz
            elif event.type == SPNAV_EVENT_BUTTON:
                if event.button.bnum < len(self._buttons):
                    self._buttons[event.button.bnum] = bool(event.button.press)

    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to normalized value"""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_motion_state_transformed(self) -> np.ndarray:
        """
        Get motion state with correct axis mapping for robot control.

        Returns:
            np.ndarray: [x, y, z, rx, ry, rz] normalized to approximately [-1, 1]

        Mapping:
            Robot X = SpaceMouse X (left/right)
            Robot Y = SpaceMouse Z (forward/backward)
            Robot Z = SpaceMouse Y (up/down)
            Robot RX = disabled (0.0)
            Robot RY = disabled (0.0)
            Robot RZ = SpaceMouse RY (yaw/twist)
        """
        self._poll()

        # Linear axes: SpaceMouse → Robot
        x = self._apply_deadzone(self._x / self.max_value)   # SM X → Robot X
        y = self._apply_deadzone(self._z / self.max_value)   # SM Z → Robot Y
        z = self._apply_deadzone(self._y / self.max_value)   # SM Y → Robot Z

        # Rotation axes: SpaceMouse → Robot
        # Full mapping (if enabled):
        #   Robot RX = SpaceMouse RX
        #   Robot RY = SpaceMouse RZ
        #   Robot RZ = SpaceMouse RY
        rx = self._apply_deadzone(self._rx / self.max_value)
        ry = self._apply_deadzone(self._ry / self.max_value)   # disabled for safety
        rz = self._apply_deadzone(self._rz / self.max_value)  # SM RY → Robot RZ (yaw)

        return np.array([x, y, z, rx, ry, rz], dtype=np.float64)

    def get_raw_state(self) -> dict:
        """
        Get raw SpaceMouse values (for debugging)

        Returns:
            dict with raw x, y, z, rx, ry, rz values
        """
        self._poll()
        return {
            'x': self._x,
            'y': self._y,
            'z': self._z,
            'rx': self._rx,
            'ry': self._ry,
            'rz': self._rz,
        }

    def is_button_pressed(self, button: int) -> bool:
        """
        Check if a button is currently pressed.

        Args:
            button: Button index (0 = left, 1 = right)

        Returns:
            True if button is pressed
        """
        self._poll()
        if 0 <= button < len(self._buttons):
            return self._buttons[button]
        return False

    def get_all_buttons(self) -> list:
        """Get state of all buttons"""
        self._poll()
        return self._buttons.copy()

    def close(self):
        """Close SpaceMouse connection"""
        if self.lib:
            self.lib.spnav_close()
            print("SpaceMouse disconnected")


def test_spacemouse():
    """Quick test function"""
    import time

    print("="*60)
    print("       SPACEMOUSE TEST")
    print("="*60)
    print("\nAxis Mapping:")
    print("  Robot X = SpaceMouse X (left/right)")
    print("  Robot Y = SpaceMouse Z (forward/backward)")
    print("  Robot Z = SpaceMouse Y (up/down)")
    print("  Robot RZ = SpaceMouse RY (yaw/twist)")
    print("\nTesting for 10 seconds...")
    print("Move the SpaceMouse and press buttons.\n")

    with SpaceMouse(deadzone=0.1) as sm:
        for i in range(100):
            state = sm.get_motion_state_transformed()
            raw = sm.get_raw_state()
            btn_l = sm.is_button_pressed(0)
            btn_r = sm.is_button_pressed(1)

            # Show both raw and transformed values
            print(f"Robot:[X={state[0]:+.2f} Y={state[1]:+.2f} Z={state[2]:+.2f} "
                  f"RZ={state[5]:+.2f}] | "
                  f"Raw:[x={raw['x']:+4d} y={raw['y']:+4d} z={raw['z']:+4d}] | "
                  f"Btn: L={int(btn_l)} R={int(btn_r)}   ", end='\r')

            time.sleep(0.1)

    print("\n\nTest completed!")


if __name__ == '__main__':
    test_spacemouse()
