"""
video_recorder.py — Reusable recorders for matplotlib figures and USB cameras.

Classes
-------
VideoRecorder   — captures matplotlib figure frames → MP4 / GIF
CameraRecorder  — captures USB camera frames via OpenCV → MP4 (with live preview)

Usage
-----
    from flowbot.video_recorder import VideoRecorder, CameraRecorder

    # Figure recording
    recorder = VideoRecorder("output.mp4", fps=15, fig=my_fig)
    recorder.capture()   # call each render tick
    recorder.close()

    # Camera recording
    cam = CameraRecorder("camera.mp4", camera_id=0, fps=15)
    cam.start()
    ...
    cam.stop()
"""
from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import numpy as np

try:
    import cv2 as _cv2
    _CV2_OK = True
except ImportError:
    _cv2 = None
    _CV2_OK = False


class VideoRecorder:
    """
    Captures matplotlib figure frames and writes them to a video/GIF file.

    Priority:
      1. MP4 via imageio+ffmpeg  (pip install imageio imageio-ffmpeg)
      2. Animated GIF via Pillow (already installed — no extra deps)

    Output path should end in .mp4 or .gif; .gif is used automatically
    if imageio-ffmpeg is not available.
    """

    def __init__(self, path: str, fps: float = 15.0, fig=None):
        self._path           = path
        self._fps            = fps
        self._fig            = fig
        self._frames         = []   # PIL Image list (GIF mode)
        self._writer         = None
        self._frame_idx      = 0
        self._mode           = None   # "imageio" | "gif"
        self._frame_interval  = 1.0 / fps   # min seconds between captured frames
        self._last_capture_t  = -1e9       # force capture on first call
        self._capture_times   = []         # wall-clock time of each captured frame

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # ── Try MP4 via imageio ──────────────────────────────────
        try:
            import imageio
            # imageio v3 uses imageio.v3; v2 uses imageio directly — try both
            try:
                writer = imageio.get_writer(
                    path, fps=fps, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p"],
                    macro_block_size=1,
                )
            except TypeError:
                writer = imageio.get_writer(path, fps=fps)
            self._writer = writer
            self._mode   = "imageio"
            print(f"[video] Recording MP4 → {path}  ({fps:.0f} fps)")
            return
        except Exception as e:
            print(f"[video] imageio MP4 unavailable ({e}). Falling back to GIF.")

        # ── Fallback: animated GIF via Pillow ────────────────────
        try:
            if importlib.util.find_spec("PIL") is None:
                raise ImportError("Pillow not found")
            gif_path = Path(path).with_suffix(".gif")
            self._path = str(gif_path)
            self._mode = "gif"
            print(f"[video] Recording animated GIF → {self._path}  ({fps:.0f} fps)")
            print(f"[video] Tip: install imageio-ffmpeg for MP4:  "
                  f"pip install imageio imageio-ffmpeg")
        except ImportError:
            print("[video] Neither imageio nor Pillow available — recording disabled.")

    def _grab_frame(self):
        """Return current figure canvas as an RGB numpy array (even dimensions for H.264)."""
        rgba_buf = self._fig.canvas.buffer_rgba()
        # physical=True accounts for Windows DPI scaling (e.g. 125% → 1.25× buffer)
        canvas_w, canvas_h = self._fig.canvas.get_width_height(physical=True)
        frame = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(canvas_h, canvas_w, 4)
        rgb = frame[..., :3]   # drop alpha → RGB
        # H.264 requires even dimensions — crop 1px if odd
        h, w = rgb.shape[:2]
        return rgb[:h & ~1, :w & ~1]

    def capture(self):
        """Grab one frame from the figure canvas (rate-limited to self._fps)."""
        if self._fig is None or self._mode is None:
            return
        now = time.perf_counter()
        if now - self._last_capture_t < self._frame_interval:
            return   # too soon — skip this tick
        self._last_capture_t = now
        try:
            frame = self._grab_frame()
            if self._mode == "imageio":
                self._writer.append_data(frame)
            elif self._mode == "gif":
                from PIL import Image
                self._frames.append(Image.fromarray(frame))
            self._capture_times.append(now)
            self._frame_idx += 1
        except Exception as exc:
            # Print first failure only to avoid log spam
            if self._frame_idx == 0:
                print(f"[video] Frame capture error: {exc}")

    def _actual_fps_str(self):
        if len(self._capture_times) >= 2:
            actual = (len(self._capture_times) - 1) / \
                     (self._capture_times[-1] - self._capture_times[0])
            return f"{actual:.1f} fps actual"
        return ""

    def _correct_mp4_speed(self, actual_fps: float):
        """Rewrite the MP4 with setpts so playback matches real wall-clock time."""
        ratio = self._fps / actual_fps   # > 1 → slow down; < 1 → speed up
        if abs(ratio - 1.0) <= 0.05:
            return   # within 5% — no correction needed
        try:
            import imageio_ffmpeg, subprocess, os
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            tmp = self._path + ".tmp.mp4"
            cmd = [
                ffmpeg_exe, "-y", "-i", self._path,
                "-vf", f"setpts={ratio:.6f}*PTS",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                tmp,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                os.replace(tmp, self._path)
                print(f"[video] Speed-corrected: {actual_fps:.1f} fps actual → "
                      f"video plays at real-time  (stretched {ratio:.2f}×)")
            else:
                print(f"[video] Speed correction failed; video plays at "
                      f"{actual_fps:.1f}/{self._fps:.0f} = "
                      f"{actual_fps/self._fps:.2f}× real speed")
                if os.path.exists(tmp):
                    os.remove(tmp)
        except Exception as exc:
            print(f"[video] Speed correction skipped ({exc}); "
                  f"video plays {self._fps/actual_fps:.2f}× real speed")

    def close(self):
        if self._mode == "imageio" and self._writer is not None:
            self._writer.close()
            if len(self._capture_times) >= 2:
                actual_fps = (len(self._capture_times) - 1) / \
                             (self._capture_times[-1] - self._capture_times[0])
                self._correct_mp4_speed(actual_fps)
            print(f"[video] Saved {self._frame_idx} frames → {self._path}"
                  f"  ({self._actual_fps_str()})")
        elif self._mode == "gif" and self._frames:
            # Use actual inter-frame intervals for correct real-time playback.
            # GIF stores delay in centiseconds (10 ms steps); minimum is 20 ms.
            if len(self._capture_times) >= 2:
                durations = []
                for i in range(1, len(self._capture_times)):
                    dt_ms = (self._capture_times[i] - self._capture_times[i - 1]) * 1000
                    # round to nearest 10 ms (centisecond resolution), floor at 20 ms
                    durations.append(max(20, round(dt_ms / 10) * 10))
                durations.append(durations[-1])   # repeat duration for last frame
            else:
                durations = int(1000 / self._fps)
            self._frames[0].save(
                self._path,
                save_all=True,
                append_images=self._frames[1:],
                duration=durations,
                loop=0,
            )
            print(f"[video] Saved {self._frame_idx} frames → {self._path}"
                  f"  ({self._actual_fps_str()})")


# ──────────────────────────────────────────────────────────────────────────────
# USB Camera Recorder
# ──────────────────────────────────────────────────────────────────────────────
class CameraRecorder:
    """
    Records frames from a USB camera in a background thread using OpenCV,
    with a live preview window.  Uses the same fps as VideoRecorder so both
    streams are temporally aligned.

    Parameters
    ----------
    path      : output MP4 path
    camera_id : OpenCV device index (default 0)
    fps       : target recording fps (default 15)
    stop_flag : optional shared dict {"stop": bool}; pressing Q in the preview
                window sets stop_flag["stop"] = True so the caller can react.

    Usage
    -----
        cam = CameraRecorder("camera.mp4", camera_id=0, fps=15, stop_flag=sf)
        cam.start()
        ...
        cam.stop()
    """

    def __init__(self, path: str, camera_id: int = 0, fps: float = 15.0,
                 stop_flag: dict = None, show_preview: bool = False):
        if not _CV2_OK:
            raise ImportError("opencv-python is required: pip install opencv-python")
        self._path      = path
        self._camera_id = camera_id
        self._fps       = fps
        self._stop_flag = stop_flag
        self._stop_evt    = threading.Event()
        self._thread: threading.Thread | None = None
        self._show_preview = show_preview
        self._win_name    = "USB Camera  (Q to stop)"

    def start(self):
        """Launch the recording thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the thread to stop, wait for it, then destroy the preview window (main thread)."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        # destroyWindow must be called from the main thread on Windows
        if self._show_preview:
            try:
                _cv2.destroyWindow(self._win_name)
            except Exception:
                pass

    def _run(self):
        cap = _cv2.VideoCapture(self._camera_id)
        if not cap.isOpened():
            print(f"[camera] Could not open camera {self._camera_id}")
            return

        w      = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _cv2.VideoWriter_fourcc(*"mp4v")

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        writer = _cv2.VideoWriter(self._path, fourcc, self._fps, (w, h))
        print(f"[camera] Recording {w}×{h} @ {self._fps:.0f} fps → {self._path}")

        win_name = self._win_name
        n_frames = 0
        t_start  = time.perf_counter()
        interval = 1.0 / self._fps

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            if self._show_preview:
                _cv2.imshow(win_name, frame)
            key = _cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self._stop_evt.set()
                if self._stop_flag is not None:
                    self._stop_flag["stop"] = True
            n_frames += 1
            # rate-limit to target fps
            elapsed   = time.perf_counter() - t_start
            remaining = n_frames * interval - elapsed
            if remaining > 0.001:
                time.sleep(remaining)

        writer.release()
        cap.release()
        elapsed_total = time.perf_counter() - t_start
        actual_fps    = n_frames / elapsed_total if elapsed_total > 0 else 0
        print(f"[camera] Saved {n_frames} frames → {self._path}"
              f"  ({actual_fps:.1f} fps actual)")