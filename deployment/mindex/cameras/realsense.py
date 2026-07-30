"""RealSense camera — ZMQ server/client.

Camera runs as a separate process (not a thread):

  # terminal 1
  python apps/run_camera_server.py --port 5000 --device-id 318122301129

  # control loop
  cam = CameraClient(host="localhost", port=5000)
  frame = cam.get()  # {"rgb": (H,W,3), "depth": (H,W,1), "timestamp": float}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_GELLO = Path(__file__).resolve().parents[3] / "gello_software"
if str(_GELLO) not in sys.path:
    sys.path.insert(0, str(_GELLO))


class _RealSenseCamera:
    """RealSense camera with automatic YUYV→RGB conversion.

    On USB 2.0 the SDK only negotiates YUYV for color regardless of profile list.
    We request YUYV explicitly and convert to RGB8 via numpy (no cv2 needed).
    Depth is optional; rs.align is skipped (incompatible with YUYV).
    """

    def __init__(self, device_id: Optional[str] = None, flip: bool = False,
                 width: int = 640, height: int = 480, fps: int = 30,
                 depth: bool = False):
        import pyrealsense2 as rs
        self._flip = flip
        self._device_id = device_id
        self._depth = depth
        self._width = width
        self._height = height

        pipeline = rs.pipeline()
        cfg = rs.config()
        if device_id is not None:
            cfg.enable_device(device_id)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.yuyv, fps)
        if depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        pipeline.start(cfg)
        self._pipeline = pipeline

    @staticmethod
    def _yuyv_to_rgb(frame, h: int, w: int) -> np.ndarray:
        raw = np.frombuffer(frame.get_data(), dtype=np.uint8).reshape(h, w * 2)
        Y = raw[:, 0::2].astype(np.float32)
        U = np.repeat(raw[:, 1::4], 2, axis=1).astype(np.float32) - 128.0
        V = np.repeat(raw[:, 3::4], 2, axis=1).astype(np.float32) - 128.0
        R = np.clip(Y + 1.402 * V,                    0, 255).astype(np.uint8)
        G = np.clip(Y - 0.344136 * U - 0.714136 * V,  0, 255).astype(np.uint8)
        B = np.clip(Y + 1.772 * U,                    0, 255).astype(np.uint8)
        return np.stack([R, G, B], axis=-1)

    def read(self, img_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        rgb = self._yuyv_to_rgb(frames.get_color_frame(), self._height, self._width)
        depth = (np.asanyarray(frames.get_depth_frame().get_data())[:, :, None]
                 if self._depth else np.zeros((*rgb.shape[:2], 1), dtype=np.uint16))

        if self._flip:
            rgb   = rgb[::-1, ::-1]
            depth = depth[::-1, ::-1]

        if img_size is not None:
            from PIL import Image
            rgb   = np.array(Image.fromarray(rgb).resize(img_size))
            depth = np.array(Image.fromarray(depth[:, :, 0]).resize(img_size))[:, :, None]

        return rgb, depth

    def __repr__(self):
        return f"RealSenseCamera(device_id={self._device_id})"


class _RobustCamera:
    """Returns last good frame on transient read errors instead of crashing."""

    def __init__(self, device_id, flip, depth=False):
        self._cam = _RealSenseCamera(device_id=device_id, flip=flip, depth=depth)
        self._last: tuple | None = None

    def read(self, img_size=None):
        try:
            self._last = self._cam.read(img_size)
        except RuntimeError as e:
            print(f"warning: camera read error ({e}); using last good frame", file=sys.stderr)
            if self._last is None:
                raise
        return self._last

    def __str__(self):
        return str(self._cam)


def serve(port: int = 5000, device_id: Optional[str] = None, flip: bool = False,
          depth: bool = False) -> None:
    """Serve RealSense frames over ZMQ. Blocks until Ctrl-C.

    Frames are read in a background thread so ZMQ replies are instant —
    the client always gets the latest buffered frame without waiting for
    the next hardware exposure.
    """
    import pickle
    import threading
    import zmq

    cam = _RobustCamera(device_id=device_id, flip=flip, depth=depth)

    # seed with first frame so buffer is never empty
    _buf: list = [cam.read()]
    _lock = threading.Lock()

    def _reader():
        while True:
            frame = cam.read()
            with _lock:
                _buf[0] = frame

    threading.Thread(target=_reader, daemon=True).start()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{port}")
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    print(f"Camera server on port {port}  depth={'on' if depth else 'off'}. Ctrl-C to stop.")
    while True:
        try:
            msg = sock.recv()
            img_size = pickle.loads(msg)
            with _lock:
                frame = _buf[0]
            if img_size is not None:
                from PIL import Image
                rgb = np.array(Image.fromarray(frame[0]).resize(img_size))
                dep = np.array(Image.fromarray(frame[1][:, :, 0]).resize(img_size))[:, :, None]
                frame = (rgb, dep)
            sock.send(pickle.dumps(frame))
        except zmq.Again:
            pass


class CameraClient:
    """ZMQ client that reads frames from a running camera server."""

    def __init__(self, host: str = "localhost", port: int = 5000,
                 img_size: Optional[Tuple[int, int]] = None):
        from gello.zmq_core.camera_node import ZMQClientCamera
        self._client = ZMQClientCamera(port=port, host=host)
        self._img_size = img_size

    def get(self) -> dict | None:
        """Return latest frame or None on error.

        Keys:
          - rgb:       (H, W, 3) uint8
          - depth:     (H, W, 1) uint16
          - timestamp: float
        """
        try:
            rgb, depth = self._client.read(self._img_size)
            return {"rgb": rgb, "depth": depth, "timestamp": time.time()}
        except Exception:
            return None
