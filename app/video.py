from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2


def _default_backend() -> int:
    """
    macOS: AVFoundation is usually most reliable.
    Other OSes: CAP_ANY is fine.
    """
    try:
        return cv2.CAP_AVFOUNDATION  # type: ignore[attr-defined]
    except Exception:
        return cv2.CAP_ANY


def probe_cameras(max_index: int = 3, backend: Optional[int] = None) -> List[int]:
    """
    Returns camera indices that can be opened and read from.
    Keep max_index small to avoid noisy logs on some systems.
    """
    backend = _default_backend() if backend is None else backend
    available: List[int] = []

    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        cap.release()

        if ok:
            available.append(i)

        time.sleep(0.03)

    return available


@dataclass
class CameraConfig:
    index: int
    width: int = 1280
    height: int = 720
    fps: int = 60
    backend: Optional[int] = None


class VideoStream:
    """
    Threaded capture loop. UI pulls latest frame with get_frame().
    """
    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None
        self._camera_index: Optional[int] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def camera_index(self) -> Optional[int]:
        return self._camera_index

    def open(self, cfg: CameraConfig) -> bool:
        self.close()

        backend = _default_backend() if cfg.backend is None else cfg.backend
        cap = cv2.VideoCapture(cfg.index, backend)

        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._camera_index = cfg.index
        return True

    def start(self) -> None:
        if self._cap is None or self._running:
            return

        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame

            time.sleep(0.001)

    def get_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def close(self) -> None:
        self._running = False
        time.sleep(0.03)

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

        self._cap = None
        self._camera_index = None
        with self._lock:
            self._latest_frame = None

