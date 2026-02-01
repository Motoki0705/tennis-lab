"""Video loading helpers for the annotation backend."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for a video file."""

    fps: float
    frame_count: int
    width: int
    height: int


class VideoFrameProvider:
    """Random-access frame provider with a small in-memory cache."""

    def __init__(self, video_path: str | Path, cache_size: int = 16) -> None:
        self._video_path = str(video_path)
        self._lock = threading.Lock()
        self._cache_size = int(cache_size)
        self._cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {self._video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()

        if frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid video metadata for {self._video_path}: "
                f"frames={frame_count}, w={width}, h={height}"
            )
        self._info = VideoInfo(
            fps=fps if fps > 0 else 30.0,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    def _cache_get(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx in self._cache:
            # LRU refresh
            try:
                self._cache_order.remove(frame_idx)
            except ValueError:
                pass
            self._cache_order.append(frame_idx)
            return self._cache[frame_idx]
        return None

    def _cache_put(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        if frame_idx in self._cache:
            self._cache[frame_idx] = frame_bgr
            try:
                self._cache_order.remove(frame_idx)
            except ValueError:
                pass
            self._cache_order.append(frame_idx)
            return

        self._cache[frame_idx] = frame_bgr
        self._cache_order.append(frame_idx)
        while len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def read_frame_bgr(self, frame_idx: int) -> np.ndarray:
        """Read a single frame in BGR format (OpenCV)."""
        if frame_idx < 0 or frame_idx >= self._info.frame_count:
            raise IndexError(
                f"frame_idx out of range: {frame_idx} (0..{self._info.frame_count - 1})"
            )

        with self._lock:
            cached = self._cache_get(frame_idx)
            if cached is not None:
                return cached.copy()

            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Failed to open video: {self._video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to decode frame {frame_idx} from video")
            self._cache_put(frame_idx, frame_bgr)
            return frame_bgr.copy()

    def encode_jpeg(self, frame_idx: int, quality: int = 90) -> bytes:
        """Encode a frame as JPEG."""
        frame_bgr = self.read_frame_bgr(frame_idx)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"Failed to encode frame {frame_idx} as JPEG")
        return bytes(buf.tobytes())

