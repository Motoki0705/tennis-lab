"""OpenCV-backed video frame readers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.video.types import FramePacket, VideoInfo


def probe_video_info(video_path: str | Path) -> VideoInfo:
    """Read video metadata without decoding all frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        return VideoInfo(
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_video_frame(
    video_path: str | Path, frame_index: int
) -> FramePacket[NDArray[np.uint8]]:
    """Read one decoded BGR frame by source frame index."""
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        frame_bgr = cast(NDArray[np.uint8], np.asarray(frame_bgr, dtype=np.uint8))
        original_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        return FramePacket(
            index=frame_index,
            frame=frame_bgr,
            original_size=original_size,
        )
    finally:
        cap.release()


class OpenCVVideoFrameReader:
    """Stream decoded BGR frames from a video file."""

    def __init__(
        self, video_path: str | Path, *, max_frames: int | None = None
    ) -> None:
        self.video_path = Path(video_path)
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[FramePacket[NDArray[np.uint8]]]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        try:
            frame_index = 0
            while self.max_frames is None or frame_index < self.max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_bgr = cast(
                    NDArray[np.uint8], np.asarray(frame_bgr, dtype=np.uint8)
                )
                original_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
                yield FramePacket(
                    index=frame_index,
                    frame=frame_bgr,
                    original_size=original_size,
                )
                frame_index += 1
        finally:
            cap.release()


class RandomAccessVideoReader:
    """Random-access BGR frame reader with a grab-ahead seek shortcut.

    Keeps one ``cv2.VideoCapture`` open across calls. Small forward jumps are
    served by ``grab()`` (cheap sequential decode); backward jumps and large
    forward jumps fall back to an absolute seek. This makes both scrubbing and
    monotonically increasing access patterns efficient on long videos.

    Args:
        video_path: Source video file.
        seek_grab_threshold: Maximum forward distance (in frames) bridged by
            sequential ``grab()`` calls before an absolute seek is used.
    """

    def __init__(
        self, video_path: str | Path, *, seek_grab_threshold: int = 24
    ) -> None:
        if seek_grab_threshold < 0:
            raise ValueError(
                f"seek_grab_threshold must be non-negative, got {seek_grab_threshold}"
            )
        self.video_path = Path(video_path)
        self.seek_grab_threshold = seek_grab_threshold
        self._cap: cv2.VideoCapture | None = None
        self._next_index = 0

    def read(self, frame_index: int) -> NDArray[np.uint8]:
        """Decode and return the BGR frame at ``frame_index``."""
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")
        if self._cap is None:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            self._cap = cap
            self._next_index = 0
        cap = self._cap

        distance = frame_index - self._next_index
        if 0 < distance <= self.seek_grab_threshold:
            for skipped in range(distance):
                if not cap.grab():
                    raise RuntimeError(
                        f"Failed to grab frame {self._next_index + skipped} "
                        f"from {self.video_path}"
                    )
        elif distance != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(
                f"Failed to read frame {frame_index} from {self.video_path}"
            )
        self._next_index = frame_index + 1
        return np.asarray(frame_bgr, dtype=np.uint8)

    def close(self) -> None:
        """Release the underlying capture (idempotent)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> RandomAccessVideoReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_video_rgb(
    video_path: str | Path,
    *,
    max_frames: int | None = None,
    scale: float = 1.0,
) -> NDArray[np.uint8]:
    """Decode a whole video into an RGB uint8 array of shape (N, H, W, 3).

    Args:
        video_path: Source video file.
        max_frames: Optional cap on the number of decoded frames.
        scale: Uniform spatial scale factor applied to every frame
            (``INTER_AREA`` for downscaling-friendly resampling).
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    frames: list[NDArray[np.uint8]] = []
    for packet in OpenCVVideoFrameReader(video_path, max_frames=max_frames):
        frame_bgr = packet.frame
        if scale != 1.0:
            frame_bgr = np.asarray(
                cv2.resize(
                    frame_bgr,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                ),
                dtype=np.uint8,
            )
        frames.append(frame_bgr[..., ::-1])
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return np.ascontiguousarray(np.stack(frames, axis=0))
