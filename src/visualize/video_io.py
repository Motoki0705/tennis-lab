"""Video IO helpers for rendering numpy frame streams to disk."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


def write_video(
    out_path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float,
    codec: str = "mp4v",
) -> None:
    """Write a sequence of HxWx3 uint8 frames to a video file."""
    path = Path(out_path)

    frame_iter = iter(frames)
    first = next(frame_iter, None)
    if first is None:
        raise ValueError("write_video() received no frames")

    if first.ndim < 2:
        raise ValueError("Frames must have at least 2 dimensions (H, W)")
    height, width = int(first.shape[0]), int(first.shape[1])

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        msg = f"Failed to open VideoWriter for path: {path}"
        raise RuntimeError(msg)

    try:

        def _write_one(frame: np.ndarray) -> None:
            if frame.shape[0] != height or frame.shape[1] != width:
                msg = (
                    "All frames must have identical spatial dimensions. "
                    f"Expected ({height}, {width}), got ({frame.shape[0]}, {frame.shape[1]})"
                )
                raise ValueError(msg)
            writer.write(frame)

        _write_one(first)
        for frame in frame_iter:
            _write_one(frame)
    finally:
        writer.release()
