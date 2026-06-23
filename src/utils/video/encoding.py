"""Video frame selection and image encoding helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.video.reader import OpenCVVideoFrameReader


def encode_jpeg(frame_bgr: NDArray[np.uint8], *, quality: int = 90) -> bytes:
    """Encode one BGR uint8 frame as JPEG bytes."""
    if not 0 <= quality <= 100:
        raise ValueError(f"quality must be in [0, 100], got {quality}.")
    encoded, buffer = cv2.imencode(
        ".jpg",
        frame_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not encoded:
        raise RuntimeError("Failed to JPEG-encode frame.")
    return buffer.tobytes()


def iter_selected_video_jpegs(
    parts: Sequence[str | Path],
    frame_indices: Collection[int],
    *,
    quality: int = 90,
) -> Iterator[tuple[int, bytes]]:
    """Yield selected frames from virtually concatenated videos as JPEG bytes."""
    remaining = {int(index) for index in frame_indices}
    if any(index < 0 for index in remaining):
        raise ValueError("frame_indices must be non-negative.")

    base = 0
    for part in parts:
        if not remaining:
            break
        decoded_count = 0
        for packet in OpenCVVideoFrameReader(part):
            decoded_count = packet.index + 1
            global_index = base + packet.index
            if global_index not in remaining:
                continue
            yield global_index, encode_jpeg(packet.frame, quality=quality)
            remaining.remove(global_index)
        base += decoded_count


__all__ = ["encode_jpeg", "iter_selected_video_jpegs"]
