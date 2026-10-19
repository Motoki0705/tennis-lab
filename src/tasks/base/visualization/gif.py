"""Shared animated-GIF writer for raster-family visualizations.

Used by the clip/frame visualization pipelines (ball detection, court detection)
that composite RGB frames and save them as an animated GIF.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image


def save_gif(
    *,
    frames_rgb: Sequence[np.ndarray],
    path: Path,
    fps: float,
    loop: int = 0,
) -> None:
    """Save rendered RGB frames as an animated GIF.

    Cumulative frame endpoints are rounded to GIF's 10 ms time unit, then
    differenced into individual durations. This bounds timing error below
    10 ms instead of accumulating it at fractional frame intervals (e.g. 15 fps),
    including when Pillow combines identical adjacent images.

    Args:
        frames_rgb: Sequence of ``(H, W, 3)`` uint8 RGB frames.
        path: Output ``.gif`` path.
        fps: Playback frames per second (must be positive).
        loop: GIF loop count (``0`` means loop forever).
    """
    if not frames_rgb:
        raise ValueError("At least one frame is required to save a GIF.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if path.suffix.lower() != ".gif":
        raise ValueError(f"Only .gif outputs are supported, got: {path}")

    # Quantize cumulative endpoints, not each frame's duration: Pillow truncates
    # millisecond durations to 10 ms units, which otherwise accumulates drift.
    durations_ms: list[int] = []
    previous_end_ms = 0
    for index in range(len(frames_rgb)):
        end_ms = round((index + 1) * 100 / fps) * 10
        durations_ms.append(end_ms - previous_end_ms)
        previous_end_ms = end_ms
    pil_frames = [
        Image.fromarray(frame).convert(
            "P",
            palette=Image.Palette.ADAPTIVE,
            colors=256,
        )
        for frame in frames_rgb
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations_ms,
        loop=loop,
        disposal=2,
        optimize=True,
    )
