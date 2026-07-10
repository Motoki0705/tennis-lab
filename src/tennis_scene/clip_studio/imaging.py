"""Frame-fitting helpers shared by the clip studio renderer and exporter."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LetterboxSpec:
    """How a source frame was fitted into a target rectangle.

    Attributes:
        scale: Uniform scale factor applied to the source frame.
        pad_x, pad_y: Left/top padding in target pixels.
        scaled_width, scaled_height: Size of the scaled content.
    """

    scale: float
    pad_x: int
    pad_y: int
    scaled_width: int
    scaled_height: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "scaled_width": self.scaled_width,
            "scaled_height": self.scaled_height,
        }


def compute_letterbox(
    source_width: int, source_height: int, target_width: int, target_height: int
) -> LetterboxSpec:
    """Compute the aspect-preserving fit of a source into a target rectangle."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"source size must be positive, got {source_width}x{source_height}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"target size must be positive, got {target_width}x{target_height}")
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = max(1, round(source_width * scale))
    scaled_height = max(1, round(source_height * scale))
    pad_x = (target_width - scaled_width) // 2
    pad_y = (target_height - scaled_height) // 2
    return LetterboxSpec(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        scaled_width=scaled_width,
        scaled_height=scaled_height,
    )


def letterbox_frame(
    frame: NDArray[np.uint8],
    target_width: int,
    target_height: int,
    *,
    fill_value: int = 0,
) -> tuple[NDArray[np.uint8], LetterboxSpec]:
    """Fit ``frame`` (H, W, 3) into a target canvas, preserving aspect ratio."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape (H, W, 3), got {frame.shape}")
    height, width = frame.shape[:2]
    spec = compute_letterbox(width, height, target_width, target_height)
    if (spec.scaled_width, spec.scaled_height) == (width, height):
        scaled = frame
    else:
        interpolation = cv2.INTER_AREA if spec.scale < 1.0 else cv2.INTER_LINEAR
        scaled = np.asarray(
            cv2.resize(
                frame,
                (spec.scaled_width, spec.scaled_height),
                interpolation=interpolation,
            ),
            dtype=np.uint8,
        )
    canvas: NDArray[np.uint8] = np.full(
        (target_height, target_width, 3), fill_value, dtype=np.uint8
    )
    canvas[
        spec.pad_y : spec.pad_y + spec.scaled_height,
        spec.pad_x : spec.pad_x + spec.scaled_width,
    ] = scaled
    return canvas, spec


__all__ = ["LetterboxSpec", "compute_letterbox", "letterbox_frame"]
