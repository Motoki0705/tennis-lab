"""Raster overlays for the ball-detection review UI.

The only dense field a ball checkpoint produces is its probability heatmap, so
that is the only raster the backend offers.  Ground-truth rasters are not
invented: GT is a set of labelled points (``FrameLabel``), and the review UI
draws it as such.

The RGBA recipe (alpha proportional to the clipped probability, one fixed hue)
follows the shared convention already used by the court-detection review UI so
that both tasks read the same way in the browser.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image

RgbaArray: TypeAlias = NDArray[np.uint8]
RgbColor: TypeAlias = tuple[int, int, int]

#: Prediction hue; matches the red used for predicted points in the viewer.
PREDICTION_RGB: RgbColor = (255, 98, 133)


@dataclass(frozen=True, slots=True)
class Raster:
    """One transparent-background overlay ready for the browser."""

    name: str
    data_url: str

    def to_dict(self) -> dict[str, object]:
        # ``legend`` is part of the shared frame-layer contract; a single-hue
        # probability overlay has no discrete classes to key.
        return {"name": self.name, "data": self.data_url, "legend": []}


def encode_png_data_url(rgba: RgbaArray) -> str:
    """Encode an ``(H, W, 4)`` RGBA array as a ``data:image/png;base64`` URL."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Raster images must be uint8 RGBA arrays shaped (H, W, 4).")
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def probability_rgba(
    probability: NDArray[np.floating],
    *,
    color: RgbColor = PREDICTION_RGB,
) -> RgbaArray:
    """Colour a ``[0, 1]`` probability map with alpha proportional to intensity."""
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Probability rasters require a 2-D map.")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Probability rasters require finite values.")
    clipped = np.clip(array, 0.0, 1.0)
    height, width = clipped.shape
    rgba: RgbaArray = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    alpha: NDArray[np.uint8] = np.rint(clipped * 200.0).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def probability_raster(
    probability: NDArray[np.floating],
    *,
    name: str = "probability",
) -> Raster:
    """Build the prediction probability overlay at heatmap resolution."""
    return Raster(name=name, data_url=encode_png_data_url(probability_rgba(probability)))


__all__ = [
    "PREDICTION_RGB",
    "Raster",
    "RgbaArray",
    "RgbColor",
    "encode_png_data_url",
    "probability_raster",
    "probability_rgba",
]
