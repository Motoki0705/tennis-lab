"""Transparent-background raster rendering for the Court review UI."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.tasks.court_detection.visualization.rendering.common import (
    COURT_SEG_PALETTE_RGB,
)

RgbaArray: TypeAlias = NDArray[np.uint8]
LabelArray: TypeAlias = NDArray[np.integer[Any]]
RgbColor = tuple[int, int, int]

# The canonical 7 ordered court-cell classes shared by both Court sources.
COURT_SEG_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "service_left",
    "service_right",
    "back_left",
    "back_right",
    "doubles_left",
    "doubles_right",
)

# One distinct colour per camera-view semantic line class (0 = background).
SEMANTIC_LINE_PALETTE_RGB: tuple[RgbColor, ...] = (
    (0, 0, 0),
    (235, 87, 87),
    (242, 153, 74),
    (242, 201, 76),
    (111, 207, 151),
    (86, 204, 242),
    (47, 128, 237),
    (155, 81, 224),
    (235, 87, 219),
    (255, 255, 255),
    (109, 76, 65),
    (155, 155, 155),
)


@dataclass(frozen=True, slots=True)
class RasterLegendEntry:
    """One raster colour key surfaced to the frontend."""

    label: str
    color: str

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "color": self.color}


@dataclass(frozen=True, slots=True)
class Raster:
    """A transparent-background colour overlay plus its optional legend."""

    name: str
    data_url: str
    legend: tuple[RasterLegendEntry, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "data": self.data_url,
            "legend": [entry.to_dict() for entry in self.legend],
        }


def _hex_color(color: RgbColor) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def encode_png_data_url(rgba: RgbaArray) -> str:
    """Encode an ``(H, W, 4)`` RGBA array as a ``data:image/png;base64`` URL."""
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError("Raster images must be uint8 RGBA arrays shaped (H, W, 4).")
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def categorical_raster(
    labels: LabelArray,
    *,
    palette: Sequence[RgbColor],
    background_label: int = 0,
    alpha: int = 200,
) -> RgbaArray:
    """Colour a categorical label map, leaving ``background_label`` transparent."""
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError("Categorical rasters require a 2-D label map.")
    if not 0 <= alpha <= 255:
        raise ValueError("Raster alpha must be within [0, 255].")
    if int(array.min(initial=0)) < 0 or int(array.max(initial=0)) >= len(palette):
        raise ValueError("Categorical raster labels exceed the palette.")
    height, width = array.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    for index, color in enumerate(palette):
        if index == background_label:
            continue
        selected = array == index
        if not bool(selected.any()):
            continue
        rgba[selected, 0] = color[0]
        rgba[selected, 1] = color[1]
        rgba[selected, 2] = color[2]
        rgba[selected, 3] = alpha
    return rgba


def binary_raster(
    mask: NDArray[np.bool_],
    *,
    color: RgbColor = (235, 60, 60),
    alpha: int = 200,
) -> RgbaArray:
    """Colour a boolean mask with a transparent background."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Binary rasters require a 2-D mask.")
    if not 0 <= alpha <= 255:
        raise ValueError("Raster alpha must be within [0, 255].")
    height, width = array.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[array, 0] = color[0]
    rgba[array, 1] = color[1]
    rgba[array, 2] = color[2]
    rgba[array, 3] = alpha
    return rgba


def probability_raster(
    probability: NDArray[np.floating[object]],
    *,
    color: RgbColor = (235, 60, 60),
) -> RgbaArray:
    """Colour a ``[0, 1]`` probability map with alpha proportional to confidence."""
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Probability rasters require a 2-D map.")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Probability rasters require finite values.")
    clipped = np.clip(array, 0.0, 1.0)
    height, width = clipped.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = np.rint(clipped * 200.0).astype(np.uint8)
    return rgba


def segmentation_raster(mask: LabelArray) -> Raster:
    """Build the court-cell SEG raster plus its class legend."""
    # The shared palette is the categorical colour table reused by every Court
    # head; the SEG contract uses its first ``len(CLASS_NAMES)`` entries.
    palette = COURT_SEG_PALETTE_RGB[: len(COURT_SEG_CLASS_NAMES)]
    rgba = categorical_raster(mask, palette=palette)
    return Raster(
        name="seg",
        data_url=encode_png_data_url(rgba),
        legend=_legend(palette, COURT_SEG_CLASS_NAMES),
    )


def semantic_line_raster(
    mask: LabelArray,
    channel_names: Sequence[str],
) -> Raster:
    """Build the semantic-line raster plus its per-class legend."""
    if len(channel_names) != len(SEMANTIC_LINE_PALETTE_RGB):
        raise ValueError("Semantic-line channel names must match the palette.")
    rgba = categorical_raster(mask, palette=SEMANTIC_LINE_PALETTE_RGB)
    return Raster(
        name="semantic_line",
        data_url=encode_png_data_url(rgba),
        legend=_legend(SEMANTIC_LINE_PALETTE_RGB, tuple(channel_names)),
    )


def line_raster(mask: NDArray[np.bool_]) -> Raster:
    """Build the binary court-line raster."""
    return Raster(name="line", data_url=encode_png_data_url(binary_raster(mask)))


def line_probability_raster(probability: NDArray[np.floating[object]]) -> Raster:
    """Build the binary court-line raster from a ``[0, 1]`` probability map."""
    return Raster(
        name="line",
        data_url=encode_png_data_url(probability_raster(probability)),
    )


def heatmap_raster(heatmap: NDArray[np.floating[object]]) -> Raster:
    """Build a keypoint heatmap raster with alpha proportional to intensity."""
    array = np.asarray(heatmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Heatmap rasters require a 2-D map.")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Heatmap rasters require finite values.")
    clipped = np.clip(array, 0.0, 1.0)
    height, width = clipped.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 214
    rgba[..., 2] = 64
    rgba[..., 3] = np.rint(clipped * 200.0).astype(np.uint8)
    return Raster(name="heatmap", data_url=encode_png_data_url(rgba))


def _legend(
    palette: Sequence[RgbColor],
    names: Sequence[str],
) -> tuple[RasterLegendEntry, ...]:
    if len(palette) != len(names):
        raise ValueError("Legend palettes and names must agree.")
    return tuple(
        RasterLegendEntry(label=name, color=_hex_color(color))
        for color, name in zip(palette, names, strict=True)
        if name != "background"
    )


__all__ = [
    "COURT_SEG_CLASS_NAMES",
    "SEMANTIC_LINE_PALETTE_RGB",
    "LabelArray",
    "Raster",
    "RasterLegendEntry",
    "RgbaArray",
    "binary_raster",
    "categorical_raster",
    "encode_png_data_url",
    "heatmap_raster",
    "line_probability_raster",
    "line_raster",
    "probability_raster",
    "segmentation_raster",
    "semantic_line_raster",
]
