"""Exact crop/resize geometry; pixel centers differ from bbox edges."""

from fractions import Fraction
from math import isfinite
from typing import Literal

from .contracts import Crop, ImageRect, TileReceipt

Coordinates = Literal["pixel_center", "edge"]


def fitted_rect(
    crop: Crop, *, left: int, top: int, width: int, height: int
) -> ImageRect:
    if min(left, top) < 0 or min(width, height) <= 0:
        raise ValueError(
            "render area must have positive dimensions and nonnegative origin"
        )
    scale = min(Fraction(width, crop.width), Fraction(height, crop.height))
    output_width = max(1, round(crop.width * scale))
    output_height = max(1, round(crop.height * scale))
    return ImageRect(
        left=left + (width - output_width) // 2,
        top=top + (height - output_height) // 2,
        width=output_width,
        height=output_height,
    )


def _scale(tile: TileReceipt) -> tuple[float, float]:
    x, y = Fraction(tile.scale_x), Fraction(tile.scale_y)
    if x != Fraction(tile.image_rect.width, tile.crop.width) or y != Fraction(
        tile.image_rect.height, tile.crop.height
    ):
        raise ValueError("tile scale does not match its crop and rendered dimensions")
    return float(x), float(y)


def native_to_image(
    tile: TileReceipt, x: float, y: float, *, coordinates: Coordinates
) -> tuple[float, float]:
    if not isfinite(x) or not isfinite(y):
        raise ValueError("image coordinates must be finite")
    sx, sy = _scale(tile)
    if coordinates not in {"pixel_center", "edge"}:
        raise ValueError("explicit pixel-center or bbox-edge coordinates are required")
    offset = 0.5 if coordinates == "pixel_center" else 0.0
    return (
        tile.image_rect.left + (x - tile.crop.left + offset) * sx - offset,
        tile.image_rect.top + (y - tile.crop.top + offset) * sy - offset,
    )


def image_to_native(
    tile: TileReceipt, x: float, y: float, *, coordinates: Coordinates
) -> tuple[float, float]:
    if not isfinite(x) or not isfinite(y):
        raise ValueError("image coordinates must be finite")
    sx, sy = _scale(tile)
    if coordinates not in {"pixel_center", "edge"}:
        raise ValueError("explicit pixel-center or bbox-edge coordinates are required")
    offset = 0.5 if coordinates == "pixel_center" else 0.0
    rect = tile.image_rect
    if not (
        rect.left - offset <= x <= rect.left + rect.width - offset
        and rect.top - offset <= y <= rect.top + rect.height - offset
    ):
        raise ValueError("image position lies in the header, padding, or another tile")
    return (
        (x - rect.left + offset) / sx + tile.crop.left - offset,
        (y - rect.top + offset) / sy + tile.crop.top - offset,
    )
