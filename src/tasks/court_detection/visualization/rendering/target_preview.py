"""Render and summarize the exact dense targets consumed by Court losses."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypeAlias, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.court_detection.visualization.rendering.common import (
    COURT_SEG_PALETTE_RGB,
    colorize_seg_mask,
    coverage_colors,
)

UInt8Array: TypeAlias = NDArray[np.uint8]
MaskArray: TypeAlias = NDArray[np.integer[Any]] | NDArray[np.floating[Any]]


def gaussian_pixel_geometry(
    sigma_ratio: float,
    *,
    height: int,
    width: int,
) -> dict[str, float]:
    """Describe the Gaussian target size in image pixels."""
    if not math.isfinite(sigma_ratio) or sigma_ratio <= 0.0:
        raise ValueError("sigma_ratio must be finite and positive.")
    if height <= 0 or width <= 0:
        raise ValueError("Heatmap dimensions must be positive.")
    sigma_px = sigma_ratio * math.hypot(height, width)
    return {
        "sigma_ratio": sigma_ratio,
        "sigma_px": sigma_px,
        "fwhm_diameter_px": 2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_px,
    }


def render_heatmap_target(
    rgb: UInt8Array,
    heatmaps: torch.Tensor,
    *,
    alpha: float,
) -> UInt8Array:
    """Overlay the channel-wise maximum of an exact KP target tensor."""
    if heatmaps.ndim != 3 or tuple(heatmaps.shape[-2:]) != rgb.shape[:2]:
        raise ValueError("KP preview requires [C,H,W] heatmaps matching RGB.")
    pooled = heatmaps.detach().cpu().float().amax(dim=0).clamp(0.0, 1.0).numpy()
    colored = cv2.applyColorMap(
        np.rint(pooled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored = cast("UInt8Array", cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))
    return _blend(rgb, colored, alpha=alpha)


def render_segmentation_target(
    rgb: UInt8Array,
    mask: torch.Tensor | MaskArray,
    *,
    alpha: float,
    max_label: int = 6,
) -> UInt8Array:
    """Overlay a categorical target while preserving background."""
    raw = (
        mask.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor)
        else np.asarray(mask)
    )
    if raw.ndim == 3 and raw.shape[0] > 1:
        if not 0 <= alpha <= 1 or raw.shape[-2:] != rgb.shape[:2]:
            raise ValueError(
                "Coverage preview requires matching RGB and alpha in [0,1]."
            )
        colors, foreground = coverage_colors(
            raw, COURT_SEG_PALETTE_RGB[: max_label + 1]
        )
        return (
            np.rint(rgb * (1.0 - alpha * foreground[..., None]) + alpha * colors)
            .clip(0, 255)
            .astype(np.uint8)
        )
    array = _mask_array(mask)
    if (
        max_label < 0
        or array.shape != rgb.shape[:2]
        or int(array.min(initial=0)) < 0
        or int(array.max(initial=0)) > max_label
    ):
        raise ValueError(
            f"Categorical preview requires a matching label map in [0,{max_label}]."
        )
    colored = colorize_seg_mask(array)
    return _blend_where(rgb, colored, array > 0, alpha=alpha)


def render_line_target(
    rgb: UInt8Array,
    mask: torch.Tensor | MaskArray,
    *,
    alpha: float,
) -> UInt8Array:
    """Overlay the exact binary line target in red."""
    array = _mask_array(mask)
    if array.shape != rgb.shape[:2]:
        raise ValueError("LINE preview requires a mask matching RGB.")
    if (
        not 0 <= alpha <= 1
        or not np.isfinite(array).all()
        or np.any((array < 0) | (array > 1))
    ):
        raise ValueError("LINE coverage and alpha must be finite in [0,1].")
    opacity = array.astype(np.float32)[..., None] * alpha
    return (
        np.rint(rgb * (1.0 - opacity) + np.array([255, 72, 72]) * opacity)
        .clip(0, 255)
        .astype(np.uint8)
    )


def render_target_only(kind: str, target: object) -> UInt8Array:
    """Show exact heatmap/coverage values without a background photograph."""
    value = (
        cast(Mapping[str, torch.Tensor], target)["heatmap"]
        if kind == "kp"
        else cast(torch.Tensor, target)
    )
    height, width = value.shape[-2:]
    black = np.zeros((height, width, 3), dtype=np.uint8)
    if kind == "kp":
        return render_heatmap_target(black, value, alpha=1.0)
    if kind in {"seg", "semantic_line"}:
        return render_segmentation_target(
            black, value, alpha=1.0, max_label=6 if kind == "seg" else 11
        )
    if kind == "line":
        gray = (
            np.rint(value[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        return np.repeat(gray[..., None], 3, axis=2)
    raise ValueError(f"Unknown target: {kind!r}.")


def summarize_targets(
    targets: Mapping[str, object],
    *,
    sigma_ratio: float | None,
) -> dict[str, object]:
    """Return JSON-safe shape and foreground statistics for selected targets."""
    result: dict[str, object] = {}
    kp = targets.get("kp")
    if kp is not None:
        payload = cast("Mapping[str, torch.Tensor]", kp)
        heatmaps = payload["heatmap"].detach().cpu().float()
        if heatmaps.ndim != 3:
            raise ValueError("KP target heatmap must have shape [C,H,W].")
        if sigma_ratio is None:
            raise ValueError("KP target statistics require sigma_ratio.")
        height, width = (int(value) for value in heatmaps.shape[-2:])
        result["kp"] = {
            "shape": list(heatmaps.shape),
            "visible_points": int(payload["point_visible"].sum().item()),
            "pixels_ge_0_5": int((heatmaps >= 0.5).sum().item()),
            **gaussian_pixel_geometry(
                sigma_ratio,
                height=height,
                width=width,
            ),
        }
    seg = targets.get("seg")
    if seg is not None:
        result["seg"] = _categorical_summary(cast(torch.Tensor, seg), channels=7)
    semantic_line = targets.get("semantic_line")
    if semantic_line is not None:
        result["semantic_line"] = _categorical_summary(
            cast(torch.Tensor, semantic_line), channels=12
        )
    line = targets.get("line")
    if line is not None:
        tensor = cast("torch.Tensor", line)
        array = _mask_array(tensor)
        foreground = int(np.count_nonzero(array))
        result["line"] = {
            "shape": list(tensor.shape),
            "foreground_pixels": foreground,
            "foreground_fraction": float(array.mean()),
            "coverage_sum": float(array.sum()),
            "fractional_pixels": int(((array > 0) & (array < 1)).sum()),
        }
    return result


def _categorical_summary(value: torch.Tensor, *, channels: int) -> dict[str, object]:
    array = value.detach().cpu().numpy()
    if array.ndim == 3:
        coverage_colors(array, COURT_SEG_PALETTE_RGB[:channels])
        return {
            "shape": list(array.shape),
            "class_pixel_mass": {
                str(index): float(mass)
                for index, mass in enumerate(array.sum(axis=(1, 2)))
            },
            "fractional_pixels": int(((array > 0) & (array < 1)).any(axis=0).sum()),
        }
    labels, counts = np.unique(_mask_array(value), return_counts=True)
    return {
        "shape": list(array.shape),
        "class_pixel_counts": {
            str(int(label)): int(count)
            for label, count in zip(labels, counts, strict=True)
        },
    }


def _mask_array(
    value: torch.Tensor | MaskArray,
) -> MaskArray:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Target mask must have shape [H,W] or [1,H,W].")
    return cast("MaskArray", array)


def _blend(rgb: UInt8Array, colored: UInt8Array, *, alpha: float) -> UInt8Array:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Target preview alpha must be in [0,1].")
    return cast(
        UInt8Array,
        np.clip(
            rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha,
            0.0,
            255.0,
        ).astype(np.uint8),
    )


def _blend_where(
    rgb: UInt8Array,
    colored: UInt8Array,
    region: NDArray[np.bool_],
    *,
    alpha: float,
) -> UInt8Array:
    blended = _blend(rgb, colored, alpha=alpha)
    output = rgb.copy()
    output[region] = blended[region]
    return output


__all__ = [
    "gaussian_pixel_geometry",
    "render_heatmap_target",
    "render_line_target",
    "render_target_only",
    "render_segmentation_target",
    "summarize_targets",
]
