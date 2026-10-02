"""Subpixel Court labels sampled on an 8x grid and reduced to pixel coverage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.court_detection.data.contracts import (
    CourtDenseTargetKind,
    CourtRawSample,
)
from src.tasks.court_detection.data.target_generation.line import metric_line_quad
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.tasks.court_detection.data.target_generation.segmentation import (
    _CELL_BOUNDS,
    _cell_corners,
)
from src.tasks.court_detection.data.target_generation.semantic_line import (
    _semantic_metric_lines,
)
from src.tasks.court_detection.target_schemas import (
    DENSE_COVERAGE_FACTOR,
    SEMANTIC_LINE_TARGET_DEFINITION,
)


def _paint(
    canvas: NDArray[np.uint8],
    projector: CourtPlaneRasterizer,
    corners: NDArray[np.floating],
    label: int,
) -> None:
    polygon = projector.project_polygon_float(corners)
    if polygon is None:
        return
    # Pixel centres must stay aligned when changing raster resolution. Keep
    # fixed-point vertices until OpenCV rasterizes: a thin quad may have only
    # two distinct integer vertices but still has nonzero projected coverage.
    high = (polygon + 0.5) * DENSE_COVERAGE_FACTOR - 0.5
    fixed = np.rint(high * 256.0).astype(np.int32)
    cv2.fillPoly(canvas, [fixed], label, shift=8)


def _coverage(
    labels: NDArray[np.uint8], *, channels: int, height: int, width: int
) -> Tensor:
    factor = DENSE_COVERAGE_FACTOR
    values: NDArray[np.float32] = np.empty((channels, height, width), dtype=np.float32)
    # Count all classes in one pass per bounded row tile. A full-resolution
    # one-hot image (or one repeated reduction per class) is unnecessary.
    columns: NDArray[np.int32] = np.arange(width * factor, dtype=np.int32) // factor
    for first in range(0, height, 32):
        rows = min(32, height - first)
        row_ids: NDArray[np.int32] = np.arange(rows * factor, dtype=np.int32) // factor
        bins = (row_ids[:, None] * width + columns[None, :]) * channels
        bins += labels[first * factor : (first + rows) * factor]
        counts = np.bincount(bins.ravel(), minlength=rows * width * channels)
        values[:, first : first + rows] = counts.reshape(
            rows, width, channels
        ).transpose(2, 0, 1) / float(factor * factor)
    return torch.from_numpy(values)


def generate_coverage_targets(
    raw: CourtRawSample,
    kinds: Iterable[CourtDenseTargetKind],
    *,
    projectors: tuple[CourtPlaneRasterizer | None, ...],
    source_to_output: NDArray[np.float64],
    output_size_hw: tuple[int, int],
    content_size_hw: tuple[int, int] | None,
) -> dict[CourtDenseTargetKind, Tensor]:
    """Return fractional coverage, with background probability one off support.

    SEG and semantic LINE remain categorical distributions; numeric class IDs
    are never interpolated. Binary LINE shares exactly the semantic foreground.
    Physical line widths stay in metres. The old artificial endpoint disks are
    unnecessary because subpixel segments are no longer discarded.
    """
    selected = set(kinds)
    if not selected or selected - {"seg", "line", "semantic_line"}:
        raise ValueError("Coverage generation requires known nonempty dense targets.")
    if len(projectors) != 1 or len(raw.court_instances) != 1:
        raise ValueError("Coverage generation requires exactly one selected court.")
    height, width = output_size_hw
    factor = DENSE_COVERAGE_FACTOR
    high_size = (height * factor, width * factor)
    to_high = np.array(
        [[factor, 0, (factor - 1) / 2], [0, factor, (factor - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    support = cv2.warpPerspective(
        np.ones((raw.image.height, raw.image.width), dtype=np.uint8),
        to_high @ source_to_output,
        (high_size[1], high_size[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    if content_size_hw is not None:
        support[content_size_hw[0] * factor :, :] = 0
        support[:, content_size_hw[1] * factor :] = 0
    projector = projectors[0]
    result: dict[CourtDenseTargetKind, Tensor] = {}
    if "seg" in selected:
        labels: NDArray[np.uint8] = np.zeros(high_size, dtype=np.uint8)
        if projector is not None:
            for label, bounds in _CELL_BOUNDS.items():
                for cell in (bounds, (-bounds[1], -bounds[0], -bounds[3], -bounds[2])):
                    _paint(labels, projector, _cell_corners(cell), label)
        labels[support == 0] = 0
        result["seg"] = _coverage(labels, channels=7, height=height, width=width)
    if selected & {"line", "semantic_line"}:
        channels = raw.keypoint_channels
        if channels is None or channels.physical_indices.shape != (14, 1):
            raise ValueError("Coverage LINE requires ordered singleton KP14 geometry.")
        labels = np.zeros(high_size, dtype=np.uint8)
        if projector is not None:
            definition = SEMANTIC_LINE_TARGET_DEFINITION
            for label, line in _semantic_metric_lines(
                semantic_to_physical=channels.physical_indices[:, 0],
                line_width_metres=definition.line_width_metres,
                baseline_width_metres=definition.baseline_width_metres,
            ):
                _paint(labels, projector, metric_line_quad(line), label)
        labels[support == 0] = 0
        if "semantic_line" in selected:
            semantic = _coverage(labels, channels=12, height=height, width=width)
            result["semantic_line"] = semantic
            if "line" in selected:
                result["line"] = (1.0 - semantic[:1]).contiguous()
        elif "line" in selected:
            foreground = cast(
                NDArray[np.float32],
                (labels > 0)
                .reshape(height, factor, width, factor)
                .mean(axis=(1, 3), dtype=np.float32),
            )
            result["line"] = torch.from_numpy(foreground).unsqueeze(0)
    return result
