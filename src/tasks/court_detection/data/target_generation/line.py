"""Source-neutral precomputation of binary Court line targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.utils.schema.court import (
    CENTER_MARK_LENGTH,
    COURT_SKELETON,
    HALF_LENGTH,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

Float32Array: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]

_LINE_WIDTH_METRES = 0.05
_BASELINE_WIDTH_METRES = 0.10


@dataclass(frozen=True, slots=True)
class _MetricLine:
    start: tuple[float, float]
    end: tuple[float, float]
    width_m: float


def _metric_lines() -> tuple[_MetricLine, ...]:
    points: Float32Array = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy()[:, :2]
    baseline_pairs = {(0, 1), (2, 3)}
    result: list[_MetricLine] = []
    for first, second in COURT_SKELETON:
        if first >= 14 or second >= 14:
            continue
        result.append(
            _MetricLine(
                (float(points[first, 0]), float(points[first, 1])),
                (float(points[second, 0]), float(points[second, 1])),
                (
                    _BASELINE_WIDTH_METRES
                    if (first, second) in baseline_pairs
                    else _LINE_WIDTH_METRES
                ),
            )
        )
    result.extend(
        (
            _MetricLine(
                (0.0, HALF_LENGTH),
                (0.0, HALF_LENGTH - CENTER_MARK_LENGTH),
                _LINE_WIDTH_METRES,
            ),
            _MetricLine(
                (0.0, -HALF_LENGTH),
                (0.0, -HALF_LENGTH + CENTER_MARK_LENGTH),
                _LINE_WIDTH_METRES,
            ),
        )
    )
    return tuple(result)


_METRIC_LINES = _metric_lines()


def _segment_quad(line: _MetricLine) -> Float32Array:
    start: Float32Array = np.asarray(line.start, dtype=np.float32)
    end: Float32Array = np.asarray(line.end, dtype=np.float32)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1.0e-6:
        raise ValueError("Degenerate metric Court line.")
    tangent = direction / length
    normal: Float32Array = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    offset = normal * (line.width_m * 0.5)
    return cast(
        Float32Array,
        np.stack(
            [start - offset, end - offset, end + offset, start + offset],
            axis=0,
        ).astype(np.float32),
    )


def generate_line_target(
    *,
    height: int,
    width: int,
    instances: tuple[CourtInstance2D, ...],
) -> UInt8Array:
    """Render all court instances into one binary uint8 line mask."""
    if height <= 0 or width <= 0 or not instances:
        raise ValueError("Court line generation requires image geometry.")
    output: UInt8Array = np.zeros((height, width), dtype=np.uint8)
    for instance in instances:
        rasterizer = CourtPlaneRasterizer.from_instance(
            instance,
            width=width,
            height=height,
        )
        if rasterizer is None:
            continue
        for line in _METRIC_LINES:
            polygon = rasterizer.project_polygon(_segment_quad(line))
            if polygon is not None:
                cv2.fillPoly(output, [polygon], 255)
        for point, in_front in zip(
            rasterizer.image_points,
            rasterizer.point_in_front,
            strict=True,
        ):
            if not bool(in_front):
                continue
            if not (
                -1.0 <= float(point[0]) <= float(width)
                and -1.0 <= float(point[1]) <= float(height)
            ):
                continue
            center = (
                int(round(float(point[0]))),
                int(round(float(point[1]))),
            )
            cv2.circle(output, center, radius=1, color=255, thickness=-1)
    return output


__all__ = ["generate_line_target"]
