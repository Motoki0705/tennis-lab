"""Camera-view semantic Court-line target generation."""

from __future__ import annotations

import math
from typing import TypeAlias, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.data.target_generation.line import (
    CourtMetricLine,
    metric_line_quad,
)
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.tasks.court_detection.target_schemas import (
    SEMANTIC_LINE_CLASS_BY_NAME,
    SEMANTIC_LINE_TARGET_DEFINITION,
)
from src.utils.schema.court import (
    CENTER_MARK_LENGTH,
    COURT_SKELETON,
    HALF_LENGTH,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

Float32Array: TypeAlias = NDArray[np.float32]
Float64Array: TypeAlias = NDArray[np.float64]
UInt8Array: TypeAlias = NDArray[np.uint8]

_SEMANTIC_CLASS_BY_SEGMENT = {
    (0, 1): "far_baseline",
    (2, 3): "near_baseline",
    (0, 2): "left_doubles_sideline",
    (1, 3): "right_doubles_sideline",
    (4, 8): "left_singles_sideline",
    (8, 10): "left_singles_sideline",
    (10, 5): "left_singles_sideline",
    (6, 9): "right_singles_sideline",
    (9, 11): "right_singles_sideline",
    (11, 7): "right_singles_sideline",
    (8, 9): "far_service_line",
    (10, 11): "near_service_line",
    (12, 13): "center_service_line",
}
_SEMANTIC_CLASS_AT_KEYPOINT = (
    "far_baseline",
    "far_baseline",
    "near_baseline",
    "near_baseline",
    "far_baseline",
    "near_baseline",
    "far_baseline",
    "near_baseline",
    "far_service_line",
    "far_service_line",
    "near_service_line",
    "near_service_line",
    "center_service_line",
    "center_service_line",
)


def _validate_widths(
    *,
    line_width_metres: float,
    baseline_width_metres: float,
) -> None:
    if (
        not math.isfinite(line_width_metres)
        or not math.isfinite(baseline_width_metres)
        or line_width_metres <= 0.0
        or baseline_width_metres <= 0.0
    ):
        raise ValueError("Court semantic-line widths must be finite and positive.")


def _semantic_to_physical_transform(
    semantic_to_physical: Tensor,
    points: Float32Array,
) -> Float64Array:
    if semantic_to_physical.shape != (14,) or semantic_to_physical.dtype != torch.long:
        raise ValueError("Semantic Court-line order must be an int64 vector of length 14.")
    mapping = semantic_to_physical.detach().cpu().numpy()
    if set(mapping.tolist()) != set(range(14)):
        raise ValueError("Semantic Court-line order must be a permutation of 0..13.")
    source = np.concatenate(
        (points.astype(np.float64), np.ones((14, 1), dtype=np.float64)),
        axis=1,
    )
    destination = points[mapping].astype(np.float64)
    transform, _, rank, _ = np.linalg.lstsq(source, destination, rcond=None)
    if rank != 3 or not np.allclose(
        source @ transform,
        destination,
        atol=1.0e-5,
        rtol=0.0,
    ):
        raise ValueError(
            "Semantic Court-line order must describe one rigid KP14 court symmetry."
        )
    linear = transform[:2]
    if not np.allclose(linear @ linear.T, np.eye(2), atol=1.0e-6, rtol=0.0):
        raise ValueError(
            "Semantic Court-line order must preserve physical Court-line widths."
        )
    return cast(Float64Array, transform)


def _transform_point(
    point: tuple[float, float],
    transform: Float64Array,
) -> tuple[float, float]:
    transformed = np.asarray([point[0], point[1], 1.0], dtype=np.float64) @ transform
    return float(transformed[0]), float(transformed[1])


def _semantic_metric_lines(
    *,
    semantic_to_physical: Tensor,
    line_width_metres: float,
    baseline_width_metres: float,
) -> tuple[tuple[int, CourtMetricLine], ...]:
    points: Float32Array = (
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2].numpy()
    )
    transform = _semantic_to_physical_transform(semantic_to_physical, points)
    mapping = semantic_to_physical.detach().cpu().tolist()
    baseline_pairs = {(0, 1), (2, 3)}
    result: list[tuple[int, CourtMetricLine]] = []
    for first, second in COURT_SKELETON:
        if first >= 14 or second >= 14:
            continue
        try:
            class_name = _SEMANTIC_CLASS_BY_SEGMENT[(first, second)]
        except KeyError as error:  # pragma: no cover - shared KP14 skeleton contract
            raise ValueError(
                f"KP14 Court segment {(first, second)!r} has no semantic class."
            ) from error
        physical_first = int(mapping[first])
        physical_second = int(mapping[second])
        result.append(
            (
                SEMANTIC_LINE_CLASS_BY_NAME[class_name],
                CourtMetricLine(
                    start=(
                        float(points[physical_first, 0]),
                        float(points[physical_first, 1]),
                    ),
                    end=(
                        float(points[physical_second, 0]),
                        float(points[physical_second, 1]),
                    ),
                    width_m=(
                        baseline_width_metres
                        if (first, second) in baseline_pairs
                        else line_width_metres
                    ),
                ),
            )
        )
    result.extend(
        (
            (
                SEMANTIC_LINE_CLASS_BY_NAME["far_center_mark"],
                CourtMetricLine(
                    start=_transform_point((0.0, HALF_LENGTH), transform),
                    end=_transform_point(
                        (0.0, HALF_LENGTH - CENTER_MARK_LENGTH), transform
                    ),
                    width_m=line_width_metres,
                ),
            ),
            (
                SEMANTIC_LINE_CLASS_BY_NAME["near_center_mark"],
                CourtMetricLine(
                    start=_transform_point((0.0, -HALF_LENGTH), transform),
                    end=_transform_point(
                        (0.0, -HALF_LENGTH + CENTER_MARK_LENGTH), transform
                    ),
                    width_m=line_width_metres,
                ),
            ),
        )
    )
    return tuple(result)


def generate_semantic_line_target(
    *,
    height: int,
    width: int,
    instances: tuple[CourtInstance2D, ...],
    semantic_to_physical: Tensor,
    line_width_metres: float = SEMANTIC_LINE_TARGET_DEFINITION.line_width_metres,
    baseline_width_metres: float = (
        SEMANTIC_LINE_TARGET_DEFINITION.baseline_width_metres
    ),
) -> UInt8Array:
    """Render one selected court as a camera-view categorical line mask.

    Later primitives have deterministic priority at line intersections. Small
    endpoint disks close projection-rounding cracks and inherit the class of a
    designated incident line.
    """
    if height <= 0 or width <= 0 or len(instances) != 1:
        raise ValueError(
            "Court semantic-line generation requires positive image geometry "
            "and exactly one selected court."
        )
    _validate_widths(
        line_width_metres=line_width_metres,
        baseline_width_metres=baseline_width_metres,
    )
    metric_lines = _semantic_metric_lines(
        semantic_to_physical=semantic_to_physical,
        line_width_metres=line_width_metres,
        baseline_width_metres=baseline_width_metres,
    )
    output: UInt8Array = np.zeros((height, width), dtype=np.uint8)
    rasterizer = CourtPlaneRasterizer.from_instance(
        instances[0],
        width=width,
        height=height,
    )
    if rasterizer is None:
        return output
    for class_index, line in metric_lines:
        polygon = rasterizer.project_polygon(metric_line_quad(line))
        if polygon is not None:
            cv2.fillPoly(output, [polygon], int(class_index))

    for semantic_index, physical_index in enumerate(
        semantic_to_physical.detach().cpu().tolist()
    ):
        point = rasterizer.image_points[int(physical_index)]
        if not bool(rasterizer.point_in_front[int(physical_index)]):
            continue
        if not (
            -1.0 <= float(point[0]) <= float(width)
            and -1.0 <= float(point[1]) <= float(height)
        ):
            continue
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        class_index = SEMANTIC_LINE_CLASS_BY_NAME[
            _SEMANTIC_CLASS_AT_KEYPOINT[semantic_index]
        ]
        cv2.circle(
            output,
            center,
            radius=1,
            color=int(class_index),
            thickness=-1,
        )
    return output


__all__ = ["generate_semantic_line_target"]
