"""Source-neutral precomputation of Court cell-segmentation targets."""

from __future__ import annotations

from typing import TypeAlias, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)

Float32Array: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]

_CELL_BOUNDS: dict[int, tuple[float, float, float, float]] = {
    1: (-HALF_SINGLES_WIDTH, 0.0, 0.0, SERVICE_LINE_DISTANCE),
    2: (0.0, HALF_SINGLES_WIDTH, 0.0, SERVICE_LINE_DISTANCE),
    3: (-HALF_SINGLES_WIDTH, 0.0, SERVICE_LINE_DISTANCE, HALF_LENGTH),
    4: (0.0, HALF_SINGLES_WIDTH, SERVICE_LINE_DISTANCE, HALF_LENGTH),
    5: (-HALF_DOUBLES_WIDTH, -HALF_SINGLES_WIDTH, 0.0, HALF_LENGTH),
    6: (HALF_SINGLES_WIDTH, HALF_DOUBLES_WIDTH, 0.0, HALF_LENGTH),
}


def _cell_corners(bounds: tuple[float, float, float, float]) -> Float32Array:
    x_min, x_max, y_min, y_max = bounds
    return np.asarray(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def generate_segmentation_target(
    *,
    height: int,
    width: int,
    instances: tuple[CourtInstance2D, ...],
) -> UInt8Array:
    """Render all court instances into one uint8 0..6 label map."""
    if height <= 0 or width <= 0 or not instances:
        raise ValueError("Court segmentation generation requires image geometry.")
    output: UInt8Array = np.zeros((height, width), dtype=np.uint8)
    for instance in instances:
        rasterizer = CourtPlaneRasterizer.from_instance(
            instance,
            width=width,
            height=height,
        )
        if rasterizer is None:
            continue
        instance_mask: UInt8Array = np.zeros_like(output)
        for label, bounds in _CELL_BOUNDS.items():
            for selected in (
                bounds,
                (-bounds[1], -bounds[0], -bounds[3], -bounds[2]),
            ):
                corners = _cell_corners(selected)
                projected = rasterizer.project_polygon(corners)
                if projected is None:
                    continue
                cv2.fillPoly(instance_mask, [projected], int(label))
        output = cast(
            UInt8Array,
            np.where(instance_mask > 0, instance_mask, output).astype(np.uint8),
        )
    return output


__all__ = ["generate_segmentation_target"]
