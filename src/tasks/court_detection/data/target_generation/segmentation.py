"""Source-neutral precomputation of Court cell-segmentation targets."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.geometry import compute_template_to_image_homography
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

_CELL_BOUNDS: dict[int, tuple[float, float, float, float]] = {
    1: (-HALF_SINGLES_WIDTH, 0.0, 0.0, SERVICE_LINE_DISTANCE),
    2: (0.0, HALF_SINGLES_WIDTH, 0.0, SERVICE_LINE_DISTANCE),
    3: (-HALF_SINGLES_WIDTH, 0.0, SERVICE_LINE_DISTANCE, HALF_LENGTH),
    4: (0.0, HALF_SINGLES_WIDTH, SERVICE_LINE_DISTANCE, HALF_LENGTH),
    5: (-HALF_DOUBLES_WIDTH, -HALF_SINGLES_WIDTH, 0.0, HALF_LENGTH),
    6: (HALF_SINGLES_WIDTH, HALF_DOUBLES_WIDTH, 0.0, HALF_LENGTH),
}


def _ordered_physical_points(instance: CourtInstance2D) -> Float32Array:
    if set(instance.physical_indices.tolist()) != set(range(14)):
        raise ValueError("Court target generation requires physical points 0..13.")
    points: Float32Array = np.empty((14, 2), dtype=np.float32)
    for physical, point in zip(
        instance.physical_indices.tolist(),
        instance.points_xy.detach().cpu().numpy(),
        strict=True,
    ):
        points[int(physical)] = point
    return points


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
        homography = compute_template_to_image_homography(
            _ordered_physical_points(instance),
            ransac_reproj_threshold=5.0,
        )
        if homography is None:
            raise ValueError(
                f"Court segmentation homography failed for {instance.court_instance_id!r}."
            )
        instance_mask: UInt8Array = np.zeros_like(output)
        for label, bounds in _CELL_BOUNDS.items():
            for selected in (
                bounds,
                (-bounds[1], -bounds[0], -bounds[3], -bounds[2]),
            ):
                corners = _cell_corners(selected)
                projected = cast(
                    NDArray[np.int32],
                    cv2.perspectiveTransform(
                        corners.reshape(1, -1, 2), homography
                    )
                    .reshape(-1, 2)
                    .astype(np.int32),
                )
                cv2.fillPoly(instance_mask, [projected], int(label))
        output = cast(
            UInt8Array,
            np.where(instance_mask > 0, instance_mask, output).astype(np.uint8),
        )
    return output


__all__ = ["generate_segmentation_target"]
