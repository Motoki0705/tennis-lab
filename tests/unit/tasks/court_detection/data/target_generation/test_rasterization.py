"""Unit tests for positive-depth Court-plane target rasterization."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.court_detection.data.contracts import CourtInstance2D
from src.tasks.court_detection.data.target_generation.line import (
    generate_line_target,
)
from src.tasks.court_detection.data.target_generation.rasterization import (
    CourtPlaneRasterizer,
)
from src.tasks.court_detection.data.target_generation.segmentation import (
    generate_segmentation_target,
)
from src.tasks.court_detection.geometry import court_template_xy


def _projected_instance(
    homography: NDArray[np.float64],
    *,
    width: int = 64,
    height: int = 48,
) -> CourtInstance2D:
    template = np.asarray(court_template_xy(14), dtype=np.float64)
    template_h = np.concatenate(
        (template, np.ones((template.shape[0], 1), dtype=np.float64)), axis=1
    )
    projected_h = template_h @ homography.T
    projected = projected_h[:, :2] / projected_h[:, 2, None]
    in_front = projected_h[:, 2] > 0.0
    visible = (
        in_front
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] <= float(width - 1))
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] <= float(height - 1))
    )
    return CourtInstance2D(
        court_instance_id="court",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=torch.tensor(projected, dtype=torch.float32),
        point_in_front=torch.tensor(in_front, dtype=torch.bool),
        point_visible=torch.tensor(visible, dtype=torch.bool),
    )


def _camera_plane_crossing_instance() -> CourtInstance2D:
    return _projected_instance(
        np.asarray(
            [
                [8.0, 6.4, 32.0],
                [0.0, 5.0, 20.0],
                [0.0, 0.2, 1.0],
            ],
            dtype=np.float64,
        )
    )


def test_rasterizer_rejects_polygon_behind_camera_and_clips_crossing_polygon() -> None:
    rasterizer = CourtPlaneRasterizer.from_instance(
        _camera_plane_crossing_instance(),
        width=64,
        height=48,
    )
    assert rasterizer is not None

    behind = rasterizer.project_polygon(
        np.asarray([[-1.0, -11.0], [1.0, -11.0], [1.0, -8.0], [-1.0, -8.0]])
    )
    crossing = rasterizer.project_polygon(
        np.asarray([[-4.0, -8.0], [4.0, -8.0], [4.0, 8.0], [-4.0, 8.0]])
    )

    assert behind is None
    assert crossing is not None
    assert np.all(crossing[:, 0] >= 0)
    assert np.all(crossing[:, 0] < 64)
    assert np.all(crossing[:, 1] >= 0)
    assert np.all(crossing[:, 1] < 48)


def test_dense_targets_do_not_fill_frame_when_court_crosses_camera_plane() -> None:
    instance = _camera_plane_crossing_instance()

    segmentation = generate_segmentation_target(
        height=48,
        width=64,
        instances=(instance,),
    )
    line = generate_line_target(height=48, width=64, instances=(instance,))

    assert 0 < np.count_nonzero(segmentation) < segmentation.size
    assert 0 < np.count_nonzero(line) < line.size // 2


def test_all_behind_court_contributes_no_target_pixels() -> None:
    instance = _projected_instance(np.eye(3, dtype=np.float64))
    all_behind = CourtInstance2D(
        court_instance_id=instance.court_instance_id,
        physical_indices=instance.physical_indices,
        points_xy=instance.points_xy,
        point_in_front=torch.zeros(14, dtype=torch.bool),
        point_visible=torch.zeros(14, dtype=torch.bool),
    )

    segmentation = generate_segmentation_target(
        height=48,
        width=64,
        instances=(all_behind,),
    )
    line = generate_line_target(height=48, width=64, instances=(all_behind,))

    assert not np.any(segmentation)
    assert not np.any(line)
