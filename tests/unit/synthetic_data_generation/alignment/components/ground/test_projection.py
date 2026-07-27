"""Tests for image-ray projection onto the fitted ground plane."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
    GroundLineMapSettings,
)
from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneEstimate,
)
from src.synthetic_data_generation.alignment.components.ground.projection import (
    project_line_pixels_to_ground,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


def _plane() -> GroundPlaneEstimate:
    return GroundPlaneEstimate(
        normal=(0.0, 0.0, 1.0),
        offset=0.0,
        origin=(0.0, 0.0, 0.0),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
        metrics={},
    )


def _camera() -> SceneCamera:
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[:3, 3] = (0.0, 0.0, 1.0)
    return SceneCamera(
        camera_id="camera-0",
        source_camera_id="synthetic",
        image_uri="images/camera-0.png",
        source_frame_index=0,
        group_id=0,
        width=201,
        height=201,
        intrinsics=(100.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


def test_projection_intersects_ground_and_downweights_distant_pixels() -> None:
    projection = project_line_pixels_to_ground(
        _camera(),
        np.asarray([[100.0, 100.0], [200.0, 100.0]]),
        np.asarray([0.9, 0.8]),
        plane=_plane(),
        bounds=(-2.0, 2.0, -2.0, 2.0),
        settings=GroundLineMapSettings(
            proximity_scale=1.0,
            max_ray_distance=5.0,
        ),
    )

    np.testing.assert_allclose(
        projection.points_scene,
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atol=1.0e-8,
    )
    assert projection.camera_ranges[0] == pytest.approx(1.0)
    assert projection.camera_ranges[1] == pytest.approx(np.sqrt(2.0))
    assert projection.proximity_weights[0] > projection.proximity_weights[1]


def test_projection_rejects_rays_behind_camera() -> None:
    camera = _camera()
    pose = np.asarray(camera.camera_to_scene).reshape(4, 4).copy()
    pose[:3, :3] = np.eye(3)
    upward_camera = replace(
        camera,
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )

    projection = project_line_pixels_to_ground(
        upward_camera,
        np.asarray([[100.0, 100.0]]),
        np.asarray([0.9]),
        plane=_plane(),
        bounds=(-2.0, 2.0, -2.0, 2.0),
        settings=GroundLineMapSettings(),
    )

    assert len(projection.points_uv) == 0
    assert projection.invalid_behind_count == 1


def test_projection_rejects_parallel_rays_without_nonfinite_intersections() -> None:
    camera = _camera()
    pose = np.asarray(camera.camera_to_scene).reshape(4, 4).copy()
    pose[:3, :3] = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    horizontal_camera = replace(
        camera,
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )

    projection = project_line_pixels_to_ground(
        horizontal_camera,
        np.asarray([[100.0, 100.0]]),
        np.asarray([0.9]),
        plane=_plane(),
        bounds=(-2.0, 2.0, -2.0, 2.0),
        settings=GroundLineMapSettings(),
    )

    assert len(projection.points_uv) == 0
    assert projection.invalid_parallel_count == 1
