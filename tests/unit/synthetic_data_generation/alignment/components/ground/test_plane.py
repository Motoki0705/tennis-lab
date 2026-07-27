"""Synthetic recovery and rejection tests for scene ground-plane fitting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneFitSettings,
    estimate_ground_plane,
)
from src.synthetic_data_generation.scene_contract import SceneCamera


def _rotation_with_camera_up(normal: NDArray[np.float64]) -> NDArray[np.float64]:
    camera_x = np.asarray([1.0, 0.0, 0.0])
    camera_x -= normal * float(camera_x @ normal)
    camera_x /= np.linalg.norm(camera_x)
    camera_y_down = -normal
    camera_z = np.cross(camera_x, camera_y_down)
    return np.stack((camera_x, camera_y_down, camera_z), axis=1)


def _camera(
    camera_id: str,
    center: NDArray[np.float64],
    normal: NDArray[np.float64],
) -> SceneCamera:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _rotation_with_camera_up(normal)
    pose[:3, 3] = center
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="synthetic",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=int(camera_id.split("-")[-1]),
        group_id=0,
        width=320,
        height=180,
        intrinsics=(250.0, 0.0, 160.0, 0.0, 250.0, 90.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


def _synthetic_scene() -> tuple[
    NDArray[np.float64],
    tuple[SceneCamera, ...],
    NDArray[np.float64],
    float,
]:
    rng = np.random.default_rng(42)
    normal = np.asarray([0.04, -0.03, 1.0])
    normal /= np.linalg.norm(normal)
    offset = 0.2
    u = np.asarray([1.0, 0.0, 0.0])
    u -= normal * float(u @ normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    uv = rng.uniform((-1.2, -1.0), (1.2, 1.0), size=(6000, 2))
    origin = -offset * normal
    ground = origin + uv[:, :1] * u + uv[:, 1:] * v
    ground += rng.normal(0.0, 0.001, size=(len(ground), 1)) * normal
    outliers = rng.uniform((-1.5, -1.3, -0.6), (1.5, 1.3, 0.6), size=(2500, 3))
    points = np.concatenate((ground, outliers), axis=0)
    camera_uv = np.asarray(
        [(-0.8, -0.6), (-0.4, 0.1), (0.0, 0.7), (0.5, -0.2), (0.8, 0.6)]
    )
    cameras = tuple(
        _camera(
            f"camera-{index}",
            origin + xy[0] * u + xy[1] * v + 0.16 * normal,
            normal,
        )
        for index, xy in enumerate(camera_uv)
    )
    return points, cameras, normal, offset


def _settings() -> GroundPlaneFitSettings:
    return GroundPlaneFitSettings(
        footprint_margin=0.6,
        histogram_bin_width=0.004,
        candidate_half_width=0.02,
        ransac_threshold=0.004,
        refine_threshold=0.006,
        ransac_iterations=500,
        ransac_sample_limit=5000,
        min_candidate_points=500,
        min_support_points=3000,
    )


def test_ground_plane_recovers_tilted_plane_amid_outliers() -> None:
    points, cameras, expected_normal, expected_offset = _synthetic_scene()

    plane = estimate_ground_plane(points, cameras, settings=_settings())

    assert np.dot(plane.normal, expected_normal) > 0.9999
    assert plane.offset == pytest.approx(expected_offset, abs=0.002)
    assert plane.metrics["support_point_count"] >= 5000
    assert plane.metrics["camera_positive_height_fraction"] == 1.0
    assert plane.metrics["camera_height_median"] == pytest.approx(0.16, abs=0.002)
    reconstructed = plane.from_uv(plane.to_uv(points[:20]))
    assert np.max(np.abs(plane.signed_distance(reconstructed))) < 1.0e-8


def test_ground_plane_rejects_insufficient_support() -> None:
    points, cameras, _normal, _offset = _synthetic_scene()
    settings = replace(_settings(), min_support_points=8000)

    with pytest.raises(ValueError, match="insufficient support"):
        estimate_ground_plane(points, cameras, settings=settings)


def test_ground_plane_rejects_inconsistent_camera_up_vectors() -> None:
    points, cameras, normal, _offset = _synthetic_scene()
    pose = np.asarray(cameras[-1].camera_to_scene).reshape(4, 4).copy()
    pose[:3, :3] = _rotation_with_camera_up(-normal)
    bad_camera = replace(
        cameras[-1],
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )

    with pytest.raises(ValueError, match="up vectors"):
        estimate_ground_plane(
            points,
            (*cameras[:-1], bad_camera),
            settings=_settings(),
        )
