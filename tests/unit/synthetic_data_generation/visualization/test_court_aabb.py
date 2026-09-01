"""Tests for the exact depth-aware Court V4 obstacle-voxel renderer."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.synthetic_data_generation.visualization.court_aabb import (
    CourtAABBRenderConfig,
    CourtAABBRenderResult,
    extract_exposed_voxel_faces,
    render_court_obstacle_aabbs,
)


def _camera(*, width: int = 17, height: int = 17) -> SceneCamera:
    focal_length = 8.0
    return SceneCamera(
        camera_id="camera",
        source_frame_index=0,
        width=width,
        height=height,
        intrinsics=(
            focal_length,
            0.0,
            float(width // 2),
            0.0,
            focal_length,
            float(height // 2),
            0.0,
            0.0,
            1.0,
        ),
        camera_to_scene=RigidTransform.identity(),
        image_path="camera.png",
    )


def _config(**overrides: object) -> CourtAABBRenderConfig:
    values: dict[str, object] = {
        "voxel_size_m": 1.0,
        "near_plane_m": 0.1,
        "depth_epsilon_m": 0.0,
        "surface_color_rgb": (1.0, 0.0, 0.0),
        "surface_opacity": 1.0,
        "background_color_rgb": (0.0, 0.0, 1.0),
        "maximum_cells": 100,
        "maximum_surface_faces": 600,
        "maximum_projected_pixels": 100_000,
    }
    values.update(overrides)
    return CourtAABBRenderConfig(**values)  # type: ignore[arg-type]


def _frame_arrays(
    camera: SceneCamera,
    *,
    rgb_value: tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha_value: float = 1.0,
    depth_value: float = 10.0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    rgb = np.empty((camera.height, camera.width, 3), dtype=np.float32)
    rgb[...] = rgb_value
    alpha = np.full(
        (camera.height, camera.width, 1), alpha_value, dtype=np.float32
    )
    depth = np.full(
        (camera.height, camera.width, 1), depth_value, dtype=np.float32
    )
    return rgb, alpha, depth


def _render(
    cells: NDArray[np.int64],
    *,
    config: CourtAABBRenderConfig | None = None,
    rgb_value: tuple[float, float, float] = (0.0, 0.0, 0.0),
    alpha_value: float = 1.0,
    depth_value: float = 10.0,
) -> CourtAABBRenderResult:
    camera = _camera()
    rgb, alpha, depth = _frame_arrays(
        camera,
        rgb_value=rgb_value,
        alpha_value=alpha_value,
        depth_value=depth_value,
    )
    return render_court_obstacle_aabbs(
        rgb=rgb,
        alpha=alpha,
        metric_depth=depth,
        camera=camera,
        occupancy_cells=cells,
        config=config or _config(),
    )


def test_exposed_faces_are_deterministic_and_remove_internal_neighbours() -> None:
    cells = np.asarray(((0, 0, 2), (1, 0, 2)), dtype=np.int64)

    first = extract_exposed_voxel_faces(
        cells,
        voxel_size_m=1.0,
        maximum_cells=2,
        maximum_surface_faces=10,
    )
    second = extract_exposed_voxel_faces(
        cells,
        voxel_size_m=1.0,
        maximum_cells=2,
        maximum_surface_faces=10,
    )

    assert first.shape == (10, 4, 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first[0],
        np.asarray(
            ((0.0, 0.0, 2.0), (0.0, 0.0, 3.0), (0.0, 1.0, 3.0), (0.0, 1.0, 2.0))
        ),
    )
    internal_x_plane = np.all(first[..., 0] == 1.0, axis=1)
    assert not np.any(internal_x_plane)
    assert not first.flags.writeable


@pytest.mark.parametrize(
    "cells,match",
    (
        (np.asarray(((1, 0, 0), (0, 0, 0)), dtype=np.int64), "lexicographically"),
        (np.asarray(((0, 0, 0), (0, 0, 0)), dtype=np.int64), "lexicographically"),
    ),
)
def test_occupancy_cells_require_strict_lexicographic_unique_order(
    cells: NDArray[np.int64],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        extract_exposed_voxel_faces(
            cells,
            voxel_size_m=1.0,
            maximum_cells=10,
            maximum_surface_faces=60,
        )


def test_cells_behind_camera_are_rejected_and_near_crossing_faces_are_clipped() -> None:
    behind = _render(np.asarray(((0, 0, -2),), dtype=np.int64))
    crossing = _render(
        np.asarray(((0, 0, 0),), dtype=np.int64),
        config=_config(near_plane_m=0.5),
    )

    assert behind.stats.near_rejected_face_count == 6
    assert behind.stats.triangle_count == 0
    assert behind.stats.surface_pixel_count == 0
    assert crossing.stats.near_rejected_face_count >= 1
    assert crossing.stats.near_clipped_face_count >= 1
    assert crossing.stats.triangle_count > 0
    assert crossing.stats.drawn_pixel_count > 0


def test_local_z_buffer_keeps_nearest_surface_instead_of_last_triangle() -> None:
    near_and_far = np.asarray(((0, 0, 2), (0, 0, 4)), dtype=np.int64)
    far_only = np.asarray(((0, 0, 4),), dtype=np.int64)

    combined = _render(near_and_far, depth_value=2.5)
    distant = _render(far_only, depth_value=2.5)

    np.testing.assert_array_equal(combined.rgb[9, 9], (255, 0, 0))
    np.testing.assert_array_equal(distant.rgb[9, 9], (0, 0, 0))
    assert combined.stats.drawn_pixel_count > 0
    assert distant.stats.occluded_pixel_count > 0


def test_slanted_face_depth_is_perspective_correct_camera_z() -> None:
    angle = math.radians(15.0)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    camera_to_scene = np.eye(4, dtype=np.float64)
    camera_to_scene[:3, :3] = (
        (cosine, 0.0, sine),
        (0.0, 1.0, 0.0),
        (-sine, 0.0, cosine),
    )
    camera_to_scene[1, 3] = 0.5
    camera = SceneCamera(
        camera_id="slanted",
        source_frame_index=0,
        width=33,
        height=33,
        intrinsics=(20.0, 0.0, 16.0, 0.0, 20.0, 16.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(camera_to_scene),
        image_path="slanted.png",
    )
    cells = np.asarray(((1, 0, 4),), dtype=np.int64)
    expected_camera_z = 4.0 / cosine
    rgb, alpha, depth = _frame_arrays(camera, depth_value=expected_camera_z - 0.01)
    occluded = render_court_obstacle_aabbs(
        rgb=rgb,
        alpha=alpha,
        metric_depth=depth,
        camera=camera,
        occupancy_cells=cells,
        config=_config(),
    )
    depth.fill(expected_camera_z + 0.01)
    visible = render_court_obstacle_aabbs(
        rgb=rgb,
        alpha=alpha,
        metric_depth=depth,
        camera=camera,
        occupancy_cells=cells,
        config=_config(),
    )

    np.testing.assert_array_equal(occluded.rgb[16, 16], (0, 0, 0))
    np.testing.assert_array_equal(visible.rgb[16, 16], (255, 0, 0))


def test_metric_depth_occlusion_uses_configured_epsilon() -> None:
    cells = np.asarray(((0, 0, 2),), dtype=np.int64)

    occluded = _render(cells, depth_value=1.9)
    epsilon_visible = _render(
        cells,
        depth_value=1.9,
        config=_config(depth_epsilon_m=0.11),
    )

    np.testing.assert_array_equal(occluded.rgb[9, 9], (0, 0, 0))
    np.testing.assert_array_equal(epsilon_visible.rgb[9, 9], (255, 0, 0))
    assert occluded.stats.occluded_pixel_count > 0
    assert epsilon_visible.stats.drawn_pixel_count > 0


def test_premultiplied_rgb_is_resolved_once_and_invalid_depth_does_not_occlude() -> None:
    camera = _camera()
    rgb, alpha, depth = _frame_arrays(
        camera,
        rgb_value=(0.5, 0.0, 0.0),
        alpha_value=0.5,
        depth_value=0.0,
    )
    empty = render_court_obstacle_aabbs(
        rgb=rgb,
        alpha=alpha,
        metric_depth=depth,
        camera=camera,
        occupancy_cells=np.empty((0, 3), dtype=np.int64),
        config=_config(),
    )
    surface = render_court_obstacle_aabbs(
        rgb=np.zeros_like(rgb),
        alpha=np.zeros_like(alpha),
        metric_depth=np.full_like(depth, 0.01),
        camera=camera,
        occupancy_cells=np.asarray(((0, 0, 2),), dtype=np.int64),
        config=_config(surface_opacity=0.5),
    )

    np.testing.assert_array_equal(empty.rgb[0, 0], (128, 0, 128))
    np.testing.assert_array_equal(surface.rgb[9, 9], (128, 0, 128))
    assert surface.stats.background_valid_pixel_count == 0
    assert surface.stats.drawn_pixel_count > 0


@pytest.mark.parametrize(
    ("config", "cells", "match"),
    (
        (_config(maximum_cells=1), np.asarray(((0, 0, 2), (1, 0, 2)), dtype=np.int64), "maximum_cells"),
        (_config(maximum_surface_faces=5), np.asarray(((0, 0, 2),), dtype=np.int64), "maximum_surface_faces"),
        (_config(maximum_projected_pixels=1), np.asarray(((0, 0, 2),), dtype=np.int64), "maximum_projected_pixels"),
    ),
)
def test_resource_caps_fail_closed_without_truncation(
    config: CourtAABBRenderConfig,
    cells: NDArray[np.int64],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _render(cells, config=config)


def test_inputs_require_exact_public_nht_shapes_dtypes_and_ranges() -> None:
    camera = _camera()
    rgb, alpha, depth = _frame_arrays(camera)
    cells = np.asarray(((0, 0, 2),), dtype=np.int64)

    with pytest.raises(TypeError, match="exact dtype int64"):
        render_court_obstacle_aabbs(
            rgb=rgb,
            alpha=alpha,
            metric_depth=depth,
            camera=camera,
            occupancy_cells=cells.astype(np.int32),
            config=_config(),
        )
    with pytest.raises(ValueError, match="camera-exact shape"):
        render_court_obstacle_aabbs(
            rgb=rgb,
            alpha=alpha[..., 0],
            metric_depth=depth,
            camera=camera,
            occupancy_cells=cells,
            config=_config(),
        )
    bad_depth = depth.copy()
    bad_depth[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite nonnegative"):
        render_court_obstacle_aabbs(
            rgb=rgb,
            alpha=alpha,
            metric_depth=bad_depth,
            camera=camera,
            occupancy_cells=cells,
            config=_config(),
        )


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"near_plane_m": 0.0}, "near_plane_m"),
        ({"depth_epsilon_m": -0.1}, "depth_epsilon_m"),
        ({"surface_opacity": 1.1}, "surface_opacity"),
        ({"surface_color_rgb": (1.0, 0.0, np.nan)}, "surface_color_rgb"),
        ({"maximum_projected_pixels": 0}, "maximum_projected_pixels"),
    ),
)
def test_configuration_is_strictly_validated(
    change: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _config(**change)


def test_result_rgb_and_stats_are_immutable() -> None:
    result = _render(np.asarray(((0, 0, 2),), dtype=np.int64))

    assert result.rgb.dtype == np.uint8
    assert not result.rgb.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.rgb[0, 0] = 0
    with pytest.raises(FrozenInstanceError):
        result.stats.cell_count = 2  # type: ignore[misc]
