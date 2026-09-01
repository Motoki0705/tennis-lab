"""Tests for the exact depth-aware Court V4 obstacle-voxel renderer."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.synthetic_data_generation.visualization.contracts import (
    CourtAABBRenderStyle,
    CourtAABBTrajectoryFilterRadiusMode,
    CourtAABBTrajectoryFilterScope,
    CourtAABBWireframeTopology,
)
from src.synthetic_data_generation.visualization.court_aabb import (
    CourtAABBRenderConfig,
    CourtAABBRenderResult,
    extract_canonical_exposed_face_edges,
    extract_exposed_voxel_faces,
    filter_court_obstacle_cells_by_trajectory,
    prepare_court_aabb_trajectory_filter,
    render_court_obstacle_aabbs,
    segment_aabb_distance_squared,
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
        "render_style": CourtAABBRenderStyle.WIREFRAME,
        "wireframe_topology": CourtAABBWireframeTopology.BOUNDARY,
        "near_plane_m": 0.1,
        "depth_epsilon_m": 0.0,
        "surface_color_rgb": (1.0, 0.0, 0.0),
        "surface_opacity": 1.0,
        "edge_opacity": 1.0,
        "edge_width_px": 1,
        "background_color_rgb": (0.0, 0.0, 1.0),
        "maximum_cells": 100,
        "maximum_surface_faces": 600,
        "maximum_edge_segments": 1_200,
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
    ("start", "end", "expected"),
    (
        ((-1.0, 0.5, 0.5), (2.0, 0.5, 0.5), 0.0),
        ((-1.0, 1.0, 0.5), (0.0, 1.0, 0.5), 0.0),
        ((-1.0, 2.0, 0.5), (2.0, 2.0, 0.5), 1.0),
        ((-1.0, -1.0, -1.0), (0.0, 0.0, 0.0), 0.0),
        ((2.0, 2.0, 2.0), (2.0, 2.0, 2.0), 3.0),
        ((-2.0, 2.0, 0.5), (-1.0, 2.0, 0.5), 2.0),
    ),
)
def test_segment_aabb_distance_is_exact_for_closed_geometry(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    expected: float,
) -> None:
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    lower = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    upper = np.asarray((1.0, 1.0, 1.0), dtype=np.float64)
    originals = tuple(
        value.copy() for value in (start_array, end_array, lower, upper)
    )

    result = segment_aabb_distance_squared(
        start_array,
        end_array,
        lower=lower,
        upper=upper,
    )

    assert isinstance(result, float)
    assert result == pytest.approx(expected)
    for value, original in zip(
        (start_array, end_array, lower, upper),
        originals,
        strict=True,
    ):
        np.testing.assert_array_equal(value, original)


def test_trajectory_filter_uses_exact_closed_segment_to_closed_cell_aabb() -> None:
    cells = np.asarray(
        ((0, 0, 0), (0, 2, 0), (4, 4, 4)),
        dtype=np.int64,
    )
    original = cells.copy()
    centers = np.asarray(
        ((-2.0, 1.0, 0.5), (3.0, 1.0, 0.5)),
        dtype=np.float64,
    )

    result = filter_court_obstacle_cells_by_trajectory(
        cells,
        trajectory_centers_scene_m=centers,
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
        radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
        resolved_radius_m=1.0,
        maximum_cells=3,
    )

    assert result.cells.tolist() == [[0, 0, 0], [0, 2, 0]]
    assert result.original_cell_count == 3
    assert result.retained_cell_count == 2
    assert result.removed_cell_count == 1
    assert result.trajectory_center_count == 2
    assert result.trajectory_segment_count == 2
    assert result.to_dict() == {
        "scope": "selected_trajectory",
        "radius_mode": "explicit_radius",
        "resolved_radius_m": 1.0,
        "distance_metric": "trajectory_segment_to_closed_cell_aabb",
        "original_cell_count": 3,
        "retained_cell_count": 2,
        "removed_cell_count": 1,
        "trajectory_center_count": 2,
        "trajectory_segment_count": 2,
        "filter_segment_count": 2,
        "closed_trajectory": True,
        "affects_collision_authority": False,
    }
    assert not result.cells.flags.writeable
    np.testing.assert_array_equal(cells, original)


def test_trajectory_filter_closes_polyline_and_supports_one_degenerate_segment() -> None:
    cells = np.asarray(((0, 0, 0), (1, 1, 0)), dtype=np.int64)
    closing_centers = np.asarray(
        ((-1.0, 0.5, 0.5), (-1.0, 3.0, 0.5), (3.0, 3.0, 0.5)),
        dtype=np.float64,
    )
    closing = filter_court_obstacle_cells_by_trajectory(
        cells,
        trajectory_centers_scene_m=closing_centers,
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
        radius_mode=CourtAABBTrajectoryFilterRadiusMode.SUPPORT_RADIUS,
        resolved_radius_m=0.01,
        maximum_cells=2,
    )
    point = filter_court_obstacle_cells_by_trajectory(
        cells,
        trajectory_centers_scene_m=np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64),
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
        radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
        resolved_radius_m=0.01,
        maximum_cells=2,
    )

    assert closing.cells.tolist() == [[1, 1, 0]]
    assert closing.trajectory_segment_count == 3
    assert point.cells.tolist() == [[0, 0, 0]]
    assert point.trajectory_segment_count == 1


def test_all_trajectory_filter_is_explicit_and_empty_results_fail_closed() -> None:
    cells = np.asarray(((0, 0, 0), (4, 4, 4)), dtype=np.int64)
    centers = np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64)

    all_cells = filter_court_obstacle_cells_by_trajectory(
        cells,
        trajectory_centers_scene_m=centers,
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.ALL,
        radius_mode=None,
        resolved_radius_m=None,
        maximum_cells=2,
    )

    assert all_cells.cells.tolist() == cells.tolist()
    assert cells.flags.writeable
    assert all_cells.to_dict()["scope"] == "all"
    assert all_cells.to_dict()["radius_mode"] is None
    assert all_cells.to_dict()["resolved_radius_m"] is None
    with pytest.raises(ValueError, match="retained no cells"):
        filter_court_obstacle_cells_by_trajectory(
            cells,
            trajectory_centers_scene_m=np.asarray(
                ((20.0, 20.0, 20.0),),
                dtype=np.float64,
            ),
            voxel_size_m=1.0,
            scope=CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
            radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
            resolved_radius_m=0.1,
            maximum_cells=2,
        )


@pytest.mark.parametrize(
    ("centers", "scope", "radius_mode", "radius", "match"),
    (
        (
            np.empty((0, 3), dtype=np.float64),
            CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
            CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
            1.0,
            "must not be empty",
        ),
        (
            np.asarray(((0.0, 0.0, math.nan),), dtype=np.float64),
            CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
            CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
            1.0,
            "finite",
        ),
        (
            np.zeros((1, 3), dtype=np.float32),
            CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
            CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
            1.0,
            "float64",
        ),
        (
            np.zeros((1, 3), dtype=np.float64),
            CourtAABBTrajectoryFilterScope.ALL,
            None,
            1.0,
            "requires radius mode and radius None",
        ),
        (
            np.zeros((1, 3), dtype=np.float64),
            CourtAABBTrajectoryFilterScope.SELECTED_TRAJECTORY,
            CourtAABBTrajectoryFilterRadiusMode.SUPPORT_RADIUS,
            0.0,
            "must be positive",
        ),
    ),
)
def test_trajectory_filter_rejects_invalid_geometry_or_policy(
    centers: NDArray[np.floating],
    scope: CourtAABBTrajectoryFilterScope,
    radius_mode: CourtAABBTrajectoryFilterRadiusMode | None,
    radius: float | None,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        filter_court_obstacle_cells_by_trajectory(
            np.asarray(((0, 0, 0),), dtype=np.int64),
            trajectory_centers_scene_m=centers,
            voxel_size_m=1.0,
            scope=scope,
            radius_mode=radius_mode,
            resolved_radius_m=radius,
            maximum_cells=1,
        )


def test_local_swept_filter_uses_incoming_and_outgoing_closed_segments() -> None:
    cells = np.asarray(
        ((-1, 5, 0), (5, 0, 0), (5, 10, 0), (10, 5, 0)),
        dtype=np.int64,
    )
    centers = np.asarray(
        ((0.0, 0.0, 0.5), (10.0, 0.0, 0.5), (10.0, 10.0, 0.5), (0.0, 10.0, 0.5)),
        dtype=np.float64,
    )
    prepared = prepare_court_aabb_trajectory_filter(
        cells,
        trajectory_centers_scene_m=centers,
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS,
        radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
        resolved_radius_m=0.01,
        maximum_cells=4,
    )

    first = prepared.filter(frame_index=0)
    middle = prepared.filter(frame_index=1)
    last = prepared.filter(frame_index=3)

    assert first.cells.tolist() == [[-1, 5, 0], [5, 0, 0]]
    assert middle.cells.tolist() == [[5, 0, 0], [10, 5, 0]]
    assert last.cells.tolist() == [[-1, 5, 0], [5, 10, 0]]
    assert first.filter_segment_count == 2
    assert middle.filter_segment_count == 2
    assert last.filter_segment_count == 2
    assert first.to_dict()["scope"] == "local_swept_segments"
    assert first.to_dict()["closed_trajectory"] is True


def test_local_swept_filter_handles_one_and_two_center_closed_trajectories() -> None:
    cells = np.asarray(((0, 0, 0),), dtype=np.int64)
    one = prepare_court_aabb_trajectory_filter(
        cells,
        trajectory_centers_scene_m=np.asarray(((0.5, 0.5, 0.5),), dtype=np.float64),
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS,
        radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
        resolved_radius_m=0.01,
        maximum_cells=1,
    ).filter(frame_index=0)
    two_prepared = prepare_court_aabb_trajectory_filter(
        cells,
        trajectory_centers_scene_m=np.asarray(
            ((0.5, 0.5, 0.5), (2.0, 0.5, 0.5)),
            dtype=np.float64,
        ),
        voxel_size_m=1.0,
        scope=CourtAABBTrajectoryFilterScope.LOCAL_SWEPT_SEGMENTS,
        radius_mode=CourtAABBTrajectoryFilterRadiusMode.EXPLICIT_RADIUS,
        resolved_radius_m=0.01,
        maximum_cells=1,
    )

    assert one.filter_segment_count == 1
    assert one.trajectory_segment_count == 1
    assert two_prepared.filter(frame_index=0).filter_segment_count == 2
    assert two_prepared.filter(frame_index=1).filter_segment_count == 2
    with pytest.raises(ValueError, match="canonical frame_index"):
        two_prepared.filter(frame_index=2)


def test_exposed_face_edges_are_canonical_deduplicated_and_deterministic() -> None:
    single_faces = extract_exposed_voxel_faces(
        np.asarray(((0, 0, 2),), dtype=np.int64),
        voxel_size_m=1.0,
        maximum_cells=1,
        maximum_surface_faces=6,
    )
    adjacent_faces = extract_exposed_voxel_faces(
        np.asarray(((0, 0, 2), (1, 0, 2)), dtype=np.int64),
        voxel_size_m=1.0,
        maximum_cells=2,
        maximum_surface_faces=10,
    )

    single_edges = extract_canonical_exposed_face_edges(
        single_faces,
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        maximum_edge_segments=12,
    )
    boundary_edges = extract_canonical_exposed_face_edges(
        adjacent_faces,
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        maximum_edge_segments=16,
    )
    all_edges = extract_canonical_exposed_face_edges(
        adjacent_faces,
        wireframe_topology=CourtAABBWireframeTopology.ALL_EDGES,
        maximum_edge_segments=20,
    )
    repeated = extract_canonical_exposed_face_edges(
        adjacent_faces,
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        maximum_edge_segments=16,
    )

    assert single_edges.shape == (12, 2, 3)
    assert boundary_edges.shape == (16, 2, 3)
    assert all_edges.shape == (20, 2, 3)
    np.testing.assert_array_equal(boundary_edges, repeated)
    canonical = [
        (tuple(segment[0]), tuple(segment[1])) for segment in boundary_edges
    ]
    assert all(start < end for start, end in canonical)
    assert canonical == sorted(set(canonical))
    seam = ((1.0, 0.0, 3.0), (1.0, 1.0, 3.0))
    assert seam not in canonical
    assert seam in [(tuple(segment[0]), tuple(segment[1])) for segment in all_edges]
    assert not boundary_edges.flags.writeable


@pytest.mark.parametrize(
    ("side_length", "all_edge_count", "boundary_edge_count"),
    ((2, 32, 20), (3, 60, 28)),
)
def test_boundary_topology_suppresses_planar_grid_seams(
    side_length: int,
    all_edge_count: int,
    boundary_edge_count: int,
) -> None:
    cells = np.asarray(
        [
            (x_index, y_index, 2)
            for x_index in range(side_length)
            for y_index in range(side_length)
        ],
        dtype=np.int64,
    )
    faces = extract_exposed_voxel_faces(
        cells,
        voxel_size_m=1.0,
        maximum_cells=side_length * side_length,
        maximum_surface_faces=6 * side_length * side_length,
    )

    all_edges = extract_canonical_exposed_face_edges(
        faces,
        wireframe_topology=CourtAABBWireframeTopology.ALL_EDGES,
        maximum_edge_segments=all_edge_count,
    )
    boundary_edges = extract_canonical_exposed_face_edges(
        faces,
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        maximum_edge_segments=boundary_edge_count,
    )

    assert all_edges.shape == (all_edge_count, 2, 3)
    assert boundary_edges.shape == (boundary_edge_count, 2, 3)
    assert all_edge_count - boundary_edge_count == (12 if side_length == 2 else 32)


def test_boundary_topology_retains_incidence_one_and_four_edges() -> None:
    face = np.asarray(
        (((0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (1.0, 1.0, 2.0), (0.0, 1.0, 2.0)),),
        dtype=np.float64,
    )

    incidence_one = extract_canonical_exposed_face_edges(
        face,
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        maximum_edge_segments=4,
    )
    incidence_four = extract_canonical_exposed_face_edges(
        np.repeat(face, 4, axis=0),
        wireframe_topology=CourtAABBWireframeTopology.BOUNDARY,
        maximum_edge_segments=4,
    )

    assert incidence_one.shape == (4, 2, 3)
    np.testing.assert_array_equal(incidence_four, incidence_one)


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

    assert behind.stats.near_rejected_edge_segment_count == 12
    assert behind.stats.raster_edge_segment_count == 0
    assert behind.stats.surface_pixel_count == 0
    assert crossing.stats.near_rejected_edge_segment_count >= 1
    assert crossing.stats.near_clipped_edge_segment_count >= 1
    assert crossing.stats.raster_edge_segment_count > 0
    assert crossing.stats.drawn_pixel_count > 0


def test_local_z_buffer_keeps_nearest_surface_instead_of_last_triangle() -> None:
    near_and_far = np.asarray(((0, 0, 2), (0, 0, 4)), dtype=np.int64)
    far_only = np.asarray(((0, 0, 4),), dtype=np.int64)

    combined = _render(near_and_far, depth_value=2.5)
    distant = _render(far_only, depth_value=2.5)

    np.testing.assert_array_equal(combined.rgb[9, 8], (255, 0, 0))
    np.testing.assert_array_equal(distant.rgb[9, 8], (0, 0, 0))
    assert combined.stats.drawn_pixel_count > 0
    assert distant.stats.occluded_pixel_count > 0


def test_wireframe_draws_boundaries_without_filling_face_interiors() -> None:
    wireframe = _render(
        np.asarray(((0, 0, 1),), dtype=np.int64),
        config=_config(voxel_size_m=2.0),
    )
    solid = _render(
        np.asarray(((0, 0, 1),), dtype=np.int64),
        config=_config(
            voxel_size_m=2.0,
            render_style=CourtAABBRenderStyle.SOLID,
        ),
    )

    np.testing.assert_array_equal(wireframe.rgb[11, 8], (255, 0, 0))
    np.testing.assert_array_equal(wireframe.rgb[11, 11], (0, 0, 0))
    np.testing.assert_array_equal(solid.rgb[11, 11], (255, 0, 0))
    assert wireframe.stats.edge_pixel_count == wireframe.stats.surface_pixel_count
    assert wireframe.stats.raster_triangle_count == 0
    assert wireframe.stats.raster_edge_segment_count == 12
    assert solid.stats.edge_pixel_count == 0
    assert solid.stats.raster_triangle_count > 0
    assert solid.stats.raster_edge_segment_count == 0


def test_solid_style_does_not_extract_or_limit_unused_edge_geometry() -> None:
    result = _render(
        np.asarray(((0, 0, 2),), dtype=np.int64),
        config=_config(
            render_style=CourtAABBRenderStyle.SOLID,
            maximum_edge_segments=1,
        ),
    )

    assert result.stats.candidate_edge_segment_count == 0
    assert result.stats.edge_segment_count == 0
    assert result.stats.suppressed_seam_segment_count == 0
    assert result.stats.raster_edge_segment_count == 0
    assert result.stats.raster_triangle_count > 0


def test_wireframe_topology_counts_and_filtered_limit_are_explicit() -> None:
    cells = np.asarray(((0, 0, 2), (1, 0, 2)), dtype=np.int64)

    boundary = _render(
        cells,
        config=_config(maximum_edge_segments=16),
    )

    assert boundary.stats.candidate_edge_segment_count == 20
    assert boundary.stats.edge_segment_count == 16
    assert boundary.stats.suppressed_seam_segment_count == 4
    with pytest.raises(ValueError, match="maximum_edge_segments=16"):
        _render(
            cells,
            config=_config(
                wireframe_topology=CourtAABBWireframeTopology.ALL_EDGES,
                maximum_edge_segments=16,
            ),
        )


def test_edge_width_expands_only_the_wireframe_raster() -> None:
    cells = np.asarray(((0, 0, 2),), dtype=np.int64)

    one_pixel = _render(cells, config=_config(edge_width_px=1))
    three_pixels = _render(cells, config=_config(edge_width_px=3))

    assert three_pixels.stats.edge_pixel_count > one_pixel.stats.edge_pixel_count
    assert three_pixels.stats.projected_pixel_count == (
        9 * one_pixel.stats.projected_pixel_count
    )


def test_wireframe_edge_depth_is_perspective_correct_camera_z() -> None:
    cells = np.asarray(((1, 0, 2),), dtype=np.int64)
    expected_depth_at_pixel = 1.0 / (0.25 / 2.0 + 0.75 / 3.0)

    occluded = _render(cells, depth_value=expected_depth_at_pixel - 0.005)
    visible = _render(cells, depth_value=expected_depth_at_pixel + 0.005)

    np.testing.assert_array_equal(occluded.rgb[11, 11], (0, 0, 0))
    np.testing.assert_array_equal(visible.rgb[11, 11], (255, 0, 0))


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
        config=_config(render_style=CourtAABBRenderStyle.SOLID),
    )
    depth.fill(expected_camera_z + 0.01)
    visible = render_court_obstacle_aabbs(
        rgb=rgb,
        alpha=alpha,
        metric_depth=depth,
        camera=camera,
        occupancy_cells=cells,
        config=_config(render_style=CourtAABBRenderStyle.SOLID),
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

    np.testing.assert_array_equal(occluded.rgb[9, 8], (0, 0, 0))
    np.testing.assert_array_equal(epsilon_visible.rgb[9, 8], (255, 0, 0))
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
        config=_config(edge_opacity=0.5),
    )

    np.testing.assert_array_equal(empty.rgb[0, 0], (128, 0, 128))
    np.testing.assert_array_equal(surface.rgb[9, 8], (128, 0, 128))
    assert surface.stats.background_valid_pixel_count == 0
    assert surface.stats.drawn_pixel_count > 0


@pytest.mark.parametrize(
    ("config", "cells", "match"),
    (
        (_config(maximum_cells=1), np.asarray(((0, 0, 2), (1, 0, 2)), dtype=np.int64), "maximum_cells"),
        (_config(maximum_surface_faces=5), np.asarray(((0, 0, 2),), dtype=np.int64), "maximum_surface_faces"),
        (_config(maximum_edge_segments=11), np.asarray(((0, 0, 2),), dtype=np.int64), "maximum_edge_segments"),
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
        ({"edge_opacity": 0.0}, "edge_opacity"),
        ({"edge_width_px": 0}, "edge_width_px"),
        ({"edge_width_px": 65}, "edge_width_px"),
        ({"render_style": "wireframe"}, "render_style"),
        ({"wireframe_topology": "boundary"}, "wireframe_topology"),
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
