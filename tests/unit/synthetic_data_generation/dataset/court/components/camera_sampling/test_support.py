from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.components.camera_sampling import (
    support as support_module,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportError,
    build_trajectory_support_model,
    evaluate_trajectory_safety,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitPathSamples,
    TrajectorySafetyReason,
    TrajectorySupportPolicy,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _policy() -> TrajectorySupportPolicy:
    return TrajectorySupportPolicy(
        decision_id="unit-support-v1",
        support_radius_m=0.3,
        endpoint_radius_m=0.2,
        maximum_camera_link_distance_m=1.5,
        maximum_source_frame_gap=1,
        occupancy_voxel_size_m=0.2,
        minimum_points_per_voxel=1,
        obstacle_inflation_m=0.2,
        camera_ball_clearance_m=0.05,
        camera_capsule_clearance_m=0.04,
        sweep_step_m=0.1,
        boundary_epsilon_m=1.0e-6,
        minimum_captured_cameras=2,
        minimum_public_points=1,
        maximum_capsule_index_cells=10_000,
        maximum_occupancy_cells=10_000,
        minimum_cycle_frame_span=8,
        maximum_cycle_frame_span=16,
        maximum_cycle_closure_distance_m=0.3,
        maximum_constructive_cycle_count=24,
        cycle_smoothing_distance_m=0.03,
    )


def _camera(index: int, xyz: tuple[float, float, float]) -> SceneCamera:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    return SceneCamera(
        camera_id=f"camera-{index}",
        source_frame_index=index,
        width=64,
        height=48,
        intrinsics=(100.0, 0.0, 31.5, 0.0, 100.0, 23.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(transform),
        image_path=f"images/{index}.png",
    )


def _open_loop_cameras() -> tuple[SceneCamera, ...]:
    positions = (
        (0.0, 0.0, 2.0),
        (1.0, 0.0, 2.0),
        (2.0, 0.0, 2.0),
        (2.0, 1.0, 2.0),
        (2.0, 2.0, 2.0),
        (1.0, 2.0, 2.0),
        (0.0, 2.0, 2.0),
        (0.0, 1.0, 2.0),
    )
    return tuple(_camera(index, position) for index, position in enumerate(positions))


def _path(points: np.ndarray) -> OrbitPathSamples:
    closed = np.vstack((points, points[:1]))
    steps = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    return OrbitPathSamples(
        trajectory_group_id="group-open-loop",
        theta_radians=np.linspace(0.0, 2.0 * math.pi, len(points), endpoint=False),
        points_local_m=points,
        points_scene_m=points,
        adjacent_steps_m=steps,
        total_arc_length_m=float(steps.sum()),
    )


def _forbidden_range(*_args: int) -> range:
    raise AssertionError("oversized occupancy AABB reached nested iteration")


def test_closed_sweep_rejects_only_the_unsupported_seam() -> None:
    cameras = _open_loop_cameras()
    points = np.asarray(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in cameras],
        dtype=np.float64,
    )
    model = build_trajectory_support_model(
        cameras=cameras,
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_policy(),
    )

    evaluation = evaluate_trajectory_safety(
        trajectory_id="trajectory-open-loop",
        trajectory_group_id="group-open-loop",
        path=_path(points),
        support_model=model,
    )

    assert not evaluation.safe
    assert evaluation.violating_point_indices == ()
    assert evaluation.violating_segment_indices == (len(points) - 1,)
    assert evaluation.reasons == (TrajectorySafetyReason.SWEPT_SEGMENT_OUTSIDE_SUPPORT,)
    assert evaluation.closed_segment_count == len(points)


def test_support_and_obstacle_boundaries_fail_closed() -> None:
    model = build_trajectory_support_model(
        cameras=_open_loop_cameras(),
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_policy(),
    )

    margin, _clearance, supported, occupied = model.evaluate_point(
        np.asarray((0.5, 0.3, 2.0), dtype=np.float64)
    )
    assert margin == pytest.approx(0.0, abs=1.0e-12)
    assert not supported
    assert not occupied

    _margin, clearance, _supported, occupied = model.evaluate_point(
        np.asarray((20.0, 20.0, 0.0), dtype=np.float64)
    )
    assert occupied
    assert clearance < 0.0


def test_obstacle_clearance_matches_exact_nearest_inflated_voxel() -> None:
    model = build_trajectory_support_model(
        cameras=_open_loop_cameras(),
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_policy(),
    )
    query = np.asarray((19.35, 20.05, 0.05), dtype=np.float64)

    _margin, clearance, _supported, occupied = model.evaluate_point(query)

    voxel = model.policy.occupancy_voxel_size_m
    lower = np.asarray(sorted(model.inflated_occupancy), dtype=np.float64) * voxel
    upper = lower + voxel
    delta = np.maximum(np.maximum(lower - query, query - upper), 0.0)
    exact = min(
        model.policy.support_radius_m,
        float(np.min(np.linalg.norm(delta, axis=1))),
    )
    assert not occupied
    assert clearance == pytest.approx(exact, abs=1.0e-12)

    occupied_cell = next(iter(model.inflated_occupancy))
    exposed_x = max(
        cell[0] for cell in model.inflated_occupancy if cell[1:] == occupied_cell[1:]
    )
    boundary_query = np.asarray(
        (
            np.nextafter((exposed_x + 1) * voxel, math.inf),
            (occupied_cell[1] + 0.5) * voxel,
            (occupied_cell[2] + 0.5) * voxel,
        ),
        dtype=np.float64,
    )
    assert tuple(np.floor(boundary_query / voxel).astype(int)) not in (
        model.inflated_occupancy
    )

    _margin, boundary_clearance, _supported, boundary_occupied = model.evaluate_point(
        boundary_query
    )

    assert boundary_clearance <= model.policy.boundary_epsilon_m
    assert boundary_occupied


def test_safety_diagnostic_clearance_includes_safe_swept_segment_interior() -> None:
    model = build_trajectory_support_model(
        cameras=_open_loop_cameras(),
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_policy(),
    )
    near_miss_cell = (2, 1, 10)
    occupancy_centers, occupancy_index = support_module._build_occupancy_index(
        frozenset({near_miss_cell}),
        policy=model.policy,
    )
    model = replace(
        model,
        inflated_occupancy=frozenset({near_miss_cell}),
        occupancy_centers_m=occupancy_centers,
        occupancy_index=occupancy_index,
    )
    points = np.asarray(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in _open_loop_cameras()],
        dtype=np.float64,
    )

    evaluation = evaluate_trajectory_safety(
        trajectory_id="trajectory-safe-near-miss",
        trajectory_group_id="group-open-loop",
        path=_path(points),
        support_model=model,
    )

    assert (
        TrajectorySafetyReason.SWEPT_SEGMENT_HITS_INFLATED_OBSTACLE
        not in evaluation.reasons
    )
    assert evaluation.minimum_obstacle_clearance_m == pytest.approx(0.2, abs=1.0e-12)


def test_missing_and_nonfinite_public_support_inputs_have_stable_reasons() -> None:
    with pytest.raises(TrajectorySupportError) as insufficient:
        build_trajectory_support_model(
            cameras=(_open_loop_cameras()[0],),
            points_scene_m=np.asarray(((0.0, 0.0, 0.0),), dtype=np.float64),
            policy=_policy(),
        )
    assert (
        insufficient.value.reason
        is TrajectorySafetyReason.INSUFFICIENT_CAPTURED_CAMERAS
    )

    with pytest.raises(TrajectorySupportError) as nonfinite:
        build_trajectory_support_model(
            cameras=_open_loop_cameras(),
            points_scene_m=np.asarray(((math.nan, 0.0, 0.0),), dtype=np.float64),
            policy=_policy(),
        )
    assert nonfinite.value.reason is TrajectorySafetyReason.NONFINITE_SUPPORT_INPUT


def test_captured_camera_balls_and_temporal_capsules_are_carved_from_occupancy() -> (
    None
):
    cameras = _open_loop_cameras()
    captured_points = np.asarray(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in cameras],
        dtype=np.float64,
    )
    model = build_trajectory_support_model(
        cameras=cameras,
        points_scene_m=np.vstack((captured_points, ((20.0, 20.0, 0.0),))),
        policy=_policy(),
    )

    decisions = [model.evaluate_point(point) for point in captured_points]

    assert all(
        supported and not occupied
        for _margin, _clearance, supported, occupied in decisions
    )
    assert model.summary.captured_camera_occupied_count == 0
    assert model.summary.camera_ball_carved_cell_count > 0
    assert model.summary.capsule_count == len(cameras) - 1


def test_exact_segment_authority_rejects_an_inflated_voxel_corner_crossing() -> None:
    policy = replace(
        _policy(),
        support_radius_m=0.8,
        endpoint_radius_m=0.6,
        maximum_cycle_closure_distance_m=1.1,
    )
    model = build_trajectory_support_model(
        cameras=_open_loop_cameras(),
        points_scene_m=np.asarray(((1.0, 0.45, 2.0), (20.0, 20.0, 0.0))),
        policy=policy,
    )
    voxel = policy.occupancy_voxel_size_m
    exposed = max(
        (cell for cell in model.inflated_occupancy if cell[0] < 20),
        key=lambda cell: (cell[0] + cell[1], cell[0], cell[1]),
    )
    upper_x = (exposed[0] + 1) * voxel
    upper_y = (exposed[1] + 1) * voxel
    midpoint_z = (exposed[2] + 0.5) * voxel
    start = np.asarray((upper_x + 0.03, upper_y - 0.03, midpoint_z))
    end = np.asarray((upper_x - 0.03, upper_y + 0.03, midpoint_z))

    assert tuple(np.floor(start / voxel).astype(int)) not in model.inflated_occupancy
    assert tuple(np.floor(end / voxel).astype(int)) not in model.inflated_occupancy
    assert not model.segment_is_safe(start, end)


def test_segment_occupancy_scan_rejects_pathological_finite_aabb_before_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support_module, "range", _forbidden_range, raising=False)

    with pytest.raises(TrajectorySupportError) as raised:
        support_module._segment_hits_occupancy(
            np.zeros(3, dtype=np.float64),
            np.full(3, 1_000.0, dtype=np.float64),
            inflated=frozenset(),
            policy=_policy(),
        )

    assert raised.value.reason is TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE
    assert "exceeding maximum_occupancy_cells=10000" in str(raised.value)


def test_carving_scan_rejects_pathological_finite_aabb_before_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support_module, "range", _forbidden_range, raising=False)

    with pytest.raises(TrajectorySupportError) as raised:
        support_module._carve_occupancy(
            frozenset({(0, 0, 0)}),
            segments=(
                (
                    np.zeros(3, dtype=np.float64),
                    np.full(3, 1_000.0, dtype=np.float64),
                ),
            ),
            radius_m=_policy().camera_capsule_clearance_m,
            policy=_policy(),
        )

    assert raised.value.reason is TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE
    assert "exceeding maximum_occupancy_cells=10000" in str(raised.value)


def test_scan_work_ceiling_preserves_exact_boundary_collision() -> None:
    policy = replace(_policy(), maximum_occupancy_cells=8)
    start = np.asarray((0.1, 0.1, 0.1), dtype=np.float64)
    end = np.asarray((0.3, 0.3, 0.3), dtype=np.float64)

    assert support_module._segment_hits_occupancy(
        start,
        end,
        inflated=frozenset({(1, 1, 1)}),
        policy=policy,
    )


def test_public_segment_and_evaluation_reject_pathological_scans_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_trajectory_support_model(
        cameras=_open_loop_cameras(),
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_policy(),
    )

    monkeypatch.setattr(support_module, "range", _forbidden_range, raising=False)
    start = np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
    end = np.asarray((1_000.0, 1_000.0, 1_000.0), dtype=np.float64)

    with pytest.raises(TrajectorySupportError) as segment_raised:
        model.segment_is_safe(start, end)
    assert (
        segment_raised.value.reason
        is TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE
    )

    path_points = np.asarray(
        [camera.camera_to_scene.matrix()[:3, 3] for camera in _open_loop_cameras()],
        dtype=np.float64,
    )
    path_points[1] = end
    with pytest.raises(TrajectorySupportError) as evaluation_raised:
        evaluate_trajectory_safety(
            trajectory_id="trajectory-pathological-finite-scan",
            trajectory_group_id="group-open-loop",
            path=_path(path_points),
            support_model=model,
        )
    assert (
        evaluation_raised.value.reason
        is TrajectorySafetyReason.EMPTY_SUPPORT_FREE_SPACE
    )
