from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.synthetic_data_generation.configuration import CourtTrajectoryPolicyV4
from src.synthetic_data_generation.dataset.court.components.camera_sampling.constructive_paths import (
    construct_free_space_cycles,
    generate_free_space_cycle_candidates,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.path_geometry import (
    closed_path_points_local,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    build_trajectory_support_model,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    PathFamilyV4,
    TrajectorySupportPolicy,
    VerticalProfileV4,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


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


def _policy() -> TrajectorySupportPolicy:
    return TrajectorySupportPolicy(
        decision_id="constructive-unit-v1",
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


def _two_laps() -> tuple[SceneCamera, ...]:
    lap = (
        (0.0, 0.0, 2.0),
        (1.0, 0.0, 2.0),
        (2.0, 0.0, 2.0),
        (2.0, 1.0, 2.0),
        (2.0, 2.0, 2.0),
        (1.0, 2.0, 2.0),
        (0.0, 2.0, 2.0),
        (0.0, 1.0, 2.0),
        (0.1, 0.0, 2.0),
    )
    return tuple(_camera(index, point) for index, point in enumerate((*lap, *lap[1:])))


def _trajectory_policy() -> CourtTrajectoryPolicyV4:
    return CourtTrajectoryPolicyV4(
        shapes=(
            PathFamilyV4.CIRCLE,
            PathFamilyV4.ELLIPSE,
            PathFamilyV4.ROUNDED_RECTANGLE,
        ),
        axis_ratios=(1.0, 0.8),
        orientations_degrees=(0.0, 45.0, 90.0),
        center_kinds=(OrbitCenterKind.COMPLEX,),
        captured_offset_scale_range=(0.85, 1.2),
        base_heights_m=(1.5, 2.25, 3.0),
        vertical_modulations_m=(0.0, 0.25),
        curve_modes=(
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.SINUSOIDAL_HEIGHT,
            VerticalProfileV4.RAISED_PHASES,
        ),
        corner_radius_ratios=(0.25,),
        vertical_phase_offsets_m=((0.0, 0.5, 0.5, 0.0),),
        anchored_half_width_m=0.1,
        anchored_half_height_m=0.1,
        anchored_corner_radius_m=0.04,
        anchored_raised_lift_m=0.25,
        anchored_reference_point_count=32,
    )


def test_constructive_cycle_uses_safe_temporal_edges_and_validated_smoothed_seam() -> (
    None
):
    policy = replace(
        _policy(),
        support_radius_m=0.8,
        endpoint_radius_m=0.6,
        maximum_cycle_closure_distance_m=1.1,
    )
    model = build_trajectory_support_model(
        cameras=_two_laps(),
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=policy,
    )

    cycles = construct_free_space_cycles(model)

    assert cycles
    assert all(
        model.segment_is_safe(first, second)
        for cycle in cycles
        for first, second in zip(
            cycle.control_points_scene_m,
            np.roll(cycle.control_points_scene_m, -1, axis=0),
            strict=True,
        )
    )


def test_constructive_candidates_are_typed_3d_cycles_with_inset_and_height_variants() -> (
    None
):
    policy = replace(
        _policy(),
        support_radius_m=0.8,
        endpoint_radius_m=0.6,
        maximum_cycle_closure_distance_m=1.1,
    )
    model = build_trajectory_support_model(
        cameras=_two_laps(),
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=policy,
    )
    center = OrbitCenter(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id="court-0",
        scene_from_center=RigidTransform.identity(),
        base_radius_m=2.0,
        captured_offset_median_m=1.5,
        captured_offset_q90_m=2.0,
        captured_camera_count=len(_two_laps()),
    )

    candidates = generate_free_space_cycle_candidates(
        support_model=model,
        centers=(center,),
        policy=_trajectory_policy(),
    )

    assert candidates
    assert {candidate.shape for candidate in candidates} == {
        PathFamilyV4.FREE_SPACE_CYCLE
    }
    assert {candidate.curve_mode for candidate in candidates} == {
        VerticalProfileV4.FREE_SPACE_CYCLE
    }
    assert {candidate.radius_scale for candidate in candidates} >= {0.97, 1.0}
    assert len({candidate.control_points_local_m for candidate in candidates}) == len(
        candidates
    )
    for candidate in candidates[::2]:
        local = closed_path_points_local(
            candidate,
            np.linspace(0.0, 1.0, 129, endpoint=False),
        )
        scene = center.scene_from_center.apply(local)
        assert all(
            model.segment_is_safe(first, second)
            for first, second in zip(
                scene,
                np.roll(scene, -1, axis=0),
                strict=True,
            )
        )
