from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.synthetic_data_generation.configuration import CourtTrajectoryPolicyV4
from src.synthetic_data_generation.dataset.court.components.camera_sampling.anchored_paths import (
    generate_anchored_rounded_rectangle_candidates,
    public_camera_inventory_digest,
    validate_anchored_trajectory_provenance,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.path_geometry import (
    closed_path_points_local,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    build_trajectory_support_model,
    evaluate_trajectory_safety,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    OrbitCoverageObjective,
    OrbitSamplingMode,
    OrbitSamplingPolicy,
    OrbitStableFieldV4,
    PathConstructorV4,
    PathFamilyV4,
    TrajectorySupportPolicy,
    VerticalProfileV4,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _camera(index: int) -> SceneCamera:
    angle = 2.0 * math.pi * index / 24
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (2.0 * math.cos(angle), 2.0 * math.sin(angle), 2.0)
    return SceneCamera(
        camera_id=f"public-camera-{index:02d}",
        source_frame_index=index,
        width=64,
        height=48,
        intrinsics=(100.0, 0.0, 31.5, 0.0, 100.0, 23.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(transform),
        image_path=f"images/{index:02d}.png",
    )


def _support_policy() -> TrajectorySupportPolicy:
    return TrajectorySupportPolicy(
        decision_id="anchored-unit-v1",
        support_radius_m=0.5,
        endpoint_radius_m=0.45,
        maximum_camera_link_distance_m=1.0,
        maximum_source_frame_gap=1,
        occupancy_voxel_size_m=0.2,
        minimum_points_per_voxel=1,
        obstacle_inflation_m=0.1,
        camera_ball_clearance_m=0.05,
        camera_capsule_clearance_m=0.04,
        sweep_step_m=0.05,
        boundary_epsilon_m=1.0e-6,
        minimum_captured_cameras=24,
        minimum_public_points=1,
        maximum_capsule_index_cells=100_000,
        maximum_occupancy_cells=100_000,
        minimum_cycle_frame_span=8,
        maximum_cycle_frame_span=16,
        maximum_cycle_closure_distance_m=0.3,
        maximum_constructive_cycle_count=24,
        cycle_smoothing_distance_m=0.03,
    )


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


def _sampling_policy() -> OrbitSamplingPolicy:
    return OrbitSamplingPolicy(
        mode=OrbitSamplingMode.UNIFORM_ARC_LENGTH,
        max_arc_step_m=0.1,
        minimum_sample_count=24,
        sample_count_multiple=8,
        seed=695,
        stable_field_order=tuple(OrbitStableFieldV4),
        coverage_objective=tuple(OrbitCoverageObjective),
        proposal_budget=4_800,
        minimum_trajectory_groups=24,
        minimum_accepted_frames=2_000,
        minimum_accepted_fraction=0.9,
        split_fractions=(0.8, 0.1, 0.1),
        shard_count=8,
    )


def test_anchored_candidates_use_every_ordered_public_camera_and_seed() -> None:
    cameras = tuple(_camera(index) for index in range(24))
    support = build_trajectory_support_model(
        cameras=cameras,
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_support_policy(),
    )
    center = OrbitCenter(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id="court-0",
        scene_from_center=RigidTransform.identity(),
        base_radius_m=2.0,
        captured_offset_median_m=2.0,
        captured_offset_q90_m=2.0,
        captured_camera_count=len(cameras),
    )

    first = generate_anchored_rounded_rectangle_candidates(
        support_model=support,
        cameras=cameras,
        centers=(center,),
        policy=_trajectory_policy(),
        seed=695,
    )
    repeated = generate_anchored_rounded_rectangle_candidates(
        support_model=support,
        cameras=cameras,
        centers=(center,),
        policy=_trajectory_policy(),
        seed=695,
    )
    different_seed = generate_anchored_rounded_rectangle_candidates(
        support_model=support,
        cameras=cameras,
        centers=(center,),
        policy=_trajectory_policy(),
        seed=696,
    )
    reversed_input = generate_anchored_rounded_rectangle_candidates(
        support_model=support,
        cameras=tuple(reversed(cameras)),
        centers=(center,),
        policy=_trajectory_policy(),
        seed=695,
    )

    assert first == repeated
    assert first == reversed_input
    assert first != different_seed
    assert public_camera_inventory_digest(cameras) == (
        public_camera_inventory_digest(tuple(reversed(cameras)))
    )
    assert len(first) == 48
    assert {item.constructor for item in first} == {
        PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
    }
    assert {
        item.anchor_provenance.ordered_camera_index
        for item in first
        if item.anchor_provenance is not None
    } == set(range(24))
    assert {
        item.anchor_provenance.camera_inventory_digest
        for item in first
        if item.anchor_provenance is not None
    } == {public_camera_inventory_digest(cameras)}


def test_anchored_geometry_points_are_genuine_safe_planar_and_raised_paths() -> None:
    cameras = tuple(_camera(index) for index in range(24))
    support = build_trajectory_support_model(
        cameras=cameras,
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_support_policy(),
    )
    center = OrbitCenter(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id="court-0",
        scene_from_center=RigidTransform.identity(),
        base_radius_m=2.0,
        captured_offset_median_m=2.0,
        captured_offset_q90_m=2.0,
        captured_camera_count=len(cameras),
    )
    candidates = generate_anchored_rounded_rectangle_candidates(
        support_model=support,
        cameras=cameras,
        centers=(center,),
        policy=_trajectory_policy(),
        seed=695,
    )
    planar = next(
        item for item in candidates if item.curve_mode is VerticalProfileV4.PLANAR
    )
    raised = next(
        item
        for item in candidates
        if item.curve_mode is VerticalProfileV4.RAISED_PHASES
    )

    for trajectory in (planar, raised):
        anchor = trajectory.anchor_provenance
        assert anchor is not None
        fractions = np.arange(len(anchor.reference_points_local_m)) / len(
            anchor.reference_points_local_m
        )
        actual = closed_path_points_local(trajectory, fractions)
        assert np.allclose(actual, anchor.reference_points_local_m, atol=1.0e-12)
        path = sample_uniform_arc_length(trajectory, center, _sampling_policy())
        evaluation = evaluate_trajectory_safety(
            trajectory_id=trajectory.trajectory_id,
            trajectory_group_id=trajectory.trajectory_group_id,
            path=path,
            support_model=support,
        )
        assert evaluation.safe
    planar_anchor = planar.anchor_provenance
    raised_anchor = raised.anchor_provenance
    assert planar_anchor is not None and raised_anchor is not None
    assert np.ptp(np.asarray(planar_anchor.reference_points_local_m)[:, 2]) == 0.0
    assert np.ptp(np.asarray(raised_anchor.reference_points_local_m)[:, 2]) == 0.25


def test_anchored_provenance_binds_public_camera_identity_and_geometry() -> None:
    cameras = tuple(_camera(index) for index in range(24))
    support = build_trajectory_support_model(
        cameras=cameras,
        points_scene_m=np.asarray(((20.0, 20.0, 0.0),), dtype=np.float64),
        policy=_support_policy(),
    )
    center = OrbitCenter(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id="court-0",
        scene_from_center=RigidTransform.identity(),
        base_radius_m=2.0,
        captured_offset_median_m=2.0,
        captured_offset_q90_m=2.0,
        captured_camera_count=len(cameras),
    )
    trajectory = generate_anchored_rounded_rectangle_candidates(
        support_model=support,
        cameras=cameras,
        centers=(center,),
        policy=_trajectory_policy(),
        seed=695,
    )[0]
    provenance = trajectory.anchor_provenance
    assert provenance is not None
    validate_anchored_trajectory_provenance(
        trajectory,
        center=center,
        support_model=support,
    )

    reference_points = list(provenance.reference_points_local_m)
    first_point = reference_points[0]
    reference_points[0] = (
        first_point[0] + 0.01,
        first_point[1],
        first_point[2],
    )
    tampered_provenance = (
        replace(provenance, camera_inventory_digest="0" * 64),
        replace(provenance, camera_id="different-public-camera"),
        replace(provenance, source_frame_index=provenance.source_frame_index + 1),
        replace(
            provenance,
            reference_points_local_m=tuple(reference_points),
        ),
    )
    for tampered in tampered_provenance:
        with pytest.raises(ValueError, match="Anchored"):
            validate_anchored_trajectory_provenance(
                replace(trajectory, anchor_provenance=tampered),
                center=center,
                support_model=support,
            )

    changed_cameras = (replace(cameras[0], camera_id="changed-camera"), *cameras[1:])
    with pytest.raises(ValueError, match="camera inventory"):
        generate_anchored_rounded_rectangle_candidates(
            support_model=support,
            cameras=changed_cameras,
            centers=(center,),
            policy=_trajectory_policy(),
            seed=695,
        )
