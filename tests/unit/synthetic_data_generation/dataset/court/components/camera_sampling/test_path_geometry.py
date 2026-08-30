from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.components.camera_sampling.path_geometry import (
    closed_path_points_local,
    rounded_rectangle_xy_and_tangent,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    OrbitCoverageObjective,
    OrbitSamplingMode,
    OrbitSamplingPolicy,
    OrbitStableFieldV4,
    OrbitTrajectorySpecV4,
    PathConstructorV4,
    PathFamilyV4,
    VerticalProfileV4,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _trajectory() -> OrbitTrajectorySpecV4:
    return OrbitTrajectorySpecV4(
        trajectory_id="trajectory-v4",
        trajectory_group_id="group-v4",
        shape=PathFamilyV4.ROUNDED_RECTANGLE,
        center_kind=OrbitCenterKind.COMPLEX,
        center_court_instance_id=None,
        base_radius_m=5.0,
        radius_scale=1.0,
        axis_ratio=0.6,
        orientation_radians=math.radians(17.0),
        base_height_m=2.0,
        vertical_amplitude_m=0.5,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=VerticalProfileV4.RAISED_PHASES,
        corner_radius_ratio=0.25,
        vertical_phase_offsets_m=(0.0, 0.5, 0.5, 0.0),
    )


def test_rounded_rectangle_is_closed_bounded_and_tangent_continuous() -> None:
    half_width = 5.0
    half_height = 3.0
    radius = 0.75
    horizontal = 2.0 * (half_width - radius)
    vertical = 2.0 * (half_height - radius)
    quarter = 0.5 * math.pi * radius
    lengths = (horizontal, quarter, vertical, quarter) * 2
    perimeter = sum(lengths)
    boundaries = np.cumsum(np.asarray((0.0, *lengths))) / perimeter
    epsilon = 1.0e-9
    fractions = np.asarray(
        [
            0.0,
            1.0,
            *(
                value
                for boundary in boundaries[1:-1]
                for value in (boundary - epsilon, boundary, boundary + epsilon)
            ),
        ],
        dtype=np.float64,
    )

    xy, tangent = rounded_rectangle_xy_and_tangent(
        half_width_m=half_width,
        half_height_m=half_height,
        corner_radius_m=radius,
        fractions=fractions,
    )

    np.testing.assert_allclose(xy[0], xy[1], atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(tangent[0], tangent[1], atol=1.0e-12, rtol=0.0)
    assert np.max(np.abs(xy[:, 0])) <= half_width
    assert np.max(np.abs(xy[:, 1])) <= half_height
    for index in range(2, len(fractions), 3):
        assert float(tangent[index] @ tangent[index + 1]) > 1.0 - 1.0e-12
        assert float(tangent[index + 1] @ tangent[index + 2]) > 1.0 - 1.0e-12


def test_raised_phases_are_periodic_and_uniform_3d_sampling_checks_the_seam() -> None:
    trajectory = _trajectory()
    fractions = np.linspace(0.0, 1.0, 257, endpoint=True)
    local = closed_path_points_local(trajectory, fractions)

    np.testing.assert_allclose(local[0], local[-1], atol=1.0e-12, rtol=0.0)
    assert local[:, 2].min() == trajectory.base_height_m
    assert local[:, 2].max() == pytest.approx(trajectory.base_height_m + 0.5)

    center = OrbitCenter(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id="court-0",
        scene_from_center=RigidTransform.identity(),
        base_radius_m=5.0,
        captured_offset_median_m=4.0,
        captured_offset_q90_m=5.0,
        captured_camera_count=12,
    )
    policy = OrbitSamplingPolicy(
        mode=OrbitSamplingMode.UNIFORM_ARC_LENGTH,
        max_arc_step_m=0.35,
        minimum_sample_count=24,
        sample_count_multiple=8,
        seed=823,
        stable_field_order=tuple(OrbitStableFieldV4),
        coverage_objective=tuple(OrbitCoverageObjective),
        proposal_budget=4_800,
        minimum_trajectory_groups=24,
        minimum_accepted_frames=2_000,
        minimum_accepted_fraction=0.9,
        split_fractions=(0.8, 0.1, 0.1),
        shard_count=8,
    )

    sampled = sample_uniform_arc_length(trajectory, center, policy)

    assert len(sampled.adjacent_steps_m) == len(sampled.points_scene_m)
    assert sampled.adjacent_steps_m[-1] <= policy.max_arc_step_m + 1.0e-9
    assert float(sampled.adjacent_steps_m.max()) <= policy.max_arc_step_m + 1.0e-9
    assert float(sampled.adjacent_steps_m.max() - sampled.adjacent_steps_m.min()) < 0.03


def test_free_space_cycle_orientation_and_base_height_transform_geometry() -> None:
    controls = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (2.0, 2.0, 0.0),
        (1.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    trajectory = OrbitTrajectorySpecV4(
        trajectory_id="trajectory-free-space",
        trajectory_group_id="group-free-space",
        shape=PathFamilyV4.FREE_SPACE_CYCLE,
        center_kind=OrbitCenterKind.COMPLEX,
        center_court_instance_id=None,
        base_radius_m=2.0,
        radius_scale=1.0,
        axis_ratio=1.0,
        orientation_radians=0.0,
        base_height_m=2.0,
        vertical_amplitude_m=0.0,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=VerticalProfileV4.FREE_SPACE_CYCLE,
        constructor=PathConstructorV4.FREE_SPACE_CYCLE,
        corner_radius_ratio=None,
        vertical_phase_offsets_m=(0.0,),
        control_points_local_m=controls,
    )
    fractions = np.asarray((0.0, 0.125, 0.25), dtype=np.float64)

    base = closed_path_points_local(trajectory, fractions)
    rotated = closed_path_points_local(
        replace(trajectory, orientation_radians=0.5 * math.pi),
        fractions,
    )
    raised = closed_path_points_local(
        replace(trajectory, base_height_m=3.0),
        fractions,
    )

    np.testing.assert_allclose(
        rotated[:, :2],
        base[:, :2] @ np.asarray(((0.0, 1.0), (-1.0, 0.0))),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(raised[:, 2], base[:, 2] + 1.0)
