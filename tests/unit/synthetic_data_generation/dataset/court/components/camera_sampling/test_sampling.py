from __future__ import annotations

import numpy as np

from src.synthetic_data_generation.dataset.court.components.camera_sampling.sampling import (
    sample_uniform_arc_length,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenter,
    OrbitCenterKind,
    OrbitCurveMode,
    OrbitSamplingMode,
    OrbitSamplingPolicy,
    OrbitShape,
    OrbitTrajectorySpec,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def test_uniform_3d_arc_length_bounds_closed_step() -> None:
    center = OrbitCenter(
        center_kind=OrbitCenterKind.COMPLEX,
        court_instance_id=None,
        reference_court_instance_id="court-0",
        scene_from_center=RigidTransform.identity(),
        base_radius_m=20.0,
        captured_offset_median_m=18.0,
        captured_offset_q90_m=20.0,
        captured_camera_count=12,
    )
    trajectory = OrbitTrajectorySpec(
        trajectory_id="trajectory-a",
        trajectory_group_id="group-a",
        shape=OrbitShape.ELLIPSE,
        center_kind=OrbitCenterKind.COMPLEX,
        center_court_instance_id=None,
        base_radius_m=20.0,
        radius_scale=1.0,
        axis_ratio=0.6,
        orientation_radians=np.pi / 4.0,
        base_height_m=7.0,
        vertical_amplitude_m=2.0,
        vertical_cycles=2,
        vertical_phase_radians=np.pi / 3.0,
        curve_mode=OrbitCurveMode.SINUSOIDAL_HEIGHT,
    )
    policy = OrbitSamplingPolicy(
        mode=OrbitSamplingMode.UNIFORM_ARC_LENGTH,
        max_arc_step_m=1.05,
        minimum_sample_count=24,
        sample_count_multiple=8,
        seed=7,
        stable_field_order=("shape",),
        coverage_objective=("trajectory_group",),
        proposal_budget=3_000,
        minimum_trajectory_groups=24,
        minimum_accepted_frames=2_000,
        minimum_accepted_fraction=0.9,
        split_fractions=(0.8, 0.1, 0.1),
        shard_count=8,
    )
    samples = sample_uniform_arc_length(trajectory, center, policy)
    assert np.ptp(samples.points_scene_m[:, 2]) > 3.9
    assert len(samples.theta_radians) % policy.sample_count_multiple == 0
    assert samples.adjacent_steps_m.max() <= 1.05 + 1.0e-9
    assert samples.adjacent_steps_m.max() / samples.adjacent_steps_m.min() < 1.03
