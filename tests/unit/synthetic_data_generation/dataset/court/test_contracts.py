from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.contracts import (
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitCoverageObjective,
    OrbitCurveMode,
    OrbitSamplingMode,
    OrbitSamplingPolicy,
    OrbitShape,
    OrbitStableField,
    OrbitTargetKind,
    OrbitTargetMode,
    OrbitTrajectorySpec,
    OrbitViewSpec,
    OrbitViewSpecV2,
    ResolvedTargetCourtV2,
    TargetCourtPolicyV2,
    TargetCourtResolutionPolicy,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


def _trajectory() -> OrbitTrajectorySpec:
    return OrbitTrajectorySpec(
        trajectory_id="trajectory-a",
        trajectory_group_id="group-a",
        shape=OrbitShape.CIRCLE,
        center_kind=OrbitCenterKind.COMPLEX,
        center_court_instance_id=None,
        base_radius_m=20.0,
        radius_scale=1.0,
        axis_ratio=1.0,
        orientation_radians=0.0,
        base_height_m=6.0,
        vertical_amplitude_m=0.0,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=OrbitCurveMode.PLANAR,
    )


def test_typed_contracts_reject_unknown_keys_and_modes() -> None:
    trajectory = _trajectory().to_dict()
    trajectory["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        OrbitTrajectorySpec.from_mapping(trajectory)

    view = OrbitViewSpec(
        view_id="view-a",
        target_kind=OrbitTargetKind.COMPLEX,
        target_court_instance_id=None,
        target_mode=OrbitTargetMode.COMPLEX_CENTER,
        coverage_mode=OrbitCoverageMode.FULL,
        look_at_height_m=0.0,
        hfov_degrees=60.0,
    ).to_dict()
    view["coverage_mode"] = "smoke"
    with pytest.raises(ValueError):
        OrbitViewSpec.from_mapping(view)


def test_shape_and_target_semantics_fail_closed() -> None:
    values = _trajectory().to_dict()
    values["axis_ratio"] = 0.8
    with pytest.raises(ValueError, match="circle"):
        OrbitTrajectorySpec.from_mapping(values)
    with pytest.raises(ValueError, match="exactly for court targets"):
        OrbitViewSpec(
            view_id="view-a",
            target_kind=OrbitTargetKind.COURT,
            target_court_instance_id=None,
            target_mode=OrbitTargetMode.COURT_CENTER,
            coverage_mode=OrbitCoverageMode.FULL,
            look_at_height_m=0.0,
            hfov_degrees=60.0,
        )


def test_sampling_contract_rejects_unknown_keys_modes_fields_and_objectives() -> None:
    policy = OrbitSamplingPolicy(
        mode=OrbitSamplingMode.UNIFORM_ARC_LENGTH,
        max_arc_step_m=1.05,
        minimum_sample_count=24,
        sample_count_multiple=8,
        seed=7,
        stable_field_order=(OrbitStableField.SHAPE,),
        coverage_objective=(OrbitCoverageObjective.TRAJECTORY_GROUP,),
        proposal_budget=3_000,
        minimum_trajectory_groups=24,
        minimum_accepted_frames=2_000,
        minimum_accepted_fraction=0.9,
        split_fractions=(0.8, 0.1, 0.1),
        shard_count=8,
    )
    assert OrbitSamplingPolicy.from_mapping(policy.to_dict()) == policy

    for key, unknown in (
        ("mode", "unknown_sampling"),
        ("stable_field_order", ["unknown_field"]),
        ("coverage_objective", ["unknown_objective"]),
    ):
        values = policy.to_dict()
        values[key] = unknown
        with pytest.raises(ValueError):
            OrbitSamplingPolicy.from_mapping(values)

    values = policy.to_dict()
    values["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        OrbitSamplingPolicy.from_mapping(values)


def test_v2_view_policy_and_sample_target_round_trip_exact_keys() -> None:
    view = OrbitViewSpecV2(
        view_id="view-v2",
        target_kind=OrbitTargetKind.COURT,
        target_mode=OrbitTargetMode.COURT_CENTER,
        coverage_mode=OrbitCoverageMode.FULL,
        look_at_height_m=1.5,
        hfov_degrees=60.0,
    )
    fixed_policy = TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT,
        centre_court_instance_id="court-a",
    )
    nearest_policy = TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.NEAREST_CAMERA,
        centre_court_instance_id=None,
    )
    target = ResolvedTargetCourtV2(
        binding=TargetCourtBinding(
            court_instance_id="court-a",
            candidate_id="candidate-a",
            scene_from_court=RigidTransform.identity(),
            selection_seed=695,
        ),
        resolution_policy=TargetCourtResolutionPolicy.NEAREST_CAMERA,
        camera_to_court_center_distance_m=12.5,
    )

    assert OrbitViewSpecV2.from_mapping(view.to_dict()) == view
    assert TargetCourtPolicyV2.from_mapping(fixed_policy.to_dict()) == fixed_policy
    assert TargetCourtPolicyV2.from_mapping(nearest_policy.to_dict()) == nearest_policy
    assert ResolvedTargetCourtV2.from_mapping(target.to_dict()) == target
    assert "target_court_instance_id" not in view.to_dict()


def test_v2_contracts_reject_v1_fields_and_mixed_discriminants() -> None:
    view = OrbitViewSpecV2(
        view_id="view-v2",
        target_kind=OrbitTargetKind.COURT,
        target_mode=OrbitTargetMode.COURT_CENTER,
        coverage_mode=OrbitCoverageMode.FULL,
        look_at_height_m=0.0,
        hfov_degrees=60.0,
    ).to_dict()
    view["target_court_instance_id"] = "court-a"
    with pytest.raises(ValueError, match="unknown"):
        OrbitViewSpecV2.from_mapping(view)

    with pytest.raises(ValueError, match="required exactly"):
        TargetCourtPolicyV2(
            mode=TargetCourtResolutionPolicy.NEAREST_CAMERA,
            centre_court_instance_id="court-a",
        )
    with pytest.raises(ValueError, match="required exactly"):
        TargetCourtPolicyV2(
            mode=TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT,
            centre_court_instance_id=None,
        )

    target = {
        "binding": {
            "court_instance_id": "court-a",
            "candidate_id": "candidate-a",
            "scene_from_court": (
                RigidTransform.identity().matrix().reshape(-1).tolist()
            ),
            "selection_seed": 695,
        },
        "resolution_policy": "nearest_camera",
        "camera_to_court_center_distance_m": 1.0,
        "unexpected": True,
    }
    with pytest.raises(ValueError, match="unknown"):
        ResolvedTargetCourtV2.from_mapping(target)
