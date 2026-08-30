from __future__ import annotations

import pytest

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlanV2,
    CourtDatasetPlanV3,
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitCoverageObjective,
    OrbitCurveMode,
    OrbitSamplingMode,
    OrbitSamplingPolicy,
    OrbitShape,
    OrbitStableField,
    OrbitStableFieldV4,
    OrbitTargetKind,
    OrbitTargetMode,
    OrbitTrajectorySpec,
    OrbitTrajectorySpecV4,
    OrbitViewSpec,
    OrbitViewSpecV2,
    PathConstructorV4,
    PathFamilyV4,
    RequiredTrajectoryCoverage,
    ResolvedTargetCourtV2,
    SelectedTrajectoryCoverage,
    TargetCourtPolicyV2,
    TargetCourtResolutionPolicy,
    TrajectorySafetyEvaluation,
    TrajectorySafetyReason,
    TrajectorySemanticPhaseEvaluation,
    TrajectorySupportPolicy,
    VerticalProfileV4,
    required_coverage_shortfall,
    semantic_phase_inventory_digest,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
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


def test_v3_plan_has_distinct_identity_without_changing_v2_target_structure() -> None:
    assert issubclass(CourtDatasetPlanV3, CourtDatasetPlanV2)
    v2_plan = object.__new__(CourtDatasetPlanV2)
    v3_plan = object.__new__(CourtDatasetPlanV3)

    assert v2_plan.schema_version is CourtDatasetSchemaVersion.V2
    assert v3_plan.schema_version is CourtDatasetSchemaVersion.V3


def test_v4_contracts_are_strict_without_extending_legacy_vocabularies() -> None:
    trajectory = OrbitTrajectorySpecV4(
        trajectory_id="trajectory-v4",
        trajectory_group_id="group-v4",
        shape=PathFamilyV4.ROUNDED_RECTANGLE,
        center_kind=OrbitCenterKind.COMPLEX,
        center_court_instance_id=None,
        base_radius_m=20.0,
        radius_scale=1.0,
        axis_ratio=0.65,
        orientation_radians=0.0,
        base_height_m=2.0,
        vertical_amplitude_m=0.75,
        vertical_cycles=0,
        vertical_phase_radians=0.0,
        curve_mode=VerticalProfileV4.RAISED_PHASES,
        corner_radius_ratio=0.25,
        vertical_phase_offsets_m=(0.0, 0.75, 0.75, 0.0),
    )
    support = TrajectorySupportPolicy(
        decision_id="b00-support-v1",
        support_radius_m=2.5,
        endpoint_radius_m=1.5,
        maximum_camera_link_distance_m=4.0,
        maximum_source_frame_gap=5,
        occupancy_voxel_size_m=0.5,
        minimum_points_per_voxel=3,
        obstacle_inflation_m=0.5,
        camera_ball_clearance_m=0.35,
        camera_capsule_clearance_m=0.25,
        sweep_step_m=0.25,
        boundary_epsilon_m=1.0e-6,
        minimum_captured_cameras=24,
        minimum_public_points=1_000,
        maximum_capsule_index_cells=2_000_000,
        maximum_occupancy_cells=5_000_000,
        minimum_cycle_frame_span=48,
        maximum_cycle_frame_span=180,
        maximum_cycle_closure_distance_m=2.5,
        maximum_constructive_cycle_count=48,
        cycle_smoothing_distance_m=0.1,
    )

    assert OrbitTrajectorySpecV4.from_mapping(trajectory.to_dict()) == trajectory
    assert TrajectorySupportPolicy.from_mapping(support.to_dict()) == support
    assert tuple(OrbitShape) == (OrbitShape.CIRCLE, OrbitShape.ELLIPSE)
    assert tuple(OrbitCurveMode) == (
        OrbitCurveMode.PLANAR,
        OrbitCurveMode.SINUSOIDAL_HEIGHT,
    )
    assert len(tuple(OrbitStableField)) == 8
    assert len(tuple(OrbitStableFieldV4)) == 11

    unknown = trajectory.to_dict()
    unknown["legacy_fallback"] = True
    with pytest.raises(ValueError, match="unknown"):
        OrbitTrajectorySpecV4.from_mapping(unknown)


def test_v4_free_space_cycle_rejects_non_positive_control_height() -> None:
    controls = tuple((float(index), float(index % 2), 2.0) for index in range(8))
    controls = (*controls[:-1], (*controls[-1][:2], -2.0))

    with pytest.raises(ValueError, match="non-positive camera height"):
        OrbitTrajectorySpecV4(
            trajectory_id="trajectory-free-space",
            trajectory_group_id="group-free-space",
            shape=PathFamilyV4.FREE_SPACE_CYCLE,
            center_kind=OrbitCenterKind.COMPLEX,
            center_court_instance_id=None,
            base_radius_m=20.0,
            radius_scale=1.0,
            axis_ratio=1.0,
            orientation_radians=0.0,
            base_height_m=2.0,
            vertical_amplitude_m=0.0,
            vertical_cycles=0,
            vertical_phase_radians=0.0,
            curve_mode=VerticalProfileV4.FREE_SPACE_CYCLE,
            corner_radius_ratio=None,
            vertical_phase_offsets_m=(0.0,),
            control_points_local_m=controls,
            constructor=PathConstructorV4.FREE_SPACE_CYCLE,
        )


def _required_v4_coverage() -> RequiredTrajectoryCoverage:
    return RequiredTrajectoryCoverage(
        constructors=(
            PathConstructorV4.FREE_SPACE_CYCLE,
            PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE,
        ),
        path_families=(PathFamilyV4.ROUNDED_RECTANGLE,),
        vertical_profiles=(
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.RAISED_PHASES,
        ),
        target_modes=(OrbitTargetMode.COURT_CENTER,),
        minimum_total_groups=24,
        minimum_free_space_cycle_groups=12,
        minimum_anchored_rounded_rectangle_groups=6,
        minimum_unique_anchors=6,
        minimum_anchored_planar_groups=3,
        minimum_anchored_raised_groups=3,
        required_raised_lift_m=0.25,
        minimum_anchored_frame_share=0.08,
    )


def _selected_v4_coverage(
    *,
    total_groups: int = 24,
    free_groups: int = 18,
    anchored_groups: int = 6,
    unique_anchors: int = 6,
    planar_groups: int = 3,
    raised_groups: int = 3,
    anchored_frames: int = 200,
    total_frames: int = 2_000,
) -> SelectedTrajectoryCoverage:
    free_frames = total_frames - anchored_frames
    planar_frames = anchored_frames // 2
    raised_frames = anchored_frames - planar_frames
    return SelectedTrajectoryCoverage(
        total_group_count=total_groups,
        total_frame_count=total_frames,
        constructors=(
            PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE,
            PathConstructorV4.FREE_SPACE_CYCLE,
        ),
        constructor_group_counts=(
            (PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, anchored_groups),
            (PathConstructorV4.FREE_SPACE_CYCLE, free_groups),
        ),
        constructor_frame_counts=(
            (PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, anchored_frames),
            (PathConstructorV4.FREE_SPACE_CYCLE, free_frames),
        ),
        path_families=(
            PathFamilyV4.FREE_SPACE_CYCLE,
            PathFamilyV4.ROUNDED_RECTANGLE,
        ),
        family_group_counts=(
            (PathFamilyV4.FREE_SPACE_CYCLE, free_groups),
            (PathFamilyV4.ROUNDED_RECTANGLE, anchored_groups),
        ),
        family_frame_counts=(
            (PathFamilyV4.FREE_SPACE_CYCLE, free_frames),
            (PathFamilyV4.ROUNDED_RECTANGLE, anchored_frames),
        ),
        vertical_profiles=(
            VerticalProfileV4.FREE_SPACE_CYCLE,
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.RAISED_PHASES,
        ),
        profile_group_counts=(
            (VerticalProfileV4.FREE_SPACE_CYCLE, free_groups),
            (VerticalProfileV4.PLANAR, planar_groups),
            (VerticalProfileV4.RAISED_PHASES, raised_groups),
        ),
        profile_frame_counts=(
            (VerticalProfileV4.FREE_SPACE_CYCLE, free_frames),
            (VerticalProfileV4.PLANAR, planar_frames),
            (VerticalProfileV4.RAISED_PHASES, raised_frames),
        ),
        target_modes=(OrbitTargetMode.COURT_CENTER,),
        target_group_counts=((OrbitTargetMode.COURT_CENTER, total_groups),),
        target_frame_counts=((OrbitTargetMode.COURT_CENTER, total_frames),),
        anchor_camera_indices=tuple(range(unique_anchors)),
        anchor_camera_ids=tuple(
            f"camera-{index}" for index in range(unique_anchors)
        ),
        unique_anchor_count=unique_anchors,
        anchored_group_count=anchored_groups,
        anchored_frame_count=anchored_frames,
        anchored_frame_share=anchored_frames / total_frames,
        anchored_planar_group_count=planar_groups,
        anchored_raised_group_count=raised_groups,
        anchored_required_lift_group_count=raised_groups,
    )


def test_v4_required_and_selected_coverage_round_trip_without_shortfall() -> None:
    required = _required_v4_coverage()
    selected = _selected_v4_coverage()

    assert RequiredTrajectoryCoverage.from_mapping(required.to_dict()) == required
    assert SelectedTrajectoryCoverage.from_mapping(selected.to_dict()) == selected
    assert required_coverage_shortfall(required, selected) == ()


def test_v4_required_coverage_reports_every_count_anchor_and_share_shortfall() -> None:
    selected = _selected_v4_coverage(
        total_groups=23,
        free_groups=18,
        anchored_groups=5,
        unique_anchors=5,
        planar_groups=2,
        raised_groups=3,
        anchored_frames=140,
    )

    assert required_coverage_shortfall(_required_v4_coverage(), selected) == (
        "minimum_anchored_frame_share",
        "minimum_anchored_planar_groups",
        "minimum_anchored_rounded_rectangle_groups",
        "minimum_total_groups",
        "minimum_unique_anchors",
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("unique_anchor_count", 5, "anchor"),
        ("anchored_frame_share", 0.09, "anchored coverage"),
        ("anchor_camera_ids", ["camera-0"] * 6, "anchor"),
    ),
)
def test_v4_selected_coverage_rejects_tampered_aggregate_authority(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _selected_v4_coverage().to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        SelectedTrajectoryCoverage.from_mapping(payload)


def test_v4_semantic_phase_contract_round_trips_and_rejects_accounting_tampering() -> (
    None
):
    evaluation = TrajectorySemanticPhaseEvaluation(
        trajectory_id="trajectory-v4",
        trajectory_group_id="group-v4",
        phase_index=2,
        phase_count=6,
        view=OrbitViewSpecV2(
            view_id="view-group-v4-semantic-phase-02",
            target_kind=OrbitTargetKind.COURT,
            target_mode=OrbitTargetMode.COURT_CENTER,
            coverage_mode=OrbitCoverageMode.PARTIAL,
            look_at_height_m=0.0,
            hfov_degrees=35.0,
        ),
        expected_frame_count=100,
        expected_valid_frame_count=91,
        semantically_viable=True,
        rejection_counts=(("insufficient_pre_render_semantic_coverage", 9),),
        disposition_digest="c" * 64,
    )

    assert (
        TrajectorySemanticPhaseEvaluation.from_mapping(evaluation.to_dict())
        == evaluation
    )
    assert len(semantic_phase_inventory_digest((evaluation,))) == 64
    tampered = evaluation.to_dict()
    tampered["expected_valid_frame_count"] = 92
    with pytest.raises(ValueError, match="partition"):
        TrajectorySemanticPhaseEvaluation.from_mapping(tampered)

    unknown = evaluation.to_dict()
    unknown["inferred_phase"] = 2
    with pytest.raises(ValueError, match="unknown"):
        TrajectorySemanticPhaseEvaluation.from_mapping(unknown)


def test_v4_safety_evidence_rejects_unordered_and_out_of_range_diagnostics() -> None:
    evaluation = TrajectorySafetyEvaluation(
        trajectory_id="trajectory-v4",
        trajectory_group_id="group-v4",
        support_input_digest="a" * 64,
        safe=False,
        reasons=(
            TrajectorySafetyReason.POINT_OUTSIDE_SUPPORT,
            TrajectorySafetyReason.SWEPT_SEGMENT_OUTSIDE_SUPPORT,
        ),
        path_point_count=8,
        closed_segment_count=8,
        swept_sample_count=8,
        violating_point_indices=(0,),
        violating_segment_indices=(7,),
        minimum_support_margin_m=-0.1,
        minimum_obstacle_clearance_m=0.5,
    )
    assert TrajectorySafetyEvaluation.from_mapping(evaluation.to_dict()) == evaluation

    unordered = evaluation.to_dict()
    reasons = unordered["reasons"]
    assert isinstance(reasons, list)
    reasons.reverse()
    with pytest.raises(ValueError, match="ordered"):
        TrajectorySafetyEvaluation.from_mapping(unordered)

    out_of_range = evaluation.to_dict()
    out_of_range["violating_segment_indices"] = [8]
    with pytest.raises(ValueError, match="out-of-range"):
        TrajectorySafetyEvaluation.from_mapping(out_of_range)
