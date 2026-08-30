"""Independent negative coverage for the Court final-render inventory gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.dataset.court.assembler import (
    _validate_parameter_table_diagnostic,
    _validate_render_inventory,
    _validate_renderer_visibility_payload,
    _validate_safety_diagnostic,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    DatasetSplit,
    OrbitCenterKind,
    OrbitCoverageMode,
    OrbitTargetKind,
    OrbitTargetMode,
    OrbitTrajectorySpecV4,
    OrbitViewSpecV2,
    PathConstructorV4,
    PathFamilyV4,
    PlannedCourtSample,
    RequiredTrajectoryCoverage,
    SelectedTrajectoryCoverage,
    SupportModelSummary,
    TargetCourtPolicyV2,
    TargetCourtResolutionPolicy,
    TrajectorySafetyEvaluation,
    TrajectorySemanticPhaseEvaluation,
    TrajectorySupportPolicy,
    VerticalProfileV4,
    semantic_phase_inventory_digest,
)
from src.synthetic_data_generation.dataset.court.schema import COURT_SCHEMA_V4
from src.synthetic_data_generation.dataset.court.shards import CourtRenderedSample
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _sample() -> PlannedCourtSample:
    camera = SceneCamera(
        camera_id="sample-000000",
        source_frame_index=0,
        width=4,
        height=3,
        intrinsics=(4.0, 0.0, 1.5, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="generated/sample-000000.png",
    )
    return PlannedCourtSample(
        sample_index=0,
        sample_id=camera.camera_id,
        trajectory_group_id="group-a",
        trajectory_id="trajectory-a",
        view_id="view-a",
        trajectory_frame_index=0,
        split=DatasetSplit.TRAIN,
        shard_id="shard-000",
        camera_center_scene_m=(0.0, 0.0, 0.0),
        camera=camera,
    )


def _rendered(root: Path, sample: PlannedCourtSample) -> CourtRenderedSample:
    sample_root = root / sample.sample_id
    sample_root.mkdir(parents=True)
    np.save(sample_root / "rgb.npy", np.zeros((3, 4, 3), dtype=np.float32))
    np.save(sample_root / "alpha.npy", np.ones((3, 4, 1), dtype=np.float32))
    np.save(sample_root / "depth.npy", np.ones((3, 4, 1), dtype=np.float32))
    Image.new("RGB", (4, 3)).save(sample_root / "rgb.png")
    Image.new("L", (4, 3)).save(sample_root / "alpha.png")
    return CourtRenderedSample(
        sample=sample,
        rgb_path=sample_root / "rgb.npy",
        rgb_preview_path=sample_root / "rgb.png",
        alpha_path=sample_root / "alpha.npy",
        alpha_preview_path=sample_root / "alpha.png",
        depth_path=sample_root / "depth.npy",
    )


def _inventory_plan(*samples: PlannedCourtSample) -> CourtDatasetPlan:
    """Provide the exact plan surface exercised by the private inventory gate."""
    return cast(CourtDatasetPlan, SimpleNamespace(samples=samples))


def test_render_inventory_rejects_missing_duplicate_and_overlapping_results(
    tmp_path: Path,
) -> None:
    sample = _sample()
    plan = _inventory_plan(sample)
    rendered = _rendered(tmp_path, sample)

    with pytest.raises(ValueError, match="partition mismatch.*missing"):
        _validate_render_inventory(
            plan,
            (),
            pre_render_rejected_sample_ids=(),
        )
    with pytest.raises(ValueError, match="Duplicate renderer sample ID"):
        _validate_render_inventory(
            plan,
            (rendered, rendered),
            pre_render_rejected_sample_ids=(),
        )
    with pytest.raises(ValueError, match="partition mismatch"):
        _validate_render_inventory(
            plan,
            (rendered,),
            pre_render_rejected_sample_ids=(sample.sample_id,),
        )
    with pytest.raises(ValueError, match="rejection inventory contains duplicates"):
        _validate_render_inventory(
            plan,
            (),
            pre_render_rejected_sample_ids=(sample.sample_id, sample.sample_id),
        )


def test_render_inventory_rejects_renderer_metadata_drift(tmp_path: Path) -> None:
    expected = _sample()
    plan = _inventory_plan(expected)
    changed = replace(expected, split=DatasetSplit.VALIDATION)
    rendered = _rendered(tmp_path, changed)

    with pytest.raises(ValueError, match="Renderer sample metadata changed"):
        _validate_render_inventory(
            plan,
            (rendered,),
            pre_render_rejected_sample_ids=(),
        )


def test_render_inventory_accepts_exact_renderer_or_rejection_partition(
    tmp_path: Path,
) -> None:
    rendered_sample = _sample()
    rejected_sample = replace(
        rendered_sample,
        sample_index=1,
        sample_id="sample-000001",
        camera=replace(
            rendered_sample.camera,
            camera_id="sample-000001",
            source_frame_index=1,
            image_path="generated/sample-000001.png",
        ),
    )
    plan = _inventory_plan(rendered_sample, rejected_sample)
    rendered = _rendered(tmp_path, rendered_sample)

    _validate_render_inventory(
        plan,
        (rendered,),
        pre_render_rejected_sample_ids=(rejected_sample.sample_id,),
    )


@pytest.mark.parametrize("mutated_field", ["alpha", "depth"])
def test_renderer_semantic_visibility_rejects_valid_range_array_mutation(
    mutated_field: str,
) -> None:
    projection = {
        "visible_point_count": 1,
        "visible_class_names": ["doubles_left"],
        "courts": [
            {
                "classes": [
                    {
                        "class_name": "doubles_left",
                        "renderer_visible": True,
                        "points": [
                            {
                                "uv": [1.0, 1.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            },
                            {
                                "uv": [10.0, 10.0],
                                "in_frame": False,
                                "renderer_visible": False,
                            },
                        ],
                    }
                ]
            }
        ],
    }
    alpha: NDArray[np.float32] = np.ones((3, 4, 1), dtype=np.float32)
    depth: NDArray[np.float32] = np.ones((3, 4, 1), dtype=np.float32)
    if mutated_field == "alpha":
        alpha.fill(0.0)
    else:
        depth.fill(0.0)

    with pytest.raises(ValueError, match="renderer-visible point disagrees"):
        _validate_renderer_visibility_payload(
            projection,
            alpha=alpha,
            depth=depth,
        )


def _v4_safety_payloads() -> tuple[
    dict[str, object], dict[str, object], dict[str, object]
]:
    policy = TrajectorySupportPolicy(
        decision_id="unit-safety-v1",
        support_radius_m=0.8,
        endpoint_radius_m=0.6,
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
        maximum_cycle_closure_distance_m=1.1,
        maximum_constructive_cycle_count=24,
        cycle_smoothing_distance_m=0.03,
    )
    summary = SupportModelSummary(
        input_digest="a" * 64,
        coordinate_space="metric_scene_metres",
        captured_camera_count=8,
        public_point_count=20,
        density_qualified_voxel_count=5,
        raw_inflated_occupancy_cell_count=20,
        inflated_occupancy_cell_count=15,
        camera_ball_carved_cell_count=4,
        camera_capsule_carved_cell_count=1,
        captured_camera_occupied_count=0,
        endpoint_ball_count=8,
        capsule_count=7,
        skipped_gap_link_count=0,
        skipped_obstacle_link_count=0,
        capsule_index_cell_count=10,
    )
    evaluation = TrajectorySafetyEvaluation(
        trajectory_id="trajectory-00000",
        trajectory_group_id="group-00000",
        support_input_digest=summary.input_digest,
        safe=True,
        reasons=(),
        path_point_count=8,
        closed_segment_count=8,
        swept_sample_count=8,
        violating_point_indices=(),
        violating_segment_indices=(),
        minimum_support_margin_m=0.1,
        minimum_obstacle_clearance_m=0.2,
    )
    semantic_phase = TrajectorySemanticPhaseEvaluation(
        trajectory_id="trajectory-00000",
        trajectory_group_id="group-00000",
        phase_index=0,
        phase_count=1,
        view=OrbitViewSpecV2(
            view_id="view-group-00000-semantic-phase-00",
            target_kind=OrbitTargetKind.COURT,
            target_mode=OrbitTargetMode.COURT_CENTER,
            coverage_mode=OrbitCoverageMode.FULL,
            look_at_height_m=0.0,
            hfov_degrees=75.0,
        ),
        expected_frame_count=8,
        expected_valid_frame_count=8,
        semantically_viable=True,
        rejection_counts=(),
        disposition_digest="b" * 64,
    )
    phase_inventory_digest = semantic_phase_inventory_digest((semantic_phase,))
    required_coverage = RequiredTrajectoryCoverage(
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
    selected_coverage = SelectedTrajectoryCoverage(
        total_group_count=25,
        total_frame_count=2_000,
        constructors=(
            PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE,
            PathConstructorV4.FREE_SPACE_CYCLE,
        ),
        constructor_group_counts=(
            (PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 7),
            (PathConstructorV4.FREE_SPACE_CYCLE, 18),
        ),
        constructor_frame_counts=(
            (PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE, 168),
            (PathConstructorV4.FREE_SPACE_CYCLE, 1_832),
        ),
        path_families=(
            PathFamilyV4.FREE_SPACE_CYCLE,
            PathFamilyV4.ROUNDED_RECTANGLE,
        ),
        family_group_counts=(
            (PathFamilyV4.FREE_SPACE_CYCLE, 18),
            (PathFamilyV4.ROUNDED_RECTANGLE, 7),
        ),
        family_frame_counts=(
            (PathFamilyV4.FREE_SPACE_CYCLE, 1_832),
            (PathFamilyV4.ROUNDED_RECTANGLE, 168),
        ),
        vertical_profiles=(
            VerticalProfileV4.FREE_SPACE_CYCLE,
            VerticalProfileV4.PLANAR,
            VerticalProfileV4.RAISED_PHASES,
        ),
        profile_group_counts=(
            (VerticalProfileV4.FREE_SPACE_CYCLE, 18),
            (VerticalProfileV4.PLANAR, 3),
            (VerticalProfileV4.RAISED_PHASES, 4),
        ),
        profile_frame_counts=(
            (VerticalProfileV4.FREE_SPACE_CYCLE, 1_832),
            (VerticalProfileV4.PLANAR, 72),
            (VerticalProfileV4.RAISED_PHASES, 96),
        ),
        target_modes=(OrbitTargetMode.COURT_CENTER,),
        target_group_counts=((OrbitTargetMode.COURT_CENTER, 25),),
        target_frame_counts=((OrbitTargetMode.COURT_CENTER, 2_000),),
        anchor_camera_indices=(0, 1, 2, 3, 4, 5, 6),
        anchor_camera_ids=tuple(f"camera-{index}" for index in range(7)),
        unique_anchor_count=7,
        anchored_group_count=7,
        anchored_frame_count=168,
        anchored_frame_share=0.084,
        anchored_planar_group_count=3,
        anchored_raised_group_count=4,
        anchored_required_lift_group_count=4,
    )
    group = {
        "trajectory": {"trajectory_group_id": "group-00000"},
        "safety_evaluation": evaluation.to_dict(),
        "semantic_phase_evaluation": semantic_phase.to_dict(),
    }
    dataset: dict[str, object] = {
        "trajectory_groups": [group],
        "metrics": {
            "support_input_digest": summary.input_digest,
            "selected_safety_violation_count": 0,
            "required_coverage": required_coverage.to_dict(),
            "selected_coverage": selected_coverage.to_dict(),
            "required_coverage_shortfall": [],
            "optional_candidate_coverage_shortfall": [],
            "semantic_phase_inventory_digest": phase_inventory_digest,
            "projected_semantic_valid_frame_count": 8,
            "projected_semantic_valid_fraction": 1.0,
        },
    }
    plan: dict[str, object] = {
        "support_policy": policy.to_dict(),
        "support_summary": summary.to_dict(),
        "candidate_safety_evaluations": [evaluation.to_dict()],
        "candidate_semantic_phase_evaluations": [semantic_phase.to_dict()],
        "semantic_phase_inventory_digest": phase_inventory_digest,
        "projected_semantic_valid_frame_count": 8,
        "projected_semantic_valid_fraction": 1.0,
        "required_coverage": required_coverage.to_dict(),
        "selected_coverage": selected_coverage.to_dict(),
        "required_coverage_shortfall": [],
        "optional_candidate_coverage_shortfall": [],
    }
    safety: dict[str, object] = {
        "schema": COURT_SCHEMA_V4.safety_diagnostics_schema,
        "support_policy": policy.to_dict(),
        "support_summary": summary.to_dict(),
        "candidate_safety_evaluations": [evaluation.to_dict()],
        "candidate_semantic_phase_evaluations": [semantic_phase.to_dict()],
        "semantic_phase_inventory_digest": phase_inventory_digest,
        "projected_semantic_valid_frame_count": 8,
        "projected_semantic_valid_fraction": 1.0,
        "selected_trajectory_group_ids": ["group-00000"],
        "required_coverage": required_coverage.to_dict(),
        "selected_coverage": selected_coverage.to_dict(),
        "required_coverage_shortfall": [],
        "optional_candidate_coverage_shortfall": [],
        "selected_point_violation_count": 0,
        "selected_segment_violation_count": 0,
        "zero_selected_safety_violations": True,
    }
    return safety, dataset, plan


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "selected_mismatch"])
def test_v4_safety_diagnostic_rejects_candidate_inventory_tampering(
    mutation: str,
) -> None:
    safety, dataset, plan = _v4_safety_payloads()
    if mutation == "duplicate":
        candidates = safety["candidate_safety_evaluations"]
        assert isinstance(candidates, list)
        candidates.append(candidates[0])
    elif mutation == "missing":
        safety["candidate_safety_evaluations"] = []
    else:
        groups = dataset["trajectory_groups"]
        assert isinstance(groups, list)
        group = groups[0]
        assert isinstance(group, dict)
        changed = dict(group["safety_evaluation"])
        changed["minimum_support_margin_m"] = 0.2
        group["safety_evaluation"] = changed

    with pytest.raises(
        ValueError, match="candidate inventory|selected safety evidence"
    ):
        _validate_safety_diagnostic(
            safety,
            dataset=dataset,
            trajectory_plan=plan,
            definition=COURT_SCHEMA_V4,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "digest"])
def test_v4_safety_diagnostic_rejects_semantic_phase_inventory_tampering(
    mutation: str,
) -> None:
    safety, dataset, plan = _v4_safety_payloads()
    phases = safety["candidate_semantic_phase_evaluations"]
    assert isinstance(phases, list)
    if mutation == "duplicate":
        phases.append(phases[0])
    elif mutation == "missing":
        phases.clear()
    else:
        safety["semantic_phase_inventory_digest"] = "0" * 64

    with pytest.raises(ValueError, match="candidate inventory"):
        _validate_safety_diagnostic(
            safety,
            dataset=dataset,
            trajectory_plan=plan,
            definition=COURT_SCHEMA_V4,
        )


def test_v4_parameter_table_dispatches_through_the_strict_v4_parser() -> None:
    trajectory = OrbitTrajectorySpecV4(
        trajectory_id="trajectory-00000",
        trajectory_group_id="group-00000",
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
        corner_radius_ratio=None,
        vertical_phase_offsets_m=(0.0,),
        constructor=PathConstructorV4.FREE_SPACE_CYCLE,
        control_points_local_m=(
            (0.0, 0.0, 2.0),
            (1.0, 0.0, 2.0),
            (2.0, 0.0, 2.0),
            (2.0, 1.0, 2.0),
            (2.0, 2.0, 2.0),
            (1.0, 2.0, 2.0),
            (0.0, 2.0, 2.0),
            (0.0, 1.0, 2.0),
        ),
    )
    target_policy = TargetCourtPolicyV2(
        mode=TargetCourtResolutionPolicy.NEAREST_CAMERA,
        centre_court_instance_id=None,
    )
    group = {
        "trajectory": trajectory.to_dict(),
        "views": [{"view_id": "view-a"}],
        "split": "train",
        "shard_id": "shard-000",
        "sample_count": 8,
        "target_court_policy": target_policy.to_dict(),
    }
    row = {
        **trajectory.to_dict(),
        "view_ids": ["view-a"],
        "split": "train",
        "shard_id": "shard-000",
        "sample_count_per_view": 8,
        "target_court_policy": target_policy.to_dict(),
    }

    _validate_parameter_table_diagnostic(
        {"schema": COURT_SCHEMA_V4.parameter_table_schema, "rows": [row]},
        dataset={"trajectory_groups": [group]},
        definition=COURT_SCHEMA_V4,
    )
