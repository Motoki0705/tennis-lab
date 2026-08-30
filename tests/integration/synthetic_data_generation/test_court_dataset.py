"""CPU integration of config, alignment layout, planning, sampling, and assignment."""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from numpy.typing import NDArray
from omegaconf import OmegaConf
from PIL import Image

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentPartitions,
    AlignmentResult,
    CandidateAlignment,
    MetricSceneAdapter,
    PartitionAssessment,
    PartitionMetrics,
    PartitionThresholds,
)
from src.synthetic_data_generation.configuration import (
    CourtDatasetConfiguration,
    CourtTrajectoryPolicyV4,
)
from src.synthetic_data_generation.dataset.court import assembler as court_assembler
from src.synthetic_data_generation.dataset.court.assembler import (
    CourtArrayValidationMode,
    assemble_court_dataset,
    validate_court_dataset,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling import (
    selection as court_selection,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportModel,
    build_trajectory_support_model,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    resolve_target_court,
    resolved_court_look_at_scene,
)
from src.synthetic_data_generation.dataset.court.components.camera_view import (
    camera_view_canonicalization,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlan,
    CourtDatasetPlanAny,
    CourtDatasetPlanV2,
    CourtDatasetPlanV3,
    CourtDatasetPlanV4,
    OrbitCenter,
    OrbitCenterKind,
    OrbitTrajectorySpecV4,
    PathConstructorV4,
    PathFamilyV4,
    PlannedCourtSampleV4,
    TargetCourtResolutionPolicy,
    TrajectoryGroupPlanV4,
    VerticalProfileV4,
    semantic_phase_inventory_digest,
)
from src.synthetic_data_generation.dataset.court.performance import (
    CourtPerformanceEvidence,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import (
    CourtNHTRenderer,
    validate_pre_render_plan,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SEMANTIC_CLASS_NAMES_V2,
)
from src.synthetic_data_generation.dataset.court.semantic_manifest import (
    COURT_SEMANTIC_MANIFEST_PATH,
    require_equal_court_semantic_manifests,
)
from src.synthetic_data_generation.dataset.court.semantic_pre_render import (
    CourtSemanticFrameDisposition,
    court_semantic_phase_disposition_digest,
    evaluate_court_semantic_pre_render,
)
from src.synthetic_data_generation.dataset.runtime import PerformanceTimer
from src.synthetic_data_generation.reconstruction.scene_export import (
    NHT_ALPHA_OUTPUT_SEMANTICS,
    NHT_DEPTH_OUTPUT_SEMANTICS,
    NHT_IMAGE_RESOLUTION_SEMANTICS,
    NHT_PIXEL_COORDINATE_CONVENTION,
    NHT_RGB_OUTPUT_SEMANTICS,
    NHT_SCENE_COORDINATE_CONVENTION,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderClient
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.io import load_json
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"


def test_court_domain_resolves_production_quantities_and_balanced_courts() -> None:
    layout = _layout()
    configuration = CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(
            OmegaConf.load(
                Path("src/synthetic_data_generation/configs/dataset/court/train.yaml")
            ),
            resolve=True,
        )
    )
    plan = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=_captured_cameras(),
        layout=layout,
        configuration=configuration,
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
    )
    assert isinstance(plan, CourtDatasetPlan)
    assert len(plan.groups) >= 24
    assert 2_000 <= plan.proposal_count <= 5_000
    assert max(group.maximum_adjacent_step_m for group in plan.groups) <= 1.05
    global_counts = Counter(
        group.target_court.court_instance_id for group in plan.groups
    )
    assert set(global_counts) == {court.court_instance_id for court in layout.courts}
    assert max(global_counts.values()) - min(global_counts.values()) <= 1
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for group in plan.groups:
        by_split[group.split.value][group.target_court.court_instance_id] += 1
    assert all(
        max(counts.values()) - min(counts.values()) <= 1 for counts in by_split.values()
    )
    assert all(
        group.trajectory.center_court_instance_id is None
        or group.trajectory.center_court_instance_id
        == group.target_court.court_instance_id
        for group in plan.groups
    )


def test_same_seed_public_renderer_runs_publish_equal_semantic_manifests(
    tmp_path: Path,
) -> None:
    executable = _write_fake_nht_render(tmp_path / "bin/nht-render")
    first_plan, first_manifest, first_root = _execute_court_render(
        tmp_path / "repeat-a",
        executable=executable,
        rgb_value=0.2,
        verify_attempt_local_reuse=True,
    )
    second_plan, second_manifest, second_root = _execute_court_render(
        tmp_path / "repeat-b",
        executable=executable,
        rgb_value=0.8,
    )

    assert first_plan.to_dict() == second_plan.to_dict()
    require_equal_court_semantic_manifests(first_manifest, second_manifest)
    assert first_manifest == second_manifest
    first_dataset = _json_mapping(load_json(first_root / "dataset.json"))
    second_dataset = _json_mapping(load_json(second_root / "dataset.json"))
    first_performance = CourtPerformanceEvidence.from_dict(
        load_json(first_root / "diagnostics/performance.json")
    )
    second_performance = CourtPerformanceEvidence.from_dict(
        load_json(second_root / "diagnostics/performance.json")
    )
    assert first_manifest["trajectory_groups"] == first_dataset["trajectory_groups"]
    assert first_manifest["counts"] == second_manifest["counts"]
    assert first_performance.post_render_rejected_sample_count > 0
    assert (
        first_performance.post_render_rejected_sample_count
        == second_performance.post_render_rejected_sample_count
    )
    assert first_performance.fresh_rendered_sample_count == (
        first_performance.renderable_sample_count
    )
    assert first_performance.retained_nht_array_bytes == 0
    _assert_no_operational_manifest_fields(first_manifest)

    first_record = _first_accepted_record(first_dataset)
    second_record = _first_accepted_record(second_dataset)
    first_rgb = np.load(
        _record_path(first_root, first_record, "rgb"), allow_pickle=False
    )
    second_rgb = np.load(
        _record_path(second_root, second_record, "rgb"), allow_pickle=False
    )
    assert not np.array_equal(first_rgb, second_rgb)
    _assert_repeat_semantic_mutations_fail(first_manifest)


@pytest.mark.parametrize(
    (
        "selector",
        "plan_type",
        "dataset_schema",
        "sample_schema",
        "manifest_schema",
        "diagnostic_suffix",
        "performance_schema",
    ),
    [
        (
            "v2",
            CourtDatasetPlanV2,
            "canonical_court_dataset_v2",
            "canonical_court_sample_v2",
            "court_renderer_semantic_manifest_v2",
            "v2",
            "court_dataset_performance_v3",
        ),
        (
            "v3",
            CourtDatasetPlanV3,
            "canonical_court_dataset_v3",
            "canonical_court_sample_v3",
            "court_renderer_semantic_manifest_v3",
            "v3",
            "court_dataset_performance_v4",
        ),
    ],
)
def test_singleton_public_renderer_publishes_exact_targets_labels_and_diagnostics(
    tmp_path: Path,
    selector: str,
    plan_type: type[CourtDatasetPlanV2],
    dataset_schema: str,
    sample_schema: str,
    manifest_schema: str,
    diagnostic_suffix: str,
    performance_schema: str,
) -> None:
    executable = _write_fake_nht_render(tmp_path / "bin/nht-render")
    plan, semantic_manifest, dataset_root = _execute_court_render(
        tmp_path / selector,
        executable=executable,
        rgb_value=0.4,
        court_selector=selector,
    )

    assert type(plan) is plan_type
    dataset = _json_mapping(load_json(dataset_root / "dataset.json"))
    report = validate_court_dataset(
        dataset_root,
        expected_plan=plan,
        expected_configuration=_configuration(selector),
        array_validation=CourtArrayValidationMode.HEADERS_ONLY,
    )
    assert dataset["schema"] == dataset_schema
    assert semantic_manifest["schema"] == manifest_schema
    assert semantic_manifest["sample_schema"] == sample_schema
    assert report.accepted_frame_count >= 2_000

    accepted = cast(list[object], dataset["samples"])
    rejected = cast(list[object], dataset["rejected_samples"])
    assert accepted
    group_by_id = {group.trajectory_group_id: group for group in plan.groups}
    targets_by_complex_group: dict[str, set[str]] = defaultdict(set)
    for raw_record in (*accepted, *rejected):
        record = _json_mapping(raw_record)
        target = _json_mapping(record["target_court"])
        binding = _json_mapping(target["binding"])
        group_id = cast(str, record["trajectory_group_id"])
        group = group_by_id[group_id]
        court_id = cast(str, binding["court_instance_id"])
        if (
            group.target_court_policy.mode
            is TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT
        ):
            assert court_id == group.trajectory.center_court_instance_id
        else:
            targets_by_complex_group.setdefault(group_id, set()).add(court_id)
    assert any(len(targets) > 1 for targets in targets_by_complex_group.values())

    first_record = _json_mapping(accepted[0])
    labels = _json_mapping(
        load_json(_record_path(dataset_root, first_record, "labels"))
    )
    assert labels["schema"] == sample_schema
    assert labels["target_court"] == first_record["target_court"]
    projection = _json_mapping(labels["projection"])
    projection_courts = cast(list[object], projection["courts"])
    for raw_court in projection_courts:
        court = _json_mapping(raw_court)
        classes = cast(list[object], court["classes"])
        assert [_json_mapping(value)["class_name"] for value in classes] == list(
            COURT_SEMANTIC_CLASS_NAMES_V2
        )
        serialized_physical_indices = [
            _json_mapping(cast(list[object], _json_mapping(value)["points"])[0])[
                "physical_index"
            ]
            for value in classes
        ]
        assert len(classes) == 14
        assert all(
            len(cast(list[object], _json_mapping(value)["points"])) == 1
            for value in classes
        )
        assert set(serialized_physical_indices) == set(range(14))

    if selector == "v3":
        camera = SceneCamera.from_dict(first_record["camera"])
        layout_by_id = {
            court.court_instance_id: court for court in _alignment().layout.courts
        }
        for raw_court in projection_courts:
            court = _json_mapping(raw_court)
            court_id = cast(str, court["court_instance_id"])
            classes = cast(list[object], court["classes"])
            canonical_physical_indices = tuple(
                cast(
                    int,
                    _json_mapping(
                        cast(list[object], _json_mapping(value)["points"])[0]
                    )["physical_index"],
                )
                for value in classes
            )
            assert (
                canonical_physical_indices
                == camera_view_canonicalization(
                    camera,
                    layout_by_id[court_id],
                ).semantic_to_physical
            )

    diagnostics = {
        "trajectory-plan.json": f"canonical_court_orbit_plan_{diagnostic_suffix}",
        "acceptance.json": f"court_acceptance_diagnostics_{diagnostic_suffix}",
        "splits.json": f"court_split_diagnostics_{diagnostic_suffix}",
        "parameter-table.json": f"court_parameter_table_{diagnostic_suffix}",
        "semantic-visibility.json": (
            f"court_semantic_visibility_diagnostics_{diagnostic_suffix}"
        ),
        "semantic-manifest.json": manifest_schema,
        "performance.json": performance_schema,
    }
    for filename, schema in diagnostics.items():
        payload = _json_mapping(load_json(dataset_root / "diagnostics" / filename))
        assert payload["schema"] == schema

    pre_render_rejected = [
        record
        for value in rejected
        for record in (_json_mapping(value),)
        if record["projection"] is None
    ]
    assert pre_render_rejected
    pre_render_reasons = {
        cast(list[str], record["reasons"])[0].split(":", maxsplit=1)[0]
        for record in pre_render_rejected
    }
    assert "ambiguous_camera_relative_near_far" in pre_render_reasons
    assert pre_render_reasons == {"ambiguous_camera_relative_near_far"}

    with pytest.raises(ValueError, match="configuration schemas are mixed"):
        validate_court_dataset(
            dataset_root,
            expected_configuration=_configuration("v3" if selector == "v2" else "v2"),
            array_validation=CourtArrayValidationMode.HEADERS_ONLY,
        )

    dataset_path = dataset_root / "dataset.json"
    original_dataset_text = dataset_path.read_text(encoding="utf-8")
    unknown_dataset = json.loads(original_dataset_text)
    unknown_dataset["schema"] = "canonical_court_dataset_v5"
    dataset_path.write_text(json.dumps(unknown_dataset), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="Unknown Court dataset schema"):
            validate_court_dataset(
                dataset_root,
                array_validation=CourtArrayValidationMode.HEADERS_ONLY,
            )
    finally:
        dataset_path.write_text(original_dataset_text, encoding="utf-8")

    label_path = _record_path(dataset_root, first_record, "labels")
    original_label_text = label_path.read_text(encoding="utf-8")
    mixed_label = json.loads(original_label_text)
    mixed_label["schema"] = (
        "canonical_court_sample_v3" if selector == "v2" else "canonical_court_sample_v2"
    )
    label_path.write_text(json.dumps(mixed_label), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="labels schema"):
            validate_court_dataset(
                dataset_root,
                array_validation=CourtArrayValidationMode.HEADERS_ONLY,
            )
    finally:
        label_path.write_text(original_label_text, encoding="utf-8")


def test_v4_plan_pre_render_and_assembly_share_semantic_phase_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        court_selection,
        "generate_trajectory_candidates_v4",
        _compact_analytic_candidates,
    )
    monkeypatch.setattr(
        court_selection,
        "generate_free_space_cycle_candidates",
        _compact_free_space_candidates,
    )
    original_anchored = (
        court_selection.generate_anchored_rounded_rectangle_candidates
    )
    monkeypatch.setattr(
        court_selection,
        "generate_anchored_rounded_rectangle_candidates",
        lambda **kwargs: _compact_anchored_candidates(
            original_anchored(**kwargs)
        ),
    )
    executable = _write_fake_nht_render(tmp_path / "bin/nht-render")
    plan, _semantic_manifest, dataset_root = _execute_court_render(
        tmp_path / "v4",
        executable=executable,
        rgb_value=0.4,
        court_selector="v4",
        render_cameras=_v4_render_cameras(),
        reject_first_render_per_shard=False,
        scene_points=np.repeat(
            np.asarray(((0.0, 0.0, 0.0, 0.5, 0.5, 0.5),), dtype=np.float32),
            1_000,
            axis=0,
        ),
    )

    assert isinstance(plan, CourtDatasetPlanV4)
    assert plan.projected_semantic_valid_frame_count >= 2_000
    assert plan.projected_semantic_valid_fraction >= 0.9
    assert {
        group.semantic_phase_evaluation.phase_index for group in plan.groups
    } == set(range(6))
    dataset = _json_mapping(load_json(dataset_root / "dataset.json"))
    metrics = _json_mapping(dataset["metrics"])
    assert metrics["semantic_phase_inventory_digest"] == (
        plan.semantic_phase_inventory_digest
    )
    safety = _json_mapping(
        load_json(dataset_root / "diagnostics/trajectory-safety.json")
    )
    assert safety["candidate_semantic_phase_evaluations"] == [
        item.to_dict() for item in plan.candidate_semantic_phase_evaluations
    ]
    assert safety["projected_semantic_valid_frame_count"] == (
        plan.projected_semantic_valid_frame_count
    )
    validate_court_dataset(
        dataset_root,
        expected_plan=plan,
        expected_configuration=_configuration("v4"),
        array_validation=CourtArrayValidationMode.HEADERS_ONLY,
    )


def test_v4_pre_render_rejects_sample_camera_outside_selected_safe_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alignment, configuration, plan, support_model = (
        _build_v4_pre_render_binding_fixture(monkeypatch)
    )

    selected_group = plan.groups[0]
    selected_samples = tuple(
        sample
        for sample in plan.samples
        if sample.trajectory_group_id == selected_group.trajectory_group_id
    )
    replacement_sample = None
    for sample in selected_samples:
        center = np.asarray(sample.camera_center_scene_m, dtype=np.float64)
        horizontal = center.copy()
        horizontal[2] = 0.0
        horizontal /= np.linalg.norm(horizontal)
        for outward_offset_m in (3.0, 4.0, 5.0, 6.0):
            unsafe_center = center + horizontal * outward_offset_m
            target_court = resolve_target_court(
                policy=selected_group.target_court_policy,
                camera_center_scene_m=unsafe_center,
                layout=alignment.layout,
                selection_seed=sample.target_court.binding.selection_seed,
            )
            target_scene = resolved_court_look_at_scene(
                target_court=target_court,
                layout=alignment.layout,
                look_at_height_m=selected_group.views[0].look_at_height_m,
            )
            camera = replace(
                sample.camera,
                camera_to_scene=RigidTransform.from_matrix(
                    _look_at_matrix(unsafe_center, target_scene)
                ),
            )
            decision = evaluate_court_semantic_pre_render(
                camera,
                alignment.layout,
                schema_version=configuration.schema_version,
            )
            _margin, _clearance, supported, _occupied = (
                support_model.evaluate_point(unsafe_center)
            )
            if decision.accepted and not supported:
                replacement_sample = replace(
                    sample,
                    camera_center_scene_m=(
                        float(unsafe_center[0]),
                        float(unsafe_center[1]),
                        float(unsafe_center[2]),
                    ),
                    camera=camera,
                    target_court=target_court,
                )
                break
        if replacement_sample is not None:
            break
    assert replacement_sample is not None
    tampered_plan = _replace_v4_sample_and_recompute_semantic_phase(
        plan,
        replacement_sample=replacement_sample,
        alignment=alignment,
        configuration=configuration,
    )

    with pytest.raises(ValueError, match="safety"):
        validate_pre_render_plan(
            tampered_plan,
            alignment=alignment,
            support_model=support_model,
        )


def test_v4_pre_render_binds_canonical_frames_centres_and_contract_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alignment, configuration, plan, support_model = (
        _build_v4_pre_render_binding_fixture(monkeypatch)
    )
    valid_evaluation = validate_pre_render_plan(
        plan,
        alignment=alignment,
        support_model=support_model,
    )
    assert valid_evaluation.trajectory_safety_evaluations == tuple(
        group.safety_evaluation for group in plan.groups
    )

    anchored_group = next(
        group
        for group in plan.groups
        if group.trajectory.constructor
        is PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
    )
    anchor = anchored_group.trajectory.anchor_provenance
    assert anchor is not None
    tampered_groups: list[TrajectoryGroupPlanV4] = []
    for group in plan.groups:
        if (
            group.trajectory.constructor
            is not PathConstructorV4.ANCHORED_ROUNDED_RECTANGLE
        ):
            tampered_groups.append(group)
            continue
        group_anchor = group.trajectory.anchor_provenance
        assert group_anchor is not None
        tampered_groups.append(
            replace(
                group,
                trajectory=replace(
                    group.trajectory,
                    anchor_provenance=replace(
                        group_anchor,
                        camera_inventory_digest="0" * 64,
                    ),
                ),
            )
        )
    tampered_plan = replace(plan, groups=tuple(tampered_groups))
    with pytest.raises(ValueError, match="camera inventory"):
        validate_pre_render_plan(
            tampered_plan,
            alignment=alignment,
            support_model=support_model,
        )

    selected_group = plan.groups[0]
    selected_samples = tuple(
        sample
        for sample in plan.samples
        if sample.trajectory_group_id == selected_group.trajectory_group_id
    )
    assert len(selected_samples) >= 2
    first_sample, second_sample = selected_samples[:2]

    duplicate_frame = replace(
        second_sample,
        trajectory_frame_index=first_sample.trajectory_frame_index,
    )
    duplicate_frame_samples = tuple(
        duplicate_frame if sample.sample_id == second_sample.sample_id else sample
        for sample in plan.samples
    )
    with pytest.raises(ValueError, match="camera-centre path|complete camera path"):
        replace(plan, samples=duplicate_frame_samples)

    contract_mismatch_m = np.asarray((2.0e-9, 0.0, 0.0), dtype=np.float64)
    first_center = np.asarray(
        first_sample.camera_center_scene_m,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="disagrees with camera_to_scene"):
        replace(
            first_sample,
            camera_center_scene_m=tuple(first_center + contract_mismatch_m),
        )
    mismatched_matrix = first_sample.camera.camera_to_scene.matrix()
    mismatched_matrix[:3, 3] += contract_mismatch_m
    with pytest.raises(ValueError, match="disagrees with camera_to_scene"):
        replace(
            first_sample,
            camera=replace(
                first_sample.camera,
                camera_to_scene=RigidTransform.from_matrix(mismatched_matrix),
            ),
        )

    supported_off_path_sample = None
    for sample in selected_samples:
        sample_center = np.asarray(sample.camera_center_scene_m, dtype=np.float64)
        for direction in (
            np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
            np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
            np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        ):
            candidate_center = sample_center + direction * 1.0e-4
            _margin, _clearance, supported, occupied = support_model.evaluate_point(
                candidate_center
            )
            if supported and not occupied:
                supported_off_path_sample = _v4_sample_at_center(
                    sample,
                    group=selected_group,
                    center_scene_m=candidate_center,
                    alignment=alignment,
                )
                break
        if supported_off_path_sample is not None:
            break
    assert supported_off_path_sample is not None
    supported_off_path_plan = _replace_v4_sample_and_recompute_semantic_phase(
        plan,
        replacement_sample=supported_off_path_sample,
        alignment=alignment,
        configuration=configuration,
    )
    with pytest.raises(ValueError, match="safety/path binding"):
        validate_pre_render_plan(
            supported_off_path_plan,
            alignment=alignment,
            support_model=support_model,
        )

    below_tolerance_sample = _v4_sample_at_center(
        first_sample,
        group=selected_group,
        center_scene_m=first_center
        + np.asarray((5.0e-10, 0.0, 0.0), dtype=np.float64),
        alignment=alignment,
    )
    below_tolerance_plan = _replace_v4_sample_and_recompute_semantic_phase(
        plan,
        replacement_sample=below_tolerance_sample,
        alignment=alignment,
        configuration=configuration,
    )
    validate_pre_render_plan(
        below_tolerance_plan,
        alignment=alignment,
        support_model=support_model,
    )

    above_tolerance_sample = _v4_sample_at_center(
        first_sample,
        group=selected_group,
        center_scene_m=first_center
        + np.asarray((2.0e-9, 0.0, 0.0), dtype=np.float64),
        alignment=alignment,
    )
    above_tolerance_plan = _replace_v4_sample_and_recompute_semantic_phase(
        plan,
        replacement_sample=above_tolerance_sample,
        alignment=alignment,
        configuration=configuration,
    )
    with pytest.raises(ValueError, match="safety/path binding"):
        validate_pre_render_plan(
            above_tolerance_plan,
            alignment=alignment,
            support_model=support_model,
        )


def _layout() -> MultiCourtLayout:
    courts = []
    for index, x in enumerate((-8.0, 8.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        scene_from_court = RigidTransform.from_matrix(matrix)
        courts.append(
            CourtInstance(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit_status="accepted",
                fit_metrics={"rms_error_m": 0.01},
                holdout_status="accepted",
                holdout_metrics={"rms_error_m": 0.02},
            )
        )
    return MultiCourtLayout(
        courts=tuple(courts),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id="court-0",
    )


def _captured_cameras() -> tuple[SceneCamera, ...]:
    result = []
    for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            24.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0 + 2.0 * math.sin(angle),
        )
        result.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=64,
                height=48,
                intrinsics=(
                    100.0,
                    0.0,
                    31.5,
                    0.0,
                    100.0,
                    23.5,
                    0.0,
                    0.0,
                    1.0,
                ),
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(result)


def _configuration(selector: str = "train") -> CourtDatasetConfiguration:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        config = compose(
            config_name="run_scene_pipeline",
            overrides=[f"dataset/court={selector}"],
        )
    return CourtDatasetConfiguration.from_mapping(
        OmegaConf.to_container(config.dataset.court, resolve=True)
    )


def _execute_court_render(
    workspace: Path,
    *,
    executable: Path,
    rgb_value: float,
    verify_attempt_local_reuse: bool = False,
    court_selector: str = "train",
    render_cameras: tuple[SceneCamera, ...] | None = None,
    scene_points: np.ndarray | None = None,
    reject_first_render_per_shard: bool = True,
) -> tuple[CourtDatasetPlanAny, dict[str, object], Path]:
    alignment = _alignment()
    scene_path = _write_standard_scene(
        workspace,
        _render_cameras() if render_cameras is None else render_cameras,
        points_scene=scene_points,
    )
    renderer = CourtNHTRenderer(
        executable=executable,
        client=NHTRenderClient(),
        environment={
            "FAKE_NHT_REJECT_FIRST": str(int(reject_first_render_per_shard)),
            "FAKE_NHT_RGB_VALUE": str(rgb_value),
        },
        timeout_seconds=180.0,
    )
    scene = renderer.preflight(scene_path)
    configuration = _configuration(court_selector)
    plan = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=scene.cameras,
        layout=alignment.layout,
        configuration=configuration,
        metric_adapter=alignment.metric_adapter,
        points_scene=(
            scene.points_scene if configuration.schema_version.value == "v4" else None
        ),
    )
    dataset_root = workspace / "datasets/court"
    dataset_root.mkdir(parents=True)
    attempt_root = dataset_root / "_attempt"
    timer = PerformanceTimer()
    result = renderer.render(
        plan=plan,
        scene=scene,
        attempt_root=attempt_root,
        attempt_token="repeat-attempt",
        alignment=alignment,
    )
    reused_result = None
    if verify_attempt_local_reuse:
        reused_result = renderer.render(
            plan=plan,
            scene=scene,
            attempt_root=attempt_root,
            attempt_token="repeat-attempt",
            alignment=alignment,
        )
        assert reused_result.samples == result.samples
        assert reused_result.nht_invocations == 0
        assert reused_result.nht_complete_array_scans == 0
        assert reused_result.retained_nht_array_bytes == 0
    report = assemble_court_dataset(
        dataset_root,
        plan=plan,
        layout=alignment.layout,
        metric_adapter=alignment.metric_adapter,
        render_result=result,
        configuration=configuration,
        attempt_root=attempt_root,
        performance_timer=timer,
    )
    if reused_result is not None:
        reused_evidence_root = workspace / "reused-performance-evidence"
        (reused_evidence_root / "diagnostics").mkdir(parents=True)
        reused_evidence = court_assembler._write_performance_evidence(
            reused_evidence_root,
            timer=PerformanceTimer(),
            render_result=reused_result,
            proposal_count=report.proposal_count,
            accepted_frame_count=report.accepted_frame_count,
            rejected_frame_count=report.rejected_frame_count,
            accepted_staged_complete_array_scans=report.accepted_frame_count,
            post_render_rejected_staged_complete_array_scans=(
                report.performance.post_render_rejected_sample_count
            ),
            budget=configuration.performance,
            visible_by_class=report.performance.visible_points_by_class,
        )
        assert reused_evidence.fresh_run_complete_array_scan_requirement == (
            report.performance.fresh_run_complete_array_scan_requirement
        )
        assert reused_evidence.complete_array_scan_budget_capacity == (
            report.performance.complete_array_scan_budget_capacity
        )
        assert reused_evidence.metrics.complete_array_scans < (
            report.performance.metrics.complete_array_scans
        )
    manifest = _json_mapping(load_json(dataset_root / COURT_SEMANTIC_MANIFEST_PATH))
    return plan, manifest, dataset_root


def _alignment() -> AlignmentResult:
    thresholds = PartitionThresholds(
        minimum_camera_count=1,
        minimum_correspondence_count=3,
        inlier_distance_m=0.01,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.01,
        maximum_q95_error_m=0.01,
    )
    policy = AlignmentAcceptancePolicy(fit=thresholds, holdout=thresholds)
    fit = PartitionAssessment.evaluate(_partition_metrics("captured-0"), thresholds)
    holdout = PartitionAssessment.evaluate(_partition_metrics("captured-1"), thresholds)
    candidates = []
    for index, x in enumerate((-8.0, 8.0)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 3] = x
        scene_from_court = RigidTransform.from_matrix(matrix)
        candidates.append(
            CandidateAlignment(
                court_instance_id=f"court-{index}",
                candidate_id=f"candidate-{index}",
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit=fit,
                holdout=holdout,
            )
        )
    candidate_tuple = tuple(candidates)
    return AlignmentResult(
        partitions=AlignmentPartitions(
            fit_camera_ids=("captured-0",),
            holdout_camera_ids=("captured-1",),
        ),
        policy=policy,
        candidates=candidate_tuple,
        layout=MultiCourtLayout(
            courts=tuple(
                candidate.to_court_instance() for candidate in candidate_tuple
            ),
            complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
            primary_court_instance_id="court-0",
        ),
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
    )


def _partition_metrics(camera_id: str) -> PartitionMetrics:
    return PartitionMetrics(
        camera_ids=(camera_id,),
        correspondence_count=3,
        inlier_count=3,
        inlier_fraction=1.0,
        rms_error_m=0.0,
        q95_error_m=0.0,
        maximum_error_m=0.0,
    )


def _render_cameras() -> tuple[SceneCamera, ...]:
    cameras = []
    for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            24.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0 + 2.0 * math.sin(angle),
        )
        cameras.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=16,
                height=12,
                intrinsics=(25.0, 0.0, 7.5, 0.0, 25.0, 5.5, 0.0, 0.0, 1.0),
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(cameras)


def _v4_render_cameras() -> tuple[SceneCamera, ...]:
    cameras = []
    for index, angle in enumerate(np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            30.0 * math.cos(angle),
            30.0 * math.sin(angle),
            6.0,
        )
        cameras.append(
            SceneCamera(
                camera_id=f"captured-{index}",
                source_frame_index=index,
                width=16,
                height=12,
                intrinsics=(25.0, 0.0, 7.5, 0.0, 25.0, 5.5, 0.0, 0.0, 1.0),
                camera_to_scene=RigidTransform.from_matrix(matrix),
                image_path=f"images/{index}.png",
            )
        )
    return tuple(cameras)


def _compact_free_space_candidates(
    *,
    support_model: TrajectorySupportModel,
    centers: tuple[OrbitCenter, ...],
    policy: CourtTrajectoryPolicyV4,
) -> tuple[OrbitTrajectorySpecV4, ...]:
    complex_center = next(
        center for center in centers if center.center_kind is OrbitCenterKind.COMPLEX
    )
    base_height = policy.base_heights_m[0]
    local = complex_center.scene_from_center.inverse().apply(
        support_model.captured_camera_centers_m
    )
    local[:, 2] -= base_height
    result: list[OrbitTrajectorySpecV4] = []
    for index in range(12):
        controls = np.roll(local, -index, axis=0)
        if index % 2:
            controls = controls[::-1]
        result.append(
            OrbitTrajectorySpecV4(
                trajectory_id="pending",
                trajectory_group_id="pending",
                shape=PathFamilyV4.FREE_SPACE_CYCLE,
                center_kind=complex_center.center_kind,
                center_court_instance_id=complex_center.court_instance_id,
                base_radius_m=complex_center.base_radius_m,
                radius_scale=1.0,
                axis_ratio=1.0,
                orientation_radians=0.0,
                base_height_m=base_height,
                vertical_amplitude_m=0.0,
                vertical_cycles=0,
                vertical_phase_radians=0.0,
                curve_mode=VerticalProfileV4.FREE_SPACE_CYCLE,
                constructor=PathConstructorV4.FREE_SPACE_CYCLE,
                corner_radius_ratio=None,
                vertical_phase_offsets_m=(0.0,),
                control_points_local_m=tuple(
                    (float(point[0]), float(point[1]), float(point[2]))
                    for point in controls
                ),
            )
        )
    return tuple(result)


def _compact_analytic_candidates(
    policy: CourtTrajectoryPolicyV4,
    centers: tuple[OrbitCenter, ...],
    **_kwargs: object,
) -> tuple[OrbitTrajectorySpecV4, ...]:
    del policy
    complex_center = next(
        center for center in centers if center.center_kind is OrbitCenterKind.COMPLEX
    )
    return tuple(
        OrbitTrajectorySpecV4(
            trajectory_id=f"raw-trajectory-{index:05d}",
            trajectory_group_id=f"raw-group-{index:05d}",
            shape=PathFamilyV4.CIRCLE,
            center_kind=OrbitCenterKind.COMPLEX,
            center_court_instance_id=None,
            base_radius_m=complex_center.base_radius_m,
            radius_scale=1.0,
            axis_ratio=1.0,
            orientation_radians=index / 100.0,
            base_height_m=6.0,
            vertical_amplitude_m=0.0,
            vertical_cycles=0,
            vertical_phase_radians=0.0,
            curve_mode=VerticalProfileV4.PLANAR,
            corner_radius_ratio=None,
            vertical_phase_offsets_m=(0.0,),
        )
        for index in range(12)
    )


def _compact_anchored_candidates(
    candidates: tuple[OrbitTrajectorySpecV4, ...],
) -> tuple[OrbitTrajectorySpecV4, ...]:
    by_anchor: dict[int, dict[VerticalProfileV4, OrbitTrajectorySpecV4]] = (
        defaultdict(dict)
    )
    for candidate in candidates:
        provenance = candidate.anchor_provenance
        assert provenance is not None
        assert isinstance(candidate.curve_mode, VerticalProfileV4)
        by_anchor[provenance.ordered_camera_index][candidate.curve_mode] = candidate
    result: list[OrbitTrajectorySpecV4] = []
    for position, anchor_index in enumerate(sorted(by_anchor)[:10]):
        profile = (
            VerticalProfileV4.PLANAR
            if position < 5
            else VerticalProfileV4.RAISED_PHASES
        )
        result.append(by_anchor[anchor_index][profile])
    assert len(result) == 10
    return tuple(result)


def _look_at_matrix(
    center_scene: NDArray[np.float64],
    target_scene: NDArray[np.float64],
) -> NDArray[np.float64]:
    forward = np.asarray(target_scene - center_scene, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_scene
    return matrix


def _build_v4_pre_render_binding_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    AlignmentResult,
    CourtDatasetConfiguration,
    CourtDatasetPlanV4,
    TrajectorySupportModel,
]:
    monkeypatch.setattr(
        court_selection,
        "generate_trajectory_candidates_v4",
        _compact_analytic_candidates,
    )
    monkeypatch.setattr(
        court_selection,
        "generate_free_space_cycle_candidates",
        _compact_free_space_candidates,
    )
    original_anchored = (
        court_selection.generate_anchored_rounded_rectangle_candidates
    )
    monkeypatch.setattr(
        court_selection,
        "generate_anchored_rounded_rectangle_candidates",
        lambda **kwargs: _compact_anchored_candidates(
            original_anchored(**kwargs)
        ),
    )
    alignment = _alignment()
    configuration = _configuration("v4")
    cameras = _v4_render_cameras()
    points_scene = np.repeat(
        np.asarray(((0.0, 0.0, 0.0, 0.5, 0.5, 0.5),), dtype=np.float32),
        1_000,
        axis=0,
    )
    plan = build_court_dataset_plan(
        scene_id="B00",
        profile="train",
        cameras=cameras,
        layout=alignment.layout,
        configuration=configuration,
        metric_adapter=alignment.metric_adapter,
        points_scene=points_scene,
    )
    assert isinstance(plan, CourtDatasetPlanV4)
    support_model = build_trajectory_support_model(
        cameras=cameras,
        points_scene_m=points_scene,
        policy=plan.support_policy,
    )
    return alignment, configuration, plan, support_model


def _v4_sample_at_center(
    sample: PlannedCourtSampleV4,
    *,
    group: TrajectoryGroupPlanV4,
    center_scene_m: NDArray[np.float64],
    alignment: AlignmentResult,
) -> PlannedCourtSampleV4:
    target_court = resolve_target_court(
        policy=group.target_court_policy,
        camera_center_scene_m=center_scene_m,
        layout=alignment.layout,
        selection_seed=sample.target_court.binding.selection_seed,
    )
    target_scene = resolved_court_look_at_scene(
        target_court=target_court,
        layout=alignment.layout,
        look_at_height_m=group.views[0].look_at_height_m,
    )
    camera = replace(
        sample.camera,
        camera_to_scene=RigidTransform.from_matrix(
            _look_at_matrix(center_scene_m, target_scene)
        ),
    )
    return replace(
        sample,
        camera_center_scene_m=(
            float(center_scene_m[0]),
            float(center_scene_m[1]),
            float(center_scene_m[2]),
        ),
        camera=camera,
        target_court=target_court,
    )


def _replace_v4_sample_and_recompute_semantic_phase(
    plan: CourtDatasetPlanV4,
    *,
    replacement_sample: PlannedCourtSampleV4,
    alignment: AlignmentResult,
    configuration: CourtDatasetConfiguration,
) -> CourtDatasetPlanV4:
    selected_group = next(
        group
        for group in plan.groups
        if group.trajectory_group_id == replacement_sample.trajectory_group_id
    )
    samples = tuple(
        replacement_sample if sample.sample_id == replacement_sample.sample_id else sample
        for sample in plan.samples
    )
    dispositions: list[CourtSemanticFrameDisposition] = []
    rejection_counts: Counter[str] = Counter()
    valid_count = 0
    for sample in samples:
        if sample.trajectory_group_id != selected_group.trajectory_group_id:
            continue
        decision = evaluate_court_semantic_pre_render(
            sample.camera,
            alignment.layout,
            schema_version=configuration.schema_version,
        )
        valid_count += int(decision.accepted)
        rejection_counts.update(decision.rejection_reasons)
        dispositions.append(
            CourtSemanticFrameDisposition(
                trajectory_frame_index=sample.trajectory_frame_index,
                camera=sample.camera,
                decision=decision,
            )
        )
    original_phase = selected_group.semantic_phase_evaluation
    changed_phase = replace(
        original_phase,
        expected_valid_frame_count=valid_count,
        semantically_viable=valid_count > 0,
        rejection_counts=tuple(sorted(rejection_counts.items())),
        disposition_digest=court_semantic_phase_disposition_digest(
            dispositions,
            schema_version=configuration.schema_version,
            trajectory_group_id=selected_group.trajectory_group_id,
            phase_index=original_phase.phase_index,
            phase_count=original_phase.phase_count,
        ),
    )
    groups = tuple(
        replace(group, semantic_phase_evaluation=changed_phase)
        if group.trajectory_group_id == selected_group.trajectory_group_id
        else group
        for group in plan.groups
    )
    candidate_phases = tuple(
        changed_phase if phase == original_phase else phase
        for phase in plan.candidate_semantic_phase_evaluations
    )
    samples = tuple(
        replace(
            sample,
            semantic_phase_disposition_digest=changed_phase.disposition_digest,
        )
        if sample.trajectory_group_id == selected_group.trajectory_group_id
        else sample
        for sample in samples
    )
    return replace(
        plan,
        groups=groups,
        samples=samples,
        candidate_semantic_phase_evaluations=candidate_phases,
        semantic_phase_inventory_digest=semantic_phase_inventory_digest(
            candidate_phases
        ),
    )


def _write_standard_scene(
    workspace: Path,
    cameras: tuple[SceneCamera, ...],
    *,
    points_scene: np.ndarray | None = None,
) -> Path:
    export_root = workspace / "reconstruction/export"
    image_root = export_root / "images"
    model_root = export_root / "model/ckpts"
    image_root.mkdir(parents=True)
    model_root.mkdir(parents=True)
    camera_records = []
    for camera in cameras:
        image_name = f"{camera.camera_id}.png"
        Image.new("RGB", (camera.width, camera.height)).save(image_root / image_name)
        intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).reshape(3, 3)
        camera_records.append(
            {
                "camera_id": camera.camera_id,
                "source_frame_index": camera.source_frame_index,
                "time_seconds": float(camera.source_frame_index),
                "split": "train",
                "image": f"images/{image_name}",
                "width": camera.width,
                "height": camera.height,
                "intrinsics": {
                    "model": "PINHOLE",
                    "distortion_model": "NONE",
                    "params": [
                        float(intrinsics[0, 0]),
                        float(intrinsics[1, 1]),
                        float(intrinsics[0, 2]),
                        float(intrinsics[1, 2]),
                    ],
                    "matrix": intrinsics.tolist(),
                },
                "camera_to_scene": camera.camera_to_scene.matrix().tolist(),
                "source_image_processing": {
                    "source_resolution": [camera.width, camera.height],
                    "crop_xywh": [0, 0, camera.width, camera.height],
                    "undistorted": True,
                    "data_factor": 1,
                },
                "diagnostics": {
                    "sfm_camera_id": camera.source_frame_index + 1,
                    "sfm_camera_to_world": camera.camera_to_scene.matrix().tolist(),
                },
                "group": "default",
            }
        )
    runtime_config = {
        "schema": "nht_runtime_config_v1",
        "camera_model": "pinhole",
        "pose_opt": False,
        "primitive_type": "3dgs",
        "antialiased": False,
        "packed": False,
        "tile_size": 16,
        "with_ut": True,
        "with_eval3d": True,
        "near_plane": 0.01,
        "far_plane": 100.0,
        "deferred_opt_feature_dim": 48,
        "deferred_opt_enable_view_encoding": True,
        "deferred_opt_view_encoding_type": "sh",
        "deferred_mlp_hidden_dim": 128,
        "deferred_mlp_num_layers": 3,
        "deferred_opt_sh_degree": 3,
        "deferred_opt_sh_scale": 3.0,
        "deferred_opt_fourier_num_freqs": 4,
        "deferred_opt_center_ray_encoding": False,
        "deferred_decode_activation": "sigmoid",
        "post_processing": None,
    }
    (export_root / "model/runtime-config.json").write_text(
        json.dumps(runtime_config), encoding="utf-8"
    )
    (model_root / "model.pt").write_bytes(b"fake-public-model")
    points = (
        np.asarray([[0.0, 0.0, 0.0, 1.0, 0.5, 0.0]], dtype=np.float32)
        if points_scene is None
        else np.asarray(points_scene, dtype=np.float32)
    )
    np.save(export_root / "points_scene.npy", points)
    identity = np.eye(4, dtype=np.float64).tolist()
    (export_root / "cameras.json").write_text(
        json.dumps(
            {
                "schema": "nht_standard_cameras_v1",
                "camera_coordinate_convention": "x-right, y-down, z-forward",
                "transform_semantics": (
                    "camera_to_scene maps homogeneous camera coordinates to scene coordinates"
                ),
                "cameras": camera_records,
            }
        ),
        encoding="utf-8",
    )
    scene = {
        "schema": "nht_standard_scene_v1",
        "scene_id": "B00",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "scene_coordinate_convention": NHT_SCENE_COORDINATE_CONVENTION,
        "pixel_coordinate_convention": NHT_PIXEL_COORDINATE_CONVENTION,
        "image_resolution_semantics": NHT_IMAGE_RESOLUTION_SEMANTICS,
        "camera_count": len(cameras),
        "cameras": "cameras.json",
        "point_cloud": {
            "path": "points_scene.npy",
            "shape": list(points.shape),
            "dtype": "float32",
            "columns": ["x", "y", "z", "red", "green", "blue"],
            "color_range": [0.0, 1.0],
        },
        "image_root": "images",
        "model_root": "model",
        "scene_from_sfm": identity,
        "sfm_from_scene": identity,
        "normalization": {
            "applied": True,
            "camera_similarity": identity,
            "principal_axis_alignment": identity,
            "upside_down_correction": identity,
        },
        "renderer": {
            "command": "nht-render",
            "model": "model",
            "runtime_config": "model/runtime-config.json",
            "checkpoint": "model/ckpts/model.pt",
            "outputs": {
                "rgb": NHT_RGB_OUTPUT_SEMANTICS,
                "alpha": NHT_ALPHA_OUTPUT_SEMANTICS,
                "depth": NHT_DEPTH_OUTPUT_SEMANTICS,
            },
        },
        "sfm_summary": {},
        "nht_training_summary": {},
        "capabilities": ["nht_rendering_model"],
    }
    scene_path = export_root / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    return scene_path


def _write_fake_nht_render(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    interpreter = Path(sys.executable)
    path.write_text(
        f"""#!{interpreter}
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--cameras", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
request = json.loads(Path(args.cameras).read_text(encoding="utf-8"))
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
rgb_value = float(os.environ["FAKE_NHT_RGB_VALUE"])
reject_first = os.environ.get("FAKE_NHT_REJECT_FIRST") == "1"
previews = {{}}
records = []
for camera_index, camera in enumerate(request["cameras"]):
    camera_id = camera["camera_id"]
    width = camera["width"]
    height = camera["height"]
    frame = output / camera_id
    frame.mkdir()
    np.save(frame / "rgb.npy", np.full((height, width, 3), rgb_value, dtype=np.float32))
    alpha_value = 0.0 if reject_first and camera_index == 0 else 1.0
    np.save(
        frame / "alpha.npy",
        np.full((height, width, 1), alpha_value, dtype=np.float32),
    )
    np.save(frame / "depth.npy", np.ones((height, width, 1), dtype=np.float32))
    key = (width, height)
    if key not in previews:
        rgb_buffer = io.BytesIO()
        alpha_buffer = io.BytesIO()
        Image.new("RGB", key).save(rgb_buffer, format="PNG")
        Image.new("L", key, color=255).save(alpha_buffer, format="PNG")
        previews[key] = (rgb_buffer.getvalue(), alpha_buffer.getvalue())
    rgb_preview, alpha_preview = previews[key]
    (frame / "rgb.png").write_bytes(rgb_preview)
    (frame / "alpha.png").write_bytes(alpha_preview)
    records.append({{
        "camera_id": camera_id,
        "request_source": "arbitrary",
        "width": width,
        "height": height,
        "rgb": f"{{camera_id}}/rgb.npy",
        "rgb_preview": f"{{camera_id}}/rgb.png",
        "alpha": f"{{camera_id}}/alpha.npy",
        "alpha_preview": f"{{camera_id}}/alpha.png",
        "depth": f"{{camera_id}}/depth.npy",
    }})
(output / "render.json").write_text(json.dumps({{
    "schema": "nht_render_result_v1",
    "scene_schema": "nht_standard_scene_v1",
    "scene_id": scene["scene_id"],
    "coordinate_space": "canonical NHT scene space",
    "export_validation": {{}},
    "renders": records,
}}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path.resolve()


def _assert_repeat_semantic_mutations_fail(manifest: dict[str, object]) -> None:
    def mutate_sample_id(payload: dict[str, object]) -> None:
        _manifest_sample(payload)["sample_id"] = "sample-mutated"

    def mutate_split(payload: dict[str, object]) -> None:
        _manifest_sample(payload)["split"] = "validation"

    def mutate_camera_pose(payload: dict[str, object]) -> None:
        camera = _json_mapping(_manifest_sample(payload)["camera"])
        transform = cast(list[object], camera["camera_to_scene"])
        transform[3] = cast(float, transform[3]) + 1.0

    def mutate_disposition(payload: dict[str, object]) -> None:
        _manifest_sample(payload)["disposition"] = "rejected"

    def mutate_class_visibility(payload: dict[str, object]) -> None:
        projection = _json_mapping(_manifest_sample(payload)["semantic_projection"])
        courts = cast(list[object], projection["courts"])
        court = _json_mapping(courts[0])
        counts = _json_mapping(court["renderer_visible_points_by_class"])
        class_name = next(iter(counts))
        counts[class_name] = cast(int, counts[class_name]) + 1

    for mutation in (
        mutate_sample_id,
        mutate_split,
        mutate_camera_pose,
        mutate_disposition,
        mutate_class_visibility,
    ):
        mutated = copy.deepcopy(manifest)
        mutation(mutated)
        with pytest.raises(ValueError, match="not exactly equal"):
            require_equal_court_semantic_manifests(manifest, mutated)


def _manifest_sample(manifest: dict[str, object]) -> dict[str, object]:
    samples = cast(list[object], manifest["samples"])
    for sample in samples:
        record = _json_mapping(sample)
        if record.get("disposition") == "accepted":
            return record
    raise AssertionError("Court semantic manifest has no accepted sample.")


def _first_accepted_record(dataset: dict[str, object]) -> dict[str, object]:
    samples = cast(list[object], dataset["samples"])
    if not samples:
        raise AssertionError("Court integration fixture produced no accepted samples.")
    return _json_mapping(samples[0])


def _record_path(root: Path, record: dict[str, object], field: str) -> Path:
    value = record[field]
    if not isinstance(value, str):
        raise TypeError(f"Court record {field} must be a path string.")
    return root / value


def _json_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError("Expected a string-keyed JSON object.")
    return value


def _assert_no_operational_manifest_fields(value: object) -> None:
    forbidden = {
        "directory",
        "rgb",
        "rgb_preview",
        "alpha",
        "alpha_preview",
        "depth",
        "labels",
        "image_path",
        "wall_seconds",
        "generated_bytes",
        "published_bytes",
    }
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for item in value.values():
            _assert_no_operational_manifest_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_operational_manifest_fields(item)
