"""Strict final assembly and semantic validation for Court datasets."""

from __future__ import annotations

import math
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from src.synthetic_data_generation.alignment.contracts import MetricSceneAdapter
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.components.camera_sampling.targeting import (
    NEAREST_COURT_TIE_TOLERANCE_M,
    target_court_policy_for_trajectory,
    validate_camera_looks_at_resolved_binding,
)
from src.synthetic_data_generation.dataset.court.components.labels import (
    AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON,
    CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M,
    MultiCourtProjectionAny,
    attach_renderer_visibility_from_validated_arrays,
    camera_center_court_y,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    LEGACY_ORBIT_STABLE_FIELDS,
    V4_ORBIT_STABLE_FIELDS,
    CourtDatasetPlanAny,
    CourtDatasetPlanV2,
    CourtDatasetPlanV4,
    OrbitCenter,
    OrbitCoverageObjective,
    OrbitSamplingPolicy,
    OrbitTargetMode,
    OrbitTrajectorySpec,
    OrbitTrajectorySpecV4,
    OrbitViewSpec,
    OrbitViewSpecV2,
    PlannedCourtSample,
    PlannedCourtSampleAny,
    PlannedCourtSampleV2,
    PlannedCourtSampleV4,
    RequiredTrajectoryCoverage,
    ResolvedTargetCourtV2,
    SelectedTrajectoryCoverage,
    SupportModelSummary,
    TargetCourtPolicyV2,
    TrajectoryGroupPlan,
    TrajectoryGroupPlanAny,
    TrajectoryGroupPlanV2,
    TrajectorySafetyEvaluation,
    TrajectorySemanticPhaseEvaluation,
    TrajectorySupportPolicy,
    build_selected_coverage_from_records,
    required_coverage_shortfall,
    semantic_phase_inventory_digest,
)
from src.synthetic_data_generation.dataset.court.diagnostics import (
    DIAGNOSTIC_FILES_V4_WITHOUT_SUPPORT_OCCUPANCY,
    diagnostic_files_for_version,
    write_court_diagnostics,
)
from src.synthetic_data_generation.dataset.court.occupancy_artifact import (
    CourtV4SupportOccupancyIdentity,
    load_court_v4_support_occupancy,
)
from src.synthetic_data_generation.dataset.court.performance import (
    CourtPerformanceEvidence,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
    CourtSchemaDefinition,
    court_schema_for_version,
    court_schema_from_dataset_schema,
)
from src.synthetic_data_generation.dataset.court.semantic_manifest import (
    COURT_SEMANTIC_MANIFEST_PATH,
    build_court_semantic_manifest,
    validate_court_semantic_manifest,
    validate_v2_published_court_geometry,
    validate_v3_published_court_geometry,
    validate_v4_published_court_geometry,
)
from src.synthetic_data_generation.dataset.court.shards import (
    CourtRenderedSample,
    CourtRenderResult,
    inspect_rendered_sample,
)
from src.synthetic_data_generation.dataset.runtime import (
    DatasetPerformanceBudget,
    DatasetPerformanceMetrics,
    PerformanceTimer,
    directory_size_bytes,
)
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.synthetic_data_generation.scene_contract import (
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.io import load_json, save_json_atomic

_COMMON_COURT_METRIC_KEYS = frozenset(
    {
        "proposal_count",
        "accepted_frame_count",
        "rejected_frame_count",
        "accepted_fraction",
        "trajectory_group_count",
        "maximum_adjacent_step_m",
        "split_frame_counts",
        "split_group_counts",
        "coverage_counts",
        "renderer_visible_points_by_class",
        "split_leakage_count",
    }
)
_COURT_METRIC_KEYS_BY_VERSION = {
    CourtDatasetSchemaVersion.V1: _COMMON_COURT_METRIC_KEYS
    | {"court_group_counts", "split_court_group_counts"},
    CourtDatasetSchemaVersion.V2: _COMMON_COURT_METRIC_KEYS
    | {"court_sample_counts", "split_court_sample_counts"},
    CourtDatasetSchemaVersion.V3: _COMMON_COURT_METRIC_KEYS
    | {"court_sample_counts", "split_court_sample_counts"},
    CourtDatasetSchemaVersion.V4: _COMMON_COURT_METRIC_KEYS
    | {
        "court_sample_counts",
        "split_court_sample_counts",
        "support_input_digest",
        "selected_safety_violation_count",
        "required_coverage",
        "selected_coverage",
        "required_coverage_shortfall",
        "optional_candidate_coverage_shortfall",
        "semantic_phase_inventory_digest",
        "projected_semantic_valid_frame_count",
        "projected_semantic_valid_fraction",
    },
}


def _uses_resolved_target_version(version: CourtDatasetSchemaVersion) -> bool:
    """Return the explicitly enumerated V2/V3 target-resolution contract."""
    if version is CourtDatasetSchemaVersion.V1:
        return False
    if version in (
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
        CourtDatasetSchemaVersion.V4,
    ):
        return True
    raise TypeError("Unsupported Court dataset schema version.")


@dataclass(frozen=True, slots=True)
class CourtAssemblyReport:
    """Release-gate evidence returned to the stage runner."""

    proposal_count: int
    accepted_frame_count: int
    rejected_frame_count: int
    trajectory_group_count: int
    maximum_adjacent_step_m: float
    accepted_fraction: float
    split_frame_counts: Mapping[str, int]
    court_group_counts: Mapping[str, int]
    performance: CourtPerformanceEvidence


class CourtArrayValidationMode(Enum):
    """Explicit array-value scan policy for canonical dataset validation."""

    FULL = "full"
    HEADERS_ONLY = "headers_only"


@dataclass(frozen=True, slots=True)
class _EvaluatedSample:
    rendered: CourtRenderedSample
    projection: MultiCourtProjectionAny
    accepted: bool
    rejection_reasons: tuple[str, ...]
    complete_array_scan_count: int


def assemble_court_dataset(
    staging_root: Path,
    *,
    plan: CourtDatasetPlanAny,
    layout: MultiCourtLayout,
    metric_adapter: MetricSceneAdapter,
    render_result: CourtRenderResult,
    configuration: CourtDatasetConfiguration,
    attempt_root: Path,
    performance_timer: PerformanceTimer,
) -> CourtAssemblyReport:
    """Stream staged samples through gates and assemble fixed outputs."""
    if (
        not staging_root.is_absolute()
        or not staging_root.is_dir()
        or staging_root.is_symlink()
    ):
        raise ValueError(
            "Court staging_root must be an existing absolute ordinary directory."
        )
    if not isinstance(render_result, CourtRenderResult):
        raise TypeError("Court assembly requires a complete CourtRenderResult.")
    if not isinstance(performance_timer, PerformanceTimer):
        raise TypeError("Court assembly requires an attempt-scoped PerformanceTimer.")
    if configuration.schema_version is not plan.schema_version:
        raise ValueError("Court configuration and plan schema versions disagree.")
    definition = court_schema_for_version(plan.schema_version)
    rendered_tuple = render_result.samples
    _validate_render_inventory(
        plan,
        rendered_tuple,
        pre_render_rejected_sample_ids=render_result.pre_render_rejected_sample_ids,
    )
    projection_by_id = render_result.projection_by_sample_id
    expected_projection_ids = tuple(
        sample.sample_id
        for sample in plan.samples
        if sample.sample_id in projection_by_id
    )
    if tuple(projection_by_id) != expected_projection_ids:
        raise ValueError(
            "Court pre-render projection inventory changed before assembly."
        )
    expected_court_ids = tuple(court.court_instance_id for court in layout.courts)
    if any(
        tuple(court.court_instance_id for court in projection.courts)
        != expected_court_ids
        for projection in projection_by_id.values()
    ):
        raise ValueError(
            "Court pre-render projections disagree with the alignment layout."
        )
    samples_root = staging_root / "samples"
    if samples_root.exists():
        raise ValueError("Court samples output already exists in staging.")
    samples_root.mkdir(parents=True, exist_ok=False)
    group_by_id = {group.trajectory_group_id: group for group in plan.groups}
    evaluated: list[_EvaluatedSample] = []
    accepted_records: list[dict[str, object]] = []
    for rendered_item in rendered_tuple:
        sample = rendered_item.sample
        destination = (
            samples_root
            / sample.split.value
            / sample.trajectory_group_id
            / sample.sample_id
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = rendered_item.source_directory
        if not source.resolve(strict=True).is_relative_to(
            attempt_root.resolve(strict=True)
        ):
            raise ValueError(
                "NHT render output must stay inside Court attempt staging."
            )
        if destination.exists():
            raise ValueError(f"Duplicate Court sample destination: {destination}")
        source.replace(destination)
        staged = _relocate_rendered_sample(rendered_item, destination=destination)
        evaluated_item = _evaluate_staged_sample(
            staged,
            projection=projection_by_id[sample.sample_id],
            metric_adapter=metric_adapter,
        )
        evaluated.append(evaluated_item)
        if not evaluated_item.accepted:
            shutil.rmtree(destination)
            continue
        group = group_by_id[sample.trajectory_group_id]
        view = next(value for value in group.views if value.view_id == sample.view_id)
        label_path = destination / "labels.json"
        metadata = _sample_metadata(
            sample,
            group=group,
            view_id=view.view_id,
            profile=plan.profile,
            metadata_fields=configuration.metadata_fields,
        )
        label_payload = {
            "schema": definition.sample_schema,
            "sample_index": sample.sample_index,
            "sample_id": sample.sample_id,
            "trajectory_group_id": sample.trajectory_group_id,
            "trajectory_id": sample.trajectory_id,
            "view_id": sample.view_id,
            "trajectory_frame_index": sample.trajectory_frame_index,
            "split": sample.split.value,
            "camera": sample.camera.to_dict(),
            "projection": evaluated_item.projection.to_dict(),
            "metadata": metadata,
        }
        if isinstance(sample, PlannedCourtSampleV2):
            label_payload["target_court"] = sample.target_court.to_dict()
        if isinstance(sample, PlannedCourtSampleV4):
            label_payload["safety_support_input_digest"] = (
                sample.safety_support_input_digest
            )
            label_payload["semantic_phase_index"] = sample.semantic_phase_index
            label_payload["semantic_phase_disposition_digest"] = (
                sample.semantic_phase_disposition_digest
            )
        save_json_atomic(label_payload, label_path)
        relative_directory = destination.relative_to(staging_root).as_posix()
        accepted_record: dict[str, object] = {
            "sample_index": sample.sample_index,
            "sample_id": sample.sample_id,
            "trajectory_group_id": sample.trajectory_group_id,
            "trajectory_id": sample.trajectory_id,
            "view_id": sample.view_id,
            "trajectory_frame_index": sample.trajectory_frame_index,
            "split": sample.split.value,
            "shard_id": sample.shard_id,
            "width": sample.camera.width,
            "height": sample.camera.height,
            "camera": sample.camera.to_dict(),
            "projection": evaluated_item.projection.to_dict(),
            "directory": relative_directory,
            "rgb": f"{relative_directory}/rgb.npy",
            "rgb_preview": f"{relative_directory}/rgb.png",
            "alpha": f"{relative_directory}/alpha.npy",
            "alpha_preview": f"{relative_directory}/alpha.png",
            "depth": f"{relative_directory}/depth.npy",
            "depth_coordinate_space": "metric_scene_metres",
            "labels": f"{relative_directory}/labels.json",
            "metadata": metadata,
        }
        if isinstance(sample, PlannedCourtSampleV2):
            accepted_record["target_court"] = sample.target_court.to_dict()
        if isinstance(sample, PlannedCourtSampleV4):
            accepted_record["safety_support_input_digest"] = (
                sample.safety_support_input_digest
            )
            accepted_record["semantic_phase_index"] = sample.semantic_phase_index
            accepted_record["semantic_phase_disposition_digest"] = (
                sample.semantic_phase_disposition_digest
            )
        accepted_records.append(accepted_record)
    accepted = tuple(item for item in evaluated if item.accepted)
    post_render_rejected = tuple(item for item in evaluated if not item.accepted)
    planned_by_id = {sample.sample_id: sample for sample in plan.samples}
    rejected_records = [
        _rejected_record(
            planned_by_id[sample_id],
            group=group_by_id[planned_by_id[sample_id].trajectory_group_id],
            projection=projection_by_id.get(sample_id),
            profile=plan.profile,
            metadata_fields=configuration.metadata_fields,
            reasons=render_result.pre_render_rejection_reason_by_sample_id[sample_id],
        )
        for sample_id in render_result.pre_render_rejected_sample_ids
    ]
    rejected_records.extend(
        _rejected_record(
            item.rendered.sample,
            group=group_by_id[item.rendered.sample.trajectory_group_id],
            projection=item.projection,
            profile=plan.profile,
            metadata_fields=configuration.metadata_fields,
            reasons=item.rejection_reasons,
        )
        for item in post_render_rejected
    )
    rejected_records.sort(key=lambda item: cast(int, item["sample_index"]))
    coverage_counts: Counter[str] = Counter(
        court.coverage_mode for item in accepted for court in item.projection.courts
    )
    visible_by_class: Counter[str] = Counter()
    accepted_by_group: Counter[str] = Counter()
    for item in accepted:
        accepted_by_group[item.rendered.sample.trajectory_group_id] += 1
        for court in item.projection.courts:
            for semantic_class in court.classes:
                visible_by_class[semantic_class.class_name] += sum(
                    point.renderer_visible is True for point in semantic_class.points
                )
    _apply_release_gates(
        plan,
        accepted_count=len(accepted),
        rejected_count=len(rejected_records),
        accepted_by_group=accepted_by_group,
        coverage_counts=coverage_counts,
        visible_by_class=visible_by_class,
    )
    if attempt_root.exists():
        if attempt_root.is_symlink() or not attempt_root.is_dir():
            raise ValueError("Court attempt root is not an ordinary directory.")
        shutil.rmtree(attempt_root)
    diagnostic_paths = write_court_diagnostics(
        staging_root / "diagnostics",
        plan=plan,
        accepted_sample_ids=[item.rendered.sample.sample_id for item in accepted],
        rejected=rejected_records,
        coverage_counts=coverage_counts,
        visible_by_class={
            name: visible_by_class[name] for name in definition.semantic_class_names
        },
        layout=layout,
    )
    metrics = _metrics(
        plan,
        accepted_records=accepted_records,
        rejected_count=len(rejected_records),
        coverage_counts=coverage_counts,
        visible_by_class=visible_by_class,
    )
    manifest = {
        "schema": definition.dataset_schema,
        "status": "completed",
        "scene_id": plan.scene_id,
        "profile": plan.profile,
        "seed": plan.policy.seed,
        "sampling_policy": plan.policy.to_dict(),
        "metadata_fields": list(configuration.metadata_fields),
        "trajectory_groups": [group.to_dict() for group in plan.groups],
        "samples": accepted_records,
        "rejected_samples": rejected_records,
        "metrics": metrics,
        "diagnostics": list(diagnostic_paths),
    }
    save_json_atomic(manifest, staging_root / "dataset.json")
    semantic_manifest = build_court_semantic_manifest(manifest)
    save_json_atomic(
        semantic_manifest,
        staging_root / COURT_SEMANTIC_MANIFEST_PATH,
    )
    _write_performance_evidence(
        staging_root,
        timer=performance_timer,
        render_result=render_result,
        proposal_count=plan.proposal_count,
        accepted_frame_count=len(accepted),
        rejected_frame_count=len(rejected_records),
        accepted_staged_complete_array_scans=sum(
            item.complete_array_scan_count for item in accepted
        ),
        post_render_rejected_staged_complete_array_scans=sum(
            item.complete_array_scan_count for item in post_render_rejected
        ),
        budget=configuration.performance,
        visible_by_class=visible_by_class,
        schema_definition=definition,
    )
    return validate_court_dataset(
        staging_root,
        expected_plan=plan,
        expected_configuration=configuration,
        array_validation=CourtArrayValidationMode.HEADERS_ONLY,
    )


def _validate_render_inventory(
    plan: CourtDatasetPlanAny,
    rendered: Sequence[CourtRenderedSample],
    *,
    pre_render_rejected_sample_ids: Sequence[str],
) -> None:
    rejected_ids = tuple(pre_render_rejected_sample_ids)
    if len(rejected_ids) != len(set(rejected_ids)):
        raise ValueError("Court pre-render rejection inventory contains duplicates.")
    by_id: dict[str, CourtRenderedSample] = {}
    for item in rendered:
        sample_id = item.sample.sample_id
        if sample_id in by_id:
            raise ValueError(f"Duplicate renderer sample ID: {sample_id}.")
        by_id[sample_id] = item
    expected_ids = [sample.sample_id for sample in plan.samples]
    if set(by_id) & set(rejected_ids) or set(by_id) | set(rejected_ids) != set(
        expected_ids
    ):
        raise ValueError(
            "Court renderer/rejection sample partition mismatch; "
            f"missing={sorted(set(expected_ids) - set(by_id) - set(rejected_ids))}, "
            f"unexpected={sorted(set(by_id) - set(expected_ids))}."
        )
    for expected in plan.samples:
        if expected.sample_id in rejected_ids:
            continue
        actual = by_id[expected.sample_id].sample
        if actual != expected:
            raise ValueError(
                f"Renderer sample metadata changed for {expected.sample_id}."
            )
        inspect_rendered_sample(by_id[expected.sample_id])


def _relocate_rendered_sample(
    rendered: CourtRenderedSample,
    *,
    destination: Path,
) -> CourtRenderedSample:
    """Bind fixed filenames after moving one NHT result into final staging."""
    return CourtRenderedSample(
        sample=rendered.sample,
        rgb_path=destination / "rgb.npy",
        rgb_preview_path=destination / "rgb.png",
        alpha_path=destination / "alpha.npy",
        alpha_preview_path=destination / "alpha.png",
        depth_path=destination / "depth.npy",
    )


def _evaluate_staged_sample(
    rendered: CourtRenderedSample,
    *,
    projection: MultiCourtProjectionAny,
    metric_adapter: MetricSceneAdapter,
) -> _EvaluatedSample:
    """Load, evaluate, and release one staged payload before advancing."""
    inspect_rendered_sample(rendered)
    arrays = NHTRenderArrays(
        rgb=np.load(rendered.rgb_path, allow_pickle=False),
        alpha=np.load(rendered.alpha_path, allow_pickle=False),
        depth=np.load(rendered.depth_path, allow_pickle=False),
    )
    visible = attach_renderer_visibility_from_validated_arrays(
        projection,
        alpha=arrays.alpha,
        depth=arrays.depth,
    )
    reasons: list[str] = []
    if visible.visible_point_count == 0:
        reasons.append("no_renderer_visible_semantic_point")
    if not reasons:
        depth_metric = arrays.metric_depth(
            nht_scene_units_per_metre=metric_adapter.nht_scene_units_per_metre,
        )
        np.save(rendered.depth_path, depth_metric, allow_pickle=False)
    return _EvaluatedSample(
        rendered=rendered,
        projection=visible,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
        complete_array_scan_count=1,
    )


def _apply_release_gates(
    plan: CourtDatasetPlanAny,
    *,
    accepted_count: int,
    rejected_count: int,
    accepted_by_group: Mapping[str, int],
    coverage_counts: Mapping[str, int],
    visible_by_class: Mapping[str, int],
) -> None:
    if accepted_count + rejected_count != plan.proposal_count:
        raise ValueError("Accepted/rejected counts do not cover every proposal.")
    if accepted_count < plan.policy.minimum_accepted_frames:
        raise ValueError(
            "Court accepted-frame count is below the resolved quality gate."
        )
    accepted_fraction = accepted_count / plan.proposal_count
    if accepted_fraction < plan.policy.minimum_accepted_fraction:
        raise ValueError("Court accepted fraction is below the resolved quality gate.")
    zero_groups = [
        group.trajectory_group_id
        for group in plan.groups
        if accepted_by_group.get(group.trajectory_group_id, 0) == 0
    ]
    if zero_groups:
        raise ValueError(
            f"Production trajectories with zero accepted frames: {zero_groups}."
        )
    required_coverage = {"full", "near_full", "partial"}
    missing_coverage = required_coverage - {
        name for name, count in coverage_counts.items() if count > 0
    }
    if missing_coverage:
        raise ValueError(
            f"Court output lacks required geometric coverage: {sorted(missing_coverage)}."
        )
    missing_classes = [
        name
        for name in court_schema_for_version(plan.schema_version).semantic_class_names
        if visible_by_class.get(name, 0) <= 0
    ]
    if missing_classes:
        raise ValueError(
            f"Court output lacks renderer-visible semantic classes: {missing_classes}."
        )


def _sample_metadata(
    sample: PlannedCourtSampleAny,
    *,
    group: TrajectoryGroupPlanAny,
    view_id: str,
    profile: str,
    metadata_fields: Sequence[str],
) -> dict[str, object]:
    if isinstance(sample, PlannedCourtSampleV2) and isinstance(
        group, TrajectoryGroupPlanV2
    ):
        binding = sample.target_court.binding
    elif isinstance(sample, PlannedCourtSample) and isinstance(
        group, TrajectoryGroupPlan
    ):
        binding = group.target_court
    else:
        raise TypeError("Court sample and group versions are mixed.")
    available: dict[str, object] = {
        "target_court": binding.court_instance_id,
        "candidate_id": binding.candidate_id,
        "transform": binding.scene_from_court.to_list(),
        "camera_profile": profile,
        "camera_parameters": {
            "view_id": view_id,
            "camera_center_scene_m": list(sample.camera_center_scene_m),
            "intrinsics": list(sample.camera.intrinsics),
            "camera_to_scene": sample.camera.camera_to_scene.to_list(),
        },
        "seed": binding.selection_seed,
    }
    unknown = set(metadata_fields) - set(available)
    if unknown:
        raise ValueError(
            f"Court metadata fields lack a semantic source: {sorted(unknown)}."
        )
    return {field: available[field] for field in metadata_fields}


def _rejected_record(
    sample: PlannedCourtSampleAny,
    *,
    group: TrajectoryGroupPlanAny,
    projection: MultiCourtProjectionAny | None,
    profile: str,
    metadata_fields: Sequence[str],
    reasons: Sequence[str],
) -> dict[str, object]:
    """Retain complete stable semantics for a rejected renderer proposal."""
    reason_tuple = tuple(reasons)
    if not reason_tuple or any(
        not isinstance(reason, str) or not reason or reason != reason.strip()
        for reason in reason_tuple
    ):
        raise ValueError("Rejected Court samples require explicit semantic reasons.")
    if projection is not None and projection.camera_id != sample.sample_id:
        raise ValueError("Rejected Court projection disagrees with the planned sample.")
    if projection is None and not (
        isinstance(sample, PlannedCourtSampleV2)
        and len(reason_tuple) == 1
        and _is_ambiguous_near_far_reason(reason_tuple[0])
    ):
        raise ValueError(
            "Only an explicit singleton mid-plane ambiguity may omit projection."
        )
    record: dict[str, object] = {
        "sample_index": sample.sample_index,
        "sample_id": sample.sample_id,
        "trajectory_group_id": sample.trajectory_group_id,
        "trajectory_id": sample.trajectory_id,
        "view_id": sample.view_id,
        "trajectory_frame_index": sample.trajectory_frame_index,
        "split": sample.split.value,
        "shard_id": sample.shard_id,
        "width": sample.camera.width,
        "height": sample.camera.height,
        "camera": sample.camera.to_dict(),
        "projection": projection.to_dict() if projection is not None else None,
        "metadata": _sample_metadata(
            sample,
            group=group,
            view_id=sample.view_id,
            profile=profile,
            metadata_fields=metadata_fields,
        ),
        "reasons": list(reason_tuple),
    }
    if isinstance(sample, PlannedCourtSampleV2):
        record["target_court"] = sample.target_court.to_dict()
    if isinstance(sample, PlannedCourtSampleV4):
        record["safety_support_input_digest"] = sample.safety_support_input_digest
        record["semantic_phase_index"] = sample.semantic_phase_index
        record["semantic_phase_disposition_digest"] = (
            sample.semantic_phase_disposition_digest
        )
    return record


def _metrics(
    plan: CourtDatasetPlanAny,
    *,
    accepted_records: Sequence[Mapping[str, object]],
    rejected_count: int,
    coverage_counts: Mapping[str, int],
    visible_by_class: Mapping[str, int],
) -> dict[str, object]:
    accepted_count = len(accepted_records)
    split_frame_counts = Counter(str(record["split"]) for record in accepted_records)
    split_group_counts = Counter(group.split.value for group in plan.groups)
    result: dict[str, object] = {
        "proposal_count": plan.proposal_count,
        "accepted_frame_count": accepted_count,
        "rejected_frame_count": rejected_count,
        "accepted_fraction": accepted_count / plan.proposal_count,
        "trajectory_group_count": len(plan.groups),
        "maximum_adjacent_step_m": max(
            group.maximum_adjacent_step_m for group in plan.groups
        ),
        "split_frame_counts": dict(sorted(split_frame_counts.items())),
        "split_group_counts": dict(sorted(split_group_counts.items())),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "renderer_visible_points_by_class": {
            name: visible_by_class.get(name, 0)
            for name in court_schema_for_version(
                plan.schema_version
            ).semantic_class_names
        },
        "split_leakage_count": 0,
    }
    if isinstance(plan, CourtDatasetPlanV2):
        court_sample_counts: Counter[str] = Counter()
        split_court_sample_counts: dict[str, Counter[str]] = defaultdict(Counter)
        accepted_ids = {str(record["sample_id"]) for record in accepted_records}
        for sample in plan.samples:
            if sample.sample_id not in accepted_ids:
                continue
            court_id = sample.target_court.binding.court_instance_id
            court_sample_counts[court_id] += 1
            split_court_sample_counts[sample.split.value][court_id] += 1
        result["court_sample_counts"] = dict(sorted(court_sample_counts.items()))
        result["split_court_sample_counts"] = {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_court_sample_counts.items())
        }
    else:
        court_group_counts = Counter(
            group.target_court.court_instance_id for group in plan.groups
        )
        split_court_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for group in plan.groups:
            split_court_group_counts[group.split.value][
                group.target_court.court_instance_id
            ] += 1
        result["court_group_counts"] = dict(sorted(court_group_counts.items()))
        result["split_court_group_counts"] = {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_court_group_counts.items())
        }
    if isinstance(plan, CourtDatasetPlanV4):
        result["support_input_digest"] = plan.support_summary.input_digest
        result["selected_safety_violation_count"] = sum(
            len(group.safety_evaluation.violating_point_indices)
            + len(group.safety_evaluation.violating_segment_indices)
            for group in plan.groups
        )
        result["required_coverage"] = plan.required_coverage.to_dict()
        result["selected_coverage"] = plan.selected_coverage.to_dict()
        result["required_coverage_shortfall"] = list(
            plan.required_coverage_shortfall
        )
        result["optional_candidate_coverage_shortfall"] = list(
            plan.optional_candidate_coverage_shortfall
        )
        result["semantic_phase_inventory_digest"] = plan.semantic_phase_inventory_digest
        result["projected_semantic_valid_frame_count"] = (
            plan.projected_semantic_valid_frame_count
        )
        result["projected_semantic_valid_fraction"] = (
            plan.projected_semantic_valid_fraction
        )
    return result


def _write_performance_evidence(
    root: Path,
    *,
    timer: PerformanceTimer,
    render_result: CourtRenderResult,
    proposal_count: int,
    accepted_frame_count: int,
    rejected_frame_count: int,
    accepted_staged_complete_array_scans: int,
    post_render_rejected_staged_complete_array_scans: int,
    budget: DatasetPerformanceBudget,
    visible_by_class: Mapping[str, int],
    schema_definition: CourtSchemaDefinition | None = None,
) -> CourtPerformanceEvidence:
    """Persist self-consistent measured bytes and fail closed on every budget."""
    definition = schema_definition or court_schema_for_version(
        CourtDatasetSchemaVersion.V1
    )
    wall_seconds, cpu_seconds, peak_rss_bytes = timer.elapsed()
    published_bytes = max(1, directory_size_bytes(root))
    performance_path = root / "diagnostics" / "performance.json"
    pre_render_rejected_sample_count = len(render_result.pre_render_rejected_sample_ids)
    renderable_sample_count = proposal_count - pre_render_rejected_sample_count
    post_render_rejected_sample_count = (
        rejected_frame_count - pre_render_rejected_sample_count
    )
    staged_complete_array_scans = (
        accepted_staged_complete_array_scans
        + post_render_rejected_staged_complete_array_scans
    )
    fresh_rendered_sample_count = render_result.nht_complete_array_scans
    reused_rendered_sample_count = renderable_sample_count - fresh_rendered_sample_count
    fresh_run_complete_array_scan_requirement = 2 * renderable_sample_count
    complete_array_scan_budget_capacity = (
        budget.maximum_complete_array_scans_per_sample * renderable_sample_count
    )
    evidence: CourtPerformanceEvidence | None = None
    for _ in range(8):
        metrics = DatasetPerformanceMetrics(
            domain="court",
            wall_seconds=wall_seconds,
            cpu_seconds=cpu_seconds,
            peak_rss_bytes=peak_rss_bytes,
            execution_device=budget.execution_device,
            cuda_peak_bytes=0,
            nht_invocations=render_result.nht_invocations,
            background_cache_misses=0,
            complete_array_scans=(
                render_result.nht_complete_array_scans + staged_complete_array_scans
            ),
            generated_bytes=render_result.generated_bytes,
            published_bytes=published_bytes,
            dense_reference_bytes=published_bytes,
            frame_count=accepted_frame_count,
            camera_count=accepted_frame_count,
            sample_count=accepted_frame_count,
        )
        evidence = CourtPerformanceEvidence(
            budget=budget,
            metrics=metrics,
            resolved_shard_count=render_result.resolved_shard_count,
            maximum_shard_sample_count=render_result.maximum_shard_sample_count,
            request_path_count=render_result.request_path_count,
            proposal_count=proposal_count,
            accepted_frame_count=accepted_frame_count,
            rejected_frame_count=rejected_frame_count,
            pre_render_checked_sample_count=proposal_count,
            pre_render_rejected_sample_count=pre_render_rejected_sample_count,
            renderable_sample_count=renderable_sample_count,
            post_render_rejected_sample_count=post_render_rejected_sample_count,
            depth_conversion_count=accepted_frame_count,
            fresh_rendered_sample_count=fresh_rendered_sample_count,
            reused_rendered_sample_count=reused_rendered_sample_count,
            nht_boundary_complete_array_scans=(render_result.nht_complete_array_scans),
            accepted_staged_complete_array_scans=(accepted_staged_complete_array_scans),
            post_render_rejected_staged_complete_array_scans=(
                post_render_rejected_staged_complete_array_scans
            ),
            staged_complete_array_scans=staged_complete_array_scans,
            fresh_run_complete_array_scan_requirement=(
                fresh_run_complete_array_scan_requirement
            ),
            complete_array_scan_budget_capacity=(complete_array_scan_budget_capacity),
            scene_validation_count=render_result.scene_validation_count,
            preview_validation_count=render_result.preview_validation_count,
            loaded_array_bytes=render_result.loaded_array_bytes,
            maximum_nht_live_array_bytes=(render_result.maximum_nht_live_array_bytes),
            retained_nht_array_bytes=render_result.retained_nht_array_bytes,
            external_nht_boundary_wall_seconds=(
                render_result.external_nht_boundary_wall_seconds
            ),
            shard_wall_seconds={
                timing.shard_id: timing.wall_seconds
                for timing in render_result.shard_timings
            },
            visible_points_by_class=visible_by_class,
            schema=definition.performance_schema,
        )
        save_json_atomic(evidence.to_dict(), performance_path)
        actual_bytes = directory_size_bytes(root)
        if actual_bytes == published_bytes:
            return evidence
        published_bytes = actual_bytes
    raise RuntimeError("Court published-byte evidence did not converge.")


def validate_court_dataset(
    root: Path,
    *,
    expected_plan: CourtDatasetPlanAny | None = None,
    expected_configuration: CourtDatasetConfiguration | None = None,
    array_validation: CourtArrayValidationMode = CourtArrayValidationMode.FULL,
    require_v4_support_occupancy: bool = True,
) -> CourtAssemblyReport:
    """Validate the complete canonical output and reject every inventory mismatch."""
    if not isinstance(array_validation, CourtArrayValidationMode):
        raise TypeError("array_validation must be a CourtArrayValidationMode.")
    if not isinstance(require_v4_support_occupancy, bool):
        raise TypeError("require_v4_support_occupancy must be boolean.")
    manifest_path = root / "dataset.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"Court dataset manifest is missing: {manifest_path}")
    raw = load_json(manifest_path)
    keys = {
        "schema",
        "status",
        "scene_id",
        "profile",
        "seed",
        "sampling_policy",
        "metadata_fields",
        "trajectory_groups",
        "samples",
        "rejected_samples",
        "metrics",
        "diagnostics",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ValueError("Court dataset manifest schema is invalid.")
    _require_finite_json(raw, name="dataset")
    definition = court_schema_from_dataset_schema(raw["schema"])
    if raw["status"] != "completed":
        raise ValueError("Court dataset schema/status is invalid.")
    if (
        expected_plan is not None
        and expected_plan.schema_version is not definition.version
    ):
        raise ValueError("Published Court dataset and expected plan schemas are mixed.")
    if (
        expected_configuration is not None
        and expected_configuration.schema_version is not definition.version
    ):
        raise ValueError(
            "Published Court dataset and expected configuration schemas are mixed."
        )
    policy = _parse_canonical_sampling_policy(
        raw["sampling_policy"], version=definition.version
    )
    groups = _mapping_sequence(raw["trajectory_groups"], name="trajectory_groups")
    samples = _mapping_sequence(raw["samples"], name="samples")
    rejected = _mapping_sequence(raw["rejected_samples"], name="rejected_samples")
    if definition.version is CourtDatasetSchemaVersion.V1:
        published_court_geometry = None
    elif definition.version is CourtDatasetSchemaVersion.V2:
        published_court_geometry = validate_v2_published_court_geometry(raw)
    elif definition.version is CourtDatasetSchemaVersion.V3:
        published_court_geometry = validate_v3_published_court_geometry(raw)
    elif definition.version is CourtDatasetSchemaVersion.V4:
        published_court_geometry = validate_v4_published_court_geometry(raw)
    else:  # pragma: no cover - exact schema registry is exhaustive
        raise TypeError("Unsupported Court dataset schema version.")
    validate_court_semantic_manifest(
        raw,
        load_json(_contained_file(root, COURT_SEMANTIC_MANIFEST_PATH)),
    )
    metrics = raw["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("Court metrics must be a mapping.")
    _validate_metric_schema(metrics, definition=definition)
    for group in groups:
        _validate_group_record(group, definition=definition)
    group_ids = [_nested_group_id(group) for group in groups]
    if not group_ids or len(group_ids) != len(set(group_ids)):
        raise ValueError("Court trajectory group IDs must be non-empty and unique.")
    split_by_group = {
        group_id: group["split"]
        for group_id, group in zip(group_ids, groups, strict=True)
    }
    group_by_id = dict(zip(group_ids, groups, strict=True))
    accepted_ids: set[str] = set()
    rejected_ids: set[str] = set()
    proposal_indices: set[int] = set()
    accepted_by_group: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    metadata_fields = raw["metadata_fields"]
    if not isinstance(metadata_fields, list) or any(
        not isinstance(field, str) for field in metadata_fields
    ):
        raise TypeError("Court metadata_fields must be a string list.")
    rejected_keys = {
        "sample_index",
        "sample_id",
        "trajectory_group_id",
        "trajectory_id",
        "view_id",
        "trajectory_frame_index",
        "split",
        "shard_id",
        "width",
        "height",
        "camera",
        "projection",
        "metadata",
        "reasons",
    }
    if _uses_resolved_target_version(definition.version):
        rejected_keys.add("target_court")
    if definition.version is CourtDatasetSchemaVersion.V4:
        rejected_keys.update(
            {
                "safety_support_input_digest",
                "semantic_phase_index",
                "semantic_phase_disposition_digest",
            }
        )
    for record in rejected:
        if set(record) != rejected_keys or not isinstance(record.get("reasons"), list):
            raise ValueError("Rejected Court sample record schema is invalid.")
        sample_id, sample_index = _validate_semantic_sample_record(
            record,
            group_by_id=group_by_id,
            metadata_fields=metadata_fields,
            profile=raw["profile"],
            definition=definition,
            published_court_geometry=published_court_geometry,
        )
        if sample_id in rejected_ids:
            raise ValueError(f"Duplicate rejected Court sample ID: {sample_id}.")
        rejected_ids.add(sample_id)
        if sample_index in proposal_indices:
            raise ValueError(f"Duplicate Court proposal index: {sample_index}.")
        proposal_indices.add(sample_index)
    for record in samples:
        expected_sample_keys = {
            "sample_index",
            "sample_id",
            "trajectory_group_id",
            "trajectory_id",
            "view_id",
            "trajectory_frame_index",
            "split",
            "shard_id",
            "width",
            "height",
            "camera",
            "projection",
            "directory",
            "rgb",
            "rgb_preview",
            "alpha",
            "alpha_preview",
            "depth",
            "depth_coordinate_space",
            "labels",
            "metadata",
        }
        if _uses_resolved_target_version(definition.version):
            expected_sample_keys.add("target_court")
        if definition.version is CourtDatasetSchemaVersion.V4:
            expected_sample_keys.update(
                {
                    "safety_support_input_digest",
                    "semantic_phase_index",
                    "semantic_phase_disposition_digest",
                }
            )
        if set(record) != expected_sample_keys:
            raise ValueError(
                "Court sample record contains missing or unexpected fields."
            )
        if record["depth_coordinate_space"] != "metric_scene_metres":
            raise ValueError("Court sample depth must use metric scene metres.")
        sample_id, sample_index = _validate_semantic_sample_record(
            record,
            group_by_id=group_by_id,
            metadata_fields=metadata_fields,
            profile=raw["profile"],
            definition=definition,
            published_court_geometry=published_court_geometry,
        )
        group_id = record.get("trajectory_group_id")
        split = record.get("split")
        if not isinstance(group_id, str):
            raise TypeError("Court sample/group IDs must be strings.")
        if sample_id in accepted_ids or sample_id in rejected_ids:
            raise ValueError(f"Duplicate Court sample ID: {sample_id}.")
        accepted_ids.add(sample_id)
        if sample_index in proposal_indices:
            raise ValueError(f"Duplicate Court proposal index: {sample_index}.")
        proposal_indices.add(sample_index)
        if group_id not in split_by_group or split_by_group[group_id] != split:
            raise ValueError("Court sample split/group relationship is invalid.")
        accepted_by_group[group_id] += 1
        split_counts[str(split)] += 1
        _validate_published_sample(
            root,
            record,
            array_validation=array_validation,
            definition=definition,
        )
    if set(accepted_by_group) != set(group_ids):
        raise ValueError("A production trajectory has zero accepted frames.")
    proposal_count = _metric_integer(metrics, "proposal_count", minimum=1)
    accepted_count = _metric_integer(metrics, "accepted_frame_count", minimum=1)
    rejected_count = _metric_integer(metrics, "rejected_frame_count", minimum=0)
    group_count = _metric_integer(metrics, "trajectory_group_count", minimum=1)
    if accepted_count != len(samples) or rejected_count != len(rejected):
        raise ValueError("Court manifest metrics disagree with sample inventories.")
    if proposal_count != accepted_count + rejected_count or group_count != len(groups):
        raise ValueError("Court proposal/group metrics are inconsistent.")
    if proposal_indices != set(range(proposal_count)):
        raise ValueError(
            "Court proposal indices do not cover the full planned inventory."
        )
    _validate_metric_inventories(
        metrics,
        groups=groups,
        samples=samples,
        split_counts=split_counts,
        definition=definition,
    )
    budget = policy.proposal_budget
    minimum_groups = policy.minimum_trajectory_groups
    minimum_frames = policy.minimum_accepted_frames
    minimum_fraction = policy.minimum_accepted_fraction
    max_allowed_step = policy.max_arc_step_m
    observed_fraction = _mapping_float(metrics, "accepted_fraction")
    observed_step = _mapping_float(metrics, "maximum_adjacent_step_m")
    if proposal_count > budget or budget > 5_000:
        raise ValueError("Court proposal budget gate failed.")
    if group_count < minimum_groups or accepted_count < minimum_frames:
        raise ValueError("Court quantitative production gates failed.")
    if observed_fraction < minimum_fraction or not math.isclose(
        observed_fraction,
        accepted_count / proposal_count,
        abs_tol=1.0e-12,
        rel_tol=0.0,
    ):
        raise ValueError("Court accepted-fraction gate failed.")
    if observed_step > max_allowed_step + 1.0e-9 or observed_step > 1.05 + 1.0e-9:
        raise ValueError("Court maximum adjacent arc-step gate failed.")
    if metrics.get("split_leakage_count") != 0:
        raise ValueError("Court trajectory group split leakage is non-zero.")
    if definition.version is CourtDatasetSchemaVersion.V4:
        digest = metrics.get("support_input_digest")
        phase_digest = metrics.get("semantic_phase_inventory_digest")
        required_coverage = RequiredTrajectoryCoverage.from_mapping(
            metrics.get("required_coverage")
        )
        selected_coverage = SelectedTrajectoryCoverage.from_mapping(
            metrics.get("selected_coverage")
        )
        recomputed_coverage = _serialized_selected_coverage(
            groups,
            required_coverage=required_coverage,
        )
        required_shortfall = metrics.get("required_coverage_shortfall")
        optional_shortfall = metrics.get("optional_candidate_coverage_shortfall")
        projected_valid_count = metrics.get("projected_semantic_valid_frame_count")
        projected_valid_fraction = metrics.get("projected_semantic_valid_fraction")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(phase_digest, str)
            or len(phase_digest) != 64
            or any(character not in "0123456789abcdef" for character in phase_digest)
            or metrics.get("selected_safety_violation_count") != 0
            or selected_coverage != recomputed_coverage
            or not isinstance(required_shortfall, list)
            or required_shortfall
            or required_shortfall
            != list(required_coverage_shortfall(required_coverage, recomputed_coverage))
            or not isinstance(optional_shortfall, list)
            or optional_shortfall != sorted(set(optional_shortfall))
            or isinstance(projected_valid_count, bool)
            or not isinstance(projected_valid_count, int)
            or projected_valid_count < minimum_frames
            or isinstance(projected_valid_fraction, bool)
            or not isinstance(projected_valid_fraction, int | float)
            or not math.isfinite(float(projected_valid_fraction))
            or float(projected_valid_fraction) < minimum_fraction
        ):
            raise ValueError("Court V4 safety metrics are invalid.")
    coverage = metrics.get("coverage_counts")
    visible = metrics.get("renderer_visible_points_by_class")
    if not isinstance(coverage, Mapping) or not {
        "full",
        "near_full",
        "partial",
    }.issubset(
        {
            name
            for name, count in coverage.items()
            if isinstance(count, int) and count > 0
        }
    ):
        raise ValueError("Court coverage diversity gate failed.")
    if (
        not isinstance(visible, Mapping)
        or set(visible) != set(definition.semantic_class_names)
        or any(
            not isinstance(visible[name], int) or visible[name] <= 0
            for name in definition.semantic_class_names
        )
    ):
        raise ValueError("Court renderer visibility gate disagrees with its schema.")
    if definition.version is CourtDatasetSchemaVersion.V1:
        _validate_court_balance(groups)
    diagnostics = raw["diagnostics"]
    expected_diagnostics = [
        f"diagnostics/{name}"
        for name in diagnostic_files_for_version(definition.version)
    ]
    legacy_v4_diagnostics = [
        f"diagnostics/{name}"
        for name in DIAGNOSTIC_FILES_V4_WITHOUT_SUPPORT_OCCUPANCY
    ]
    has_support_occupancy = diagnostics == expected_diagnostics
    legacy_semantic_v4 = (
        definition.version is CourtDatasetSchemaVersion.V4
        and not require_v4_support_occupancy
        and diagnostics == legacy_v4_diagnostics
    )
    if not has_support_occupancy and not legacy_semantic_v4:
        raise ValueError("Court diagnostic inventory is incomplete or unexpected.")
    validated_diagnostics = (
        expected_diagnostics if has_support_occupancy else legacy_v4_diagnostics
    )
    for relative in validated_diagnostics:
        _contained_file(root, relative)
    _validate_diagnostic_schemas(
        root,
        definition=definition,
        dataset=raw,
        validate_support_occupancy=has_support_occupancy,
    )
    performance = CourtPerformanceEvidence.from_dict(
        load_json(_contained_file(root, "diagnostics/performance.json"))
    )
    if (
        performance.proposal_count != proposal_count
        or performance.accepted_frame_count != accepted_count
        or performance.rejected_frame_count != rejected_count
        or performance.visible_points_by_class
        != {name: int(visible[name]) for name in definition.semantic_class_names}
        or performance.schema != definition.performance_schema
    ):
        raise ValueError("Court performance evidence disagrees with semantic metrics.")
    actual_published_bytes = directory_size_bytes(root)
    if performance.metrics.published_bytes != actual_published_bytes:
        raise ValueError("Court published-byte evidence disagrees with the dataset.")
    if expected_plan is not None:
        if (
            raw["scene_id"] != expected_plan.scene_id
            or raw["profile"] != expected_plan.profile
        ):
            raise ValueError(
                "Published Court scene/profile disagrees with the resolved plan."
            )
        if proposal_count != expected_plan.proposal_count or group_ids != [
            group.trajectory_group_id for group in expected_plan.groups
        ]:
            raise ValueError(
                "Published Court inventory disagrees with the resolved plan."
            )
        if list(groups) != [group.to_dict() for group in expected_plan.groups]:
            raise ValueError(
                "Published Court trajectory semantics changed after planning."
            )
        if policy.to_dict() != expected_plan.policy.to_dict():
            raise ValueError("Published Court sampling policy changed after planning.")
        if isinstance(expected_plan, CourtDatasetPlanV4) and has_support_occupancy:
            expected_occupancy = expected_plan.support_occupancy_snapshot
            published_occupancy = load_court_v4_support_occupancy(
                root,
                expected_scene_id=expected_plan.scene_id,
                expected_profile=expected_plan.profile,
                expected_policy_decision_id=expected_occupancy.policy_decision_id,
                expected_support_input_digest=(
                    expected_occupancy.support_input_digest
                ),
                expected_voxel_size_m=expected_occupancy.voxel_size_m,
                expected_cell_count=expected_occupancy.cell_count,
                expected_content_digest=expected_occupancy.content_digest,
                maximum_cells=expected_plan.support_policy.maximum_occupancy_cells,
            )
            if (
                published_occupancy.snapshot.content_digest
                != expected_occupancy.content_digest
            ):
                raise ValueError(
                    "Published Court V4 occupancy changed after planning."
                )
        expected_samples = {
            sample.sample_id: sample.to_dict() for sample in expected_plan.samples
        }
        for record in (*samples, *rejected):
            sample_id = cast(str, record["sample_id"])
            try:
                expected = expected_samples[sample_id]
            except KeyError as error:
                raise ValueError(
                    "Published Court sample is absent from the resolved plan."
                ) from error
            for key in (
                "sample_index",
                "sample_id",
                "trajectory_group_id",
                "trajectory_id",
                "view_id",
                "trajectory_frame_index",
                "split",
                "shard_id",
                "camera",
            ):
                if record[key] != expected[key]:
                    raise ValueError(
                        "Published Court sample semantics changed after planning."
                    )
            if _uses_resolved_target_version(definition.version) and (
                record.get("target_court") != expected.get("target_court")
            ):
                raise ValueError(
                    "Published Court sample target changed after planning."
                )
    if expected_configuration is not None and metadata_fields != list(
        expected_configuration.metadata_fields
    ):
        raise ValueError("Published Court metadata fields changed after configuration.")
    if (
        expected_configuration is not None
        and performance.budget != expected_configuration.performance
    ):
        raise ValueError(
            "Published Court performance budget changed after configuration."
        )
    court_counts_key = (
        "court_group_counts"
        if not _uses_resolved_target_version(definition.version)
        else "court_sample_counts"
    )
    court_counts_raw = metrics.get(court_counts_key)
    if not isinstance(court_counts_raw, Mapping):
        raise TypeError(f"{court_counts_key} must be a mapping.")
    return CourtAssemblyReport(
        proposal_count=proposal_count,
        accepted_frame_count=accepted_count,
        rejected_frame_count=rejected_count,
        trajectory_group_count=group_count,
        maximum_adjacent_step_m=observed_step,
        accepted_fraction=observed_fraction,
        split_frame_counts=dict(split_counts),
        court_group_counts={
            str(key): int(value) for key, value in court_counts_raw.items()
        },
        performance=performance,
    )


def _validate_semantic_sample_record(
    record: Mapping[str, object],
    *,
    group_by_id: Mapping[str, Mapping[str, object]],
    metadata_fields: Sequence[str],
    profile: object,
    definition: CourtSchemaDefinition,
    published_court_geometry: Mapping[str, RigidTransform] | None,
) -> tuple[str, int]:
    """Cross-check embedded sample semantics against its trajectory group."""
    sample_index = _record_integer(record, "sample_index", minimum=0)
    trajectory_frame_index = _record_integer(
        record,
        "trajectory_frame_index",
        minimum=0,
    )
    width = _record_integer(record, "width", minimum=2)
    height = _record_integer(record, "height", minimum=2)
    sample_id = record.get("sample_id")
    group_id = record.get("trajectory_group_id")
    trajectory_id = record.get("trajectory_id")
    view_id = record.get("view_id")
    split = record.get("split")
    shard_id = record.get("shard_id")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in (sample_id, group_id, trajectory_id, view_id, split, shard_id)
    ):
        raise TypeError("Court sample semantic identifiers must be trimmed strings.")
    assert isinstance(sample_id, str)
    assert isinstance(group_id, str)
    assert isinstance(trajectory_id, str)
    assert isinstance(view_id, str)
    assert isinstance(split, str)
    assert isinstance(shard_id, str)
    try:
        group = group_by_id[group_id]
    except KeyError as error:
        raise ValueError(
            "Court sample references an unknown trajectory group."
        ) from error
    trajectory = group.get("trajectory")
    views = group.get("views")
    if not isinstance(trajectory, Mapping) or not isinstance(views, list):
        raise TypeError("Court trajectory group semantics are incomplete.")
    matching_views = [
        view
        for view in views
        if isinstance(view, Mapping) and view["view_id"] == view_id
    ]
    if (
        trajectory.get("trajectory_id") != trajectory_id
        or group.get("split") != split
        or group.get("shard_id") != shard_id
        or len(matching_views) != 1
    ):
        raise ValueError("Court sample semantics disagree with its trajectory group.")
    sample_count = _mapping_integer(group, "sample_count", minimum=8)
    if trajectory_frame_index >= sample_count:
        raise ValueError("Court sample frame index exceeds its trajectory group.")

    camera = SceneCamera.from_dict(record.get("camera"))
    if (
        camera.camera_id != sample_id
        or camera.source_frame_index != sample_index
        or camera.width != width
        or camera.height != height
    ):
        raise ValueError(
            "Court embedded camera disagrees with sample identity/resolution."
        )
    projection = record.get("projection")
    if projection is None:
        reasons = record["reasons"]
        if (
            not _uses_resolved_target_version(definition.version)
            or not isinstance(reasons, list)
            or published_court_geometry is None
        ):
            raise TypeError("Court sample projection must be a mapping.")
        _validate_published_ambiguity_reason(
            reasons,
            camera=camera,
            published_court_geometry=published_court_geometry,
        )
    elif not isinstance(projection, Mapping):
        raise TypeError("Court sample projection must be a mapping.")
    elif projection.get("camera_id") != sample_id or projection.get("resolution") != [
        width,
        height,
    ]:
        raise ValueError("Court projection disagrees with sample identity/resolution.")

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping) or list(metadata) != list(metadata_fields):
        raise ValueError("Court sample metadata does not match metadata_fields.")
    if not isinstance(profile, str) or not profile:
        raise TypeError("Court dataset profile must be a non-empty string.")
    expected_camera_parameters = {
        "view_id": view_id,
        "camera_center_scene_m": camera.camera_to_scene.matrix()[:3, 3].tolist(),
        "intrinsics": list(camera.intrinsics),
        "camera_to_scene": camera.camera_to_scene.to_list(),
    }
    if not _uses_resolved_target_version(definition.version):
        target = group.get("target_court")
        if not isinstance(target, Mapping):
            raise TypeError("Court v1 trajectory group target is incomplete.")
        expected_target_court = target.get("court_instance_id")
        expected_candidate_id = target.get("candidate_id")
        expected_transform = target.get("scene_from_court")
        expected_seed = target.get("selection_seed")
    else:
        target = ResolvedTargetCourtV2.from_mapping(record["target_court"])
        trajectory_spec = (
            OrbitTrajectorySpecV4.from_mapping(trajectory)
            if definition.version is CourtDatasetSchemaVersion.V4
            else OrbitTrajectorySpec.from_mapping(trajectory)
        )
        policy = _validated_v2_target_policy(
            trajectory=trajectory_spec,
            value=group["target_court_policy"],
        )
        view = OrbitViewSpecV2.from_mapping(matching_views[0])
        if published_court_geometry is None:
            raise ValueError("Court v2 published geometry is unavailable.")
        try:
            published_transform = published_court_geometry[
                target.binding.court_instance_id
            ]
        except KeyError as error:
            raise ValueError(
                "Court v2 target references unpublished court geometry."
            ) from error
        if not np.allclose(
            target.binding.scene_from_court.matrix(),
            published_transform.matrix(),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "Court v2 target binding disagrees with published court geometry."
            )
        if target.resolution_policy is not policy.mode:
            raise ValueError("Court v2 sample target policy disagrees with its group.")
        if (
            policy.centre_court_instance_id is not None
            and target.binding.court_instance_id != policy.centre_court_instance_id
        ):
            raise ValueError("Court v2 fixed target disagrees with its centre court.")
        expected_target_court = target.binding.court_instance_id
        expected_candidate_id = target.binding.candidate_id
        expected_transform = target.binding.scene_from_court.to_list()
        expected_seed = target.binding.selection_seed
        validate_camera_looks_at_resolved_binding(
            camera=camera,
            target_court=target,
            look_at_height_m=view.look_at_height_m,
        )
        if definition.version is CourtDatasetSchemaVersion.V4:
            safety = TrajectorySafetyEvaluation.from_mapping(group["safety_evaluation"])
            semantic_phase = TrajectorySemanticPhaseEvaluation.from_mapping(
                group["semantic_phase_evaluation"]
            )
            if (
                record.get("safety_support_input_digest") != safety.support_input_digest
                or not safety.safe
                or record.get("semantic_phase_index") != semantic_phase.phase_index
                or record.get("semantic_phase_disposition_digest")
                != semantic_phase.disposition_digest
            ):
                raise ValueError(
                    "Court V4 sample safety/semantic-phase authority disagrees with its group."
                )
    expected_metadata = {
        "target_court": expected_target_court,
        "candidate_id": expected_candidate_id,
        "transform": expected_transform,
        "camera_profile": profile,
        "camera_parameters": expected_camera_parameters,
        "seed": expected_seed,
    }
    if dict(metadata) != {
        field: expected_metadata[field]
        for field in metadata_fields
        if field in expected_metadata
    } or any(field not in expected_metadata for field in metadata_fields):
        raise ValueError(
            "Court sample metadata disagrees with canonical semantic sources."
        )
    return sample_id, sample_index


def _validate_published_sample(
    root: Path,
    record: Mapping[str, object],
    *,
    array_validation: CourtArrayValidationMode,
    definition: CourtSchemaDefinition,
) -> None:
    width = _record_integer(record, "width", minimum=2)
    height = _record_integer(record, "height", minimum=2)
    arrays = (
        ("rgb", (height, width, 3), True, False),
        ("alpha", (height, width, 1), True, False),
        ("depth", (height, width, 1), False, True),
    )
    loaded_arrays: dict[str, NDArray[np.float32]] = {}
    for field, shape, unit_range, nonnegative in arrays:
        if field not in record:
            raise TypeError(f"Court sample {field} path is missing.")
        value = record[field]
        if not isinstance(value, str):
            raise TypeError(f"Court sample {field} path must be a string.")
        path = _contained_file(root, value)
        array = np.load(
            path,
            allow_pickle=False,
            mmap_mode=(
                "r"
                if array_validation is CourtArrayValidationMode.HEADERS_ONLY
                else None
            ),
        )
        if array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"Court sample {field} array is semantically invalid.")
        if array_validation is CourtArrayValidationMode.FULL:
            loaded_arrays[field] = np.asarray(array, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"Court sample {field} array is semantically invalid.")
            if unit_range and (np.any(array < 0.0) or np.any(array > 1.0)):
                raise ValueError(f"Court sample {field} must stay in [0, 1].")
            if nonnegative and np.any(array < 0.0):
                raise ValueError("Court sample depth must be non-negative.")
    for field in ("rgb_preview", "alpha_preview"):
        if field not in record:
            raise TypeError(f"Court sample {field} path is missing.")
        value = record[field]
        if not isinstance(value, str):
            raise TypeError(f"Court sample {field} path must be a string.")
        path = _contained_file(root, value)
        if array_validation is CourtArrayValidationMode.FULL:
            try:
                with Image.open(path) as image:
                    size = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Court sample {field} is unreadable.") from error
            if size != (width, height):
                raise ValueError(f"Court sample {field} resolution is inconsistent.")
    if "labels" not in record:
        raise TypeError("Court labels path is missing.")
    labels = record["labels"]
    if not isinstance(labels, str):
        raise TypeError("Court labels path must be a string.")
    label_payload = load_json(_contained_file(root, labels))
    label_keys = {
        "schema",
        "sample_index",
        "sample_id",
        "trajectory_group_id",
        "trajectory_id",
        "view_id",
        "trajectory_frame_index",
        "split",
        "camera",
        "projection",
        "metadata",
    }
    if _uses_resolved_target_version(definition.version):
        label_keys.add("target_court")
    if definition.version is CourtDatasetSchemaVersion.V4:
        label_keys.update(
            {
                "safety_support_input_digest",
                "semantic_phase_index",
                "semantic_phase_disposition_digest",
            }
        )
    if (
        not isinstance(label_payload, Mapping)
        or set(label_payload) != label_keys
        or label_payload["schema"] != definition.sample_schema
    ):
        raise ValueError("Court sample labels schema is invalid.")
    for field in (
        "sample_index",
        "sample_id",
        "trajectory_group_id",
        "trajectory_id",
        "view_id",
        "trajectory_frame_index",
        "split",
        "camera",
        "projection",
        "metadata",
    ):
        if label_payload[field] != record[field]:
            raise ValueError(f"Court sample labels {field} mismatch.")
    if _uses_resolved_target_version(definition.version) and (
        label_payload["target_court"] != record["target_court"]
    ):
        raise ValueError("Court sample labels target_court mismatch.")
    if definition.version is CourtDatasetSchemaVersion.V4:
        for field in (
            "safety_support_input_digest",
            "semantic_phase_index",
            "semantic_phase_disposition_digest",
        ):
            if label_payload[field] != record[field]:
                raise ValueError(f"Court V4 sample labels {field} authority mismatch.")
    _require_finite_json(label_payload, name="labels")
    if array_validation is CourtArrayValidationMode.FULL:
        _validate_renderer_visibility_payload(
            label_payload["projection"],
            alpha=loaded_arrays["alpha"],
            depth=loaded_arrays["depth"],
            definition=definition,
        )


def _validate_renderer_visibility_payload(
    value: object,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
    definition: CourtSchemaDefinition | None = None,
) -> None:
    """Recompute every stored visibility bit from published renderer arrays."""
    if not isinstance(value, Mapping):
        raise TypeError("Court semantic projection must be a mapping.")
    courts = value.get("courts")
    if not isinstance(courts, list):
        raise TypeError("Court semantic projection courts must be a list.")
    schema_definition = definition or court_schema_for_version(
        CourtDatasetSchemaVersion.V1
    )
    visible_names: set[str] = set()
    visible_point_count = 0
    for court in courts:
        if not isinstance(court, Mapping) or not isinstance(court.get("classes"), list):
            raise TypeError("Court semantic projection class inventory is invalid.")
        for semantic_class in court["classes"]:
            if not isinstance(semantic_class, Mapping) or not isinstance(
                semantic_class.get("points"), list
            ):
                raise TypeError("Court semantic class point inventory is invalid.")
            class_name = semantic_class.get("class_name")
            if not isinstance(class_name, str):
                raise TypeError("Court semantic class name must be a string.")
            class_visible = False
            for point in semantic_class["points"]:
                if not isinstance(point, Mapping):
                    raise TypeError("Court semantic point must be a mapping.")
                in_frame = point.get("in_frame")
                uv = point.get("uv")
                if (
                    not isinstance(in_frame, bool)
                    or not isinstance(uv, list)
                    or len(uv) != 2
                ):
                    raise TypeError(
                        "Court semantic point visibility inputs are invalid."
                    )
                visible = False
                if in_frame:
                    x = int(round(_json_float(uv[0], name="point.uv")))
                    y = int(round(_json_float(uv[1], name="point.uv")))
                    x0 = max(0, x - 1)
                    x1 = min(alpha.shape[1], x + 2)
                    y0 = max(0, y - 1)
                    y1 = min(alpha.shape[0], y + 2)
                    visible = bool(
                        np.any(
                            (alpha[y0:y1, x0:x1, 0] >= 0.01)
                            & (depth[y0:y1, x0:x1, 0] > 0.0)
                        )
                    )
                if point.get("renderer_visible") is not visible:
                    raise ValueError(
                        "Court renderer-visible point disagrees with alpha/depth output."
                    )
                class_visible |= visible
                visible_point_count += int(visible)
            if semantic_class.get("renderer_visible") is not class_visible:
                raise ValueError(
                    "Court renderer-visible class disagrees with its point semantics."
                )
            if class_visible:
                visible_names.add(class_name)
    if value.get("visible_point_count") != visible_point_count:
        raise ValueError("Court renderer-visible point summary is inconsistent.")
    expected_names = [
        name for name in schema_definition.semantic_class_names if name in visible_names
    ]
    if value.get("visible_class_names") != expected_names:
        raise ValueError("Court renderer-visible class summary is inconsistent.")


def _validate_court_balance(groups: Sequence[Mapping[str, object]]) -> None:
    global_counts: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    all_courts: set[str] = set()
    for group in groups:
        if "target_court" not in group or "split" not in group:
            raise TypeError("Court group target/split metadata is missing.")
        target = group["target_court"]
        split = group["split"]
        if not isinstance(target, Mapping) or not isinstance(split, str):
            raise TypeError("Court group target/split metadata is invalid.")
        if "court_instance_id" not in target:
            raise TypeError("Court group target ID is missing.")
        court_id = target["court_instance_id"]
        if not isinstance(court_id, str):
            raise TypeError("Court group target ID must be a string.")
        all_courts.add(court_id)
        global_counts[court_id] += 1
        by_split[split][court_id] += 1
    if max(global_counts.values()) - min(global_counts.values()) > 1:
        raise ValueError("Court trajectory groups are not globally court-balanced.")
    for split, counts in by_split.items():
        values = [counts[court_id] for court_id in all_courts]
        if max(values) - min(values) > 1:
            raise ValueError(f"Court groups are not balanced within split {split!r}.")


def _validate_group_record(
    group: Mapping[str, object],
    *,
    definition: CourtSchemaDefinition,
) -> None:
    keys = {
        "trajectory",
        "center",
        "views",
        "split",
        "shard_id",
        "sample_count",
        "maximum_adjacent_step_m",
        "total_arc_length_m",
    }
    if not _uses_resolved_target_version(definition.version):
        keys.add("target_court")
    else:
        keys.add("target_court_policy")
    if definition.version is CourtDatasetSchemaVersion.V4:
        keys.update({"safety_evaluation", "semantic_phase_evaluation"})
    if set(group) != keys:
        raise ValueError("Court trajectory group schema is invalid.")
    trajectory = (
        OrbitTrajectorySpecV4.from_mapping(group["trajectory"])
        if definition.version is CourtDatasetSchemaVersion.V4
        else OrbitTrajectorySpec.from_mapping(group["trajectory"])
    )
    views = group["views"]
    if not isinstance(views, list) or not views:
        raise TypeError("Court group views must be a non-empty list.")
    for view in views:
        if not _uses_resolved_target_version(definition.version):
            OrbitViewSpec.from_mapping(view)
        else:
            OrbitViewSpecV2.from_mapping(view)
    center = OrbitCenter.from_mapping(group["center"])
    if center.key() != (
        trajectory.center_kind,
        trajectory.center_court_instance_id,
    ):
        raise ValueError("Court resolved centre disagrees with its trajectory.")
    if not _uses_resolved_target_version(definition.version):
        target = group["target_court"]
        target_keys = {
            "court_instance_id",
            "candidate_id",
            "scene_from_court",
            "selection_seed",
        }
        if not isinstance(target, Mapping) or set(target) != target_keys:
            raise ValueError("Court target binding schema is invalid.")
        target_transform = target["scene_from_court"]
        if not isinstance(target_transform, list):
            raise TypeError("Court target transform must be a list.")
        RigidTransform(
            tuple(
                _json_float(value, name="scene_from_court")
                for value in target_transform
            )
        )
    else:
        _validated_v2_target_policy(
            trajectory=trajectory,
            value=group["target_court_policy"],
        )
        if definition.version is CourtDatasetSchemaVersion.V4:
            safety = TrajectorySafetyEvaluation.from_mapping(group["safety_evaluation"])
            semantic_phase = TrajectorySemanticPhaseEvaluation.from_mapping(
                group["semantic_phase_evaluation"]
            )
            if (
                not safety.safe
                or safety.trajectory_id != trajectory.trajectory_id
                or safety.trajectory_group_id != trajectory.trajectory_group_id
                or semantic_phase.trajectory_id != trajectory.trajectory_id
                or semantic_phase.trajectory_group_id != trajectory.trajectory_group_id
                or tuple(OrbitViewSpecV2.from_mapping(view) for view in views)
                != (semantic_phase.view,)
                or semantic_phase.expected_frame_count
                != _mapping_integer(group, "sample_count", minimum=8)
            ):
                raise ValueError(
                    "Court V4 group safety/semantic-phase evaluation is invalid."
                )
    if group["split"] not in {"train", "validation", "test"}:
        raise ValueError("Court group split is invalid.")
    if not isinstance(group["shard_id"], str) or not group["shard_id"]:
        raise TypeError("Court group shard_id must be non-empty.")
    _mapping_integer(group, "sample_count", minimum=8)
    _mapping_float(group, "maximum_adjacent_step_m")
    _mapping_float(group, "total_arc_length_m")


def _serialized_selected_coverage(
    groups: Sequence[Mapping[str, object]],
    *,
    required_coverage: RequiredTrajectoryCoverage,
) -> SelectedTrajectoryCoverage:
    """Recompute selected coverage from the immutable serialized groups."""
    records: list[tuple[OrbitTrajectorySpecV4, OrbitTargetMode, int]] = []
    for group in groups:
        trajectory = OrbitTrajectorySpecV4.from_mapping(group.get("trajectory"))
        views = group.get("views")
        if not isinstance(views, list) or len(views) != 1:
            raise ValueError("V4 selected coverage requires one serialized group view.")
        view = OrbitViewSpecV2.from_mapping(views[0])
        sample_count = group.get("sample_count")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
        ):
            raise ValueError("V4 selected coverage group sample_count is invalid.")
        records.append((trajectory, view.target_mode, sample_count))
    return build_selected_coverage_from_records(
        records,
        required_raised_lift_m=required_coverage.required_raised_lift_m,
    )


def _nested_group_id(group: Mapping[str, object]) -> str:
    trajectory = group.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise TypeError("Court group trajectory must be a mapping.")
    group_id = trajectory.get("trajectory_group_id")
    if not isinstance(group_id, str) or not group_id:
        raise TypeError("trajectory_group_id must be a non-empty string.")
    return group_id


def _validate_diagnostic_schemas(
    root: Path,
    *,
    definition: CourtSchemaDefinition,
    dataset: Mapping[str, object],
    validate_support_occupancy: bool,
) -> None:
    """Require every versioned diagnostic to use the selected exact schema."""
    expected = {
        "trajectory-plan.json": definition.plan_schema,
        "arc-step-distribution.json": definition.arc_step_diagnostics_schema,
        "acceptance.json": definition.acceptance_diagnostics_schema,
        "splits.json": definition.split_diagnostics_schema,
        "parameter-table.json": definition.parameter_table_schema,
        "semantic-visibility.json": definition.semantic_visibility_diagnostics_schema,
        "semantic-manifest.json": definition.semantic_manifest_schema,
        "performance.json": definition.performance_schema,
    }
    if definition.version is CourtDatasetSchemaVersion.V4:
        if definition.safety_diagnostics_schema is None:
            raise RuntimeError("Court V4 safety diagnostics schema is unavailable.")
        expected["trajectory-safety.json"] = definition.safety_diagnostics_schema
    payloads: dict[str, object] = {}
    for filename, schema in expected.items():
        payload = load_json(_contained_file(root, f"diagnostics/{filename}"))
        payloads[filename] = payload
        if not isinstance(payload, Mapping) or payload.get("schema") != schema:
            raise ValueError(
                f"Court diagnostic {filename} schema disagrees with dataset schema."
            )
    _validate_trajectory_plan_diagnostic(
        payloads["trajectory-plan.json"],
        dataset=dataset,
        definition=definition,
        require_support_occupancy_identity=validate_support_occupancy,
    )
    _validate_acceptance_diagnostic(
        payloads["acceptance.json"],
        dataset=dataset,
        definition=definition,
    )
    _validate_arc_step_diagnostic(
        payloads["arc-step-distribution.json"],
        dataset=dataset,
        definition=definition,
    )
    _validate_split_diagnostic(
        payloads["splits.json"],
        dataset=dataset,
        trajectory_plan=payloads["trajectory-plan.json"],
        definition=definition,
    )
    _validate_parameter_table_diagnostic(
        payloads["parameter-table.json"],
        dataset=dataset,
        definition=definition,
    )
    _validate_semantic_visibility_diagnostic(
        payloads["semantic-visibility.json"],
        dataset=dataset,
        definition=definition,
    )
    if definition.version is CourtDatasetSchemaVersion.V4:
        _validate_safety_diagnostic(
            payloads["trajectory-safety.json"],
            dataset=dataset,
            trajectory_plan=payloads["trajectory-plan.json"],
            definition=definition,
        )
        if validate_support_occupancy:
            trajectory_plan = cast(
                Mapping[str, object],
                payloads["trajectory-plan.json"],
            )
            support_policy = TrajectorySupportPolicy.from_mapping(
                trajectory_plan["support_policy"]
            )
            support_summary = SupportModelSummary.from_mapping(
                trajectory_plan["support_summary"]
            )
            occupancy_identity = CourtV4SupportOccupancyIdentity.from_mapping(
                trajectory_plan["support_occupancy_identity"]
            )
            published_occupancy = load_court_v4_support_occupancy(
                root,
                expected_scene_id=cast(str, dataset["scene_id"]),
                expected_profile=cast(str, dataset["profile"]),
                expected_policy_decision_id=support_policy.decision_id,
                expected_support_input_digest=support_summary.input_digest,
                expected_voxel_size_m=support_policy.occupancy_voxel_size_m,
                expected_cell_count=support_summary.inflated_occupancy_cell_count,
                expected_content_digest=occupancy_identity.content_digest,
                maximum_cells=support_policy.maximum_occupancy_cells,
            )
            if published_occupancy.snapshot.identity != occupancy_identity:
                raise ValueError(
                    "Court V4 occupancy artifact disagrees with plan identity."
                )
    points_path = _contained_file(root, "diagnostics/sample-points.npy")
    points = np.load(points_path, allow_pickle=False, mmap_mode="r")
    proposal_count = _mapping_integer(
        cast(Mapping[str, object], dataset["metrics"]),
        "proposal_count",
        minimum=1,
    )
    if points.dtype != np.float32 or points.shape != (proposal_count, 3):
        raise ValueError("Court sample-point diagnostics are invalid.")


def _validate_arc_step_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    definition: CourtSchemaDefinition,
) -> None:
    """Cross-check every arc-step field against the published plan semantics."""
    keys = {
        "schema",
        "policy_maximum_m",
        "observed_maximum_m",
        "observed_quantiles_m",
        "groups",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court arc-step diagnostic schema is invalid.")
    if value["schema"] != definition.arc_step_diagnostics_schema:
        raise ValueError("Court arc-step diagnostic version is invalid.")
    policy = dataset["sampling_policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("Court sampling policy must be a mapping.")
    groups = _mapping_sequence(
        dataset["trajectory_groups"],
        name="trajectory_groups",
    )
    expected_groups: dict[str, dict[str, object]] = {}
    maximum_steps: list[float] = []
    for group in groups:
        group_id = _nested_group_id(group)
        sample_count = _mapping_integer(group, "sample_count", minimum=8)
        total_arc_length_m = _mapping_float(group, "total_arc_length_m")
        maximum_adjacent_step_m = _mapping_float(
            group,
            "maximum_adjacent_step_m",
        )
        expected_groups[group_id] = {
            "sample_count": sample_count,
            "total_arc_length_m": total_arc_length_m,
            "maximum_adjacent_step_m": maximum_adjacent_step_m,
            "mean_arc_step_m": total_arc_length_m / sample_count,
        }
        maximum_steps.append(maximum_adjacent_step_m)
    raw_groups = value["groups"]
    if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(expected_groups):
        raise ValueError("Court arc-step group inventory is invalid.")
    for group_id, expected_group in expected_groups.items():
        raw_group = raw_groups[group_id]
        if not isinstance(raw_group, Mapping) or set(raw_group) != set(expected_group):
            raise ValueError("Court arc-step group schema is invalid.")
        _mapping_integer(raw_group, "sample_count", minimum=8)
        for field in (
            "total_arc_length_m",
            "maximum_adjacent_step_m",
            "mean_arc_step_m",
        ):
            _mapping_float(raw_group, field)
        if dict(raw_group) != expected_group:
            raise ValueError("Court arc-step group semantics are inconsistent.")
    quantile_names = {"minimum", "p10", "median", "p90", "maximum"}
    raw_quantiles = value["observed_quantiles_m"]
    if not isinstance(raw_quantiles, Mapping) or set(raw_quantiles) != quantile_names:
        raise ValueError("Court arc-step quantile schema is invalid.")
    for name in quantile_names:
        _mapping_float(raw_quantiles, name)
    maximum_array = np.asarray(maximum_steps, dtype=np.float64)
    quantiles = np.asarray(
        np.quantile(maximum_array, (0.0, 0.1, 0.5, 0.9, 1.0)),
        dtype=np.float64,
    ).reshape(5)
    expected_quantiles = {
        name: float(quantiles[index])
        for index, name in enumerate(("minimum", "p10", "median", "p90", "maximum"))
    }
    policy_maximum_m = _mapping_float(value, "policy_maximum_m")
    observed_maximum_m = _mapping_float(value, "observed_maximum_m")
    if (
        policy_maximum_m != _mapping_float(policy, "max_arc_step_m")
        or observed_maximum_m != max(maximum_steps)
        or dict(raw_quantiles) != expected_quantiles
    ):
        raise ValueError("Court arc-step diagnostic disagrees with the dataset plan.")


def _validate_split_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    trajectory_plan: object,
    definition: CourtSchemaDefinition,
) -> None:
    """Validate exact v1/v2 split records and v2 target-resolution evidence."""
    keys = {"schema", "groups"}
    if _uses_resolved_target_version(definition.version):
        keys.add("target_resolution")
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court split diagnostic schema is invalid.")
    if value["schema"] != definition.split_diagnostics_schema:
        raise ValueError("Court split diagnostic version is invalid.")
    groups = _mapping_sequence(
        dataset["trajectory_groups"],
        name="trajectory_groups",
    )
    expected_groups: dict[str, dict[str, object]] = {}
    for group in groups:
        group_id = _nested_group_id(group)
        expected_group: dict[str, object] = {
            "split": group["split"],
            "shard_id": group["shard_id"],
        }
        if not _uses_resolved_target_version(definition.version):
            target = group["target_court"]
            if not isinstance(target, Mapping):
                raise TypeError("Court v1 group target must be a mapping.")
            expected_group["target_court_instance_id"] = target["court_instance_id"]
        else:
            policy = TargetCourtPolicyV2.from_mapping(group["target_court_policy"])
            expected_group["target_court_policy"] = policy.to_dict()
        expected_groups[group_id] = expected_group
    raw_groups = value["groups"]
    if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(expected_groups):
        raise ValueError("Court split group inventory is invalid.")
    for group_id, expected_group in expected_groups.items():
        raw_group = raw_groups[group_id]
        if not isinstance(raw_group, Mapping) or set(raw_group) != set(expected_group):
            raise ValueError("Court split group schema is invalid.")
        if not isinstance(raw_group["split"], str) or not isinstance(
            raw_group["shard_id"], str
        ):
            raise TypeError("Court split group identifiers must be strings.")
        if not _uses_resolved_target_version(definition.version):
            if not isinstance(raw_group["target_court_instance_id"], str):
                raise TypeError("Court split target court ID must be a string.")
        else:
            TargetCourtPolicyV2.from_mapping(raw_group["target_court_policy"])
        if dict(raw_group) != expected_group:
            raise ValueError("Court split group semantics are inconsistent.")
    if _uses_resolved_target_version(definition.version):
        _validate_v2_target_resolution_diagnostic(
            value["target_resolution"],
            dataset=dataset,
            groups=groups,
            trajectory_plan=trajectory_plan,
            version=definition.version,
        )


def _validate_v2_target_resolution_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    groups: Sequence[Mapping[str, object]],
    trajectory_plan: object,
    version: CourtDatasetSchemaVersion,
) -> None:
    """Recompute every public v2 target-resolution inventory available in the plan."""
    keys = {
        "sample_counts_by_target_court",
        "target_switch_counts_by_trajectory",
        "resolution_policy_counts",
        "nearest_court_tie_count",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court v2 target-resolution diagnostic schema is invalid.")
    if not isinstance(trajectory_plan, Mapping):
        raise TypeError("Court trajectory plan diagnostic must be a mapping.")
    planned = _mapping_sequence(
        trajectory_plan["samples"],
        name="planned samples",
    )
    target_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    tie_count = 0
    samples_by_variant: dict[
        tuple[str, str],
        list[tuple[int, str]],
    ] = defaultdict(list)
    court_centers = _court_centers_from_published_projection(dataset)
    policy_by_group: dict[str, TargetCourtPolicyV2] = {}
    for group in groups:
        trajectory = (
            OrbitTrajectorySpecV4.from_mapping(group["trajectory"])
            if version is CourtDatasetSchemaVersion.V4
            else OrbitTrajectorySpec.from_mapping(group["trajectory"])
        )
        policy_by_group[_nested_group_id(group)] = _validated_v2_target_policy(
            trajectory=trajectory,
            value=group["target_court_policy"],
        )
    for sample in planned:
        target = ResolvedTargetCourtV2.from_mapping(sample.get("target_court"))
        group_id = sample.get("trajectory_group_id")
        view_id = sample.get("view_id")
        if not isinstance(group_id, str) or not isinstance(view_id, str):
            raise TypeError("Court planned sample group/view IDs must be strings.")
        frame_index = _record_integer(
            sample,
            "trajectory_frame_index",
            minimum=0,
        )
        camera_center = sample.get("camera_center_scene_m")
        if not isinstance(camera_center, list) or len(camera_center) != 3:
            raise TypeError("Court planned camera centre must be a three-vector.")
        camera_center_array = np.asarray(
            [
                _json_float(coordinate, name="camera_center_scene_m")
                for coordinate in camera_center
            ],
            dtype=np.float64,
        )
        court_id = target.binding.court_instance_id
        if court_id not in court_centers:
            raise ValueError("Court v2 target references an unpublished court.")
        policy = policy_by_group[group_id]
        if target.resolution_policy is not policy.mode:
            raise ValueError("Court v2 target policy disagrees with its group.")
        distances = {
            candidate_id: float(np.linalg.norm(camera_center_array - center))
            for candidate_id, center in court_centers.items()
        }
        minimum_distance = min(distances.values())
        tied_ids = tuple(
            sorted(
                candidate_id
                for candidate_id, distance in distances.items()
                if distance <= minimum_distance + NEAREST_COURT_TIE_TOLERANCE_M
            )
        )
        if policy.centre_court_instance_id is not None:
            expected_court_id = policy.centre_court_instance_id
        else:
            expected_court_id = tied_ids[0]
            tie_count += int(len(tied_ids) > 1)
        if court_id != expected_court_id or not math.isclose(
            target.camera_to_court_center_distance_m,
            distances[court_id],
            abs_tol=NEAREST_COURT_TIE_TOLERANCE_M,
            rel_tol=0.0,
        ):
            raise ValueError("Court v2 target-resolution geometry is inconsistent.")
        target_counts[court_id] += 1
        policy_counts[target.resolution_policy.value] += 1
        samples_by_variant[(group_id, view_id)].append((frame_index, court_id))
    switch_counts: dict[str, int] = {}
    for group in groups:
        group_id = _nested_group_id(group)
        views = _mapping_sequence(group["views"], name="group views")
        switches = 0
        for view_record in views:
            view_id = view_record.get("view_id")
            if not isinstance(view_id, str):
                raise TypeError("Court view_id must be a string.")
            variant = sorted(samples_by_variant[(group_id, view_id)])
            target_ids = [court_id for _index, court_id in variant]
            switches += sum(
                current != previous
                for previous, current in zip(
                    target_ids,
                    target_ids[1:],
                    strict=False,
                )
            )
        switch_counts[group_id] = switches
    _require_exact_count_mapping(
        value["sample_counts_by_target_court"],
        expected=dict(sorted(target_counts.items())),
        name="target_resolution.sample_counts_by_target_court",
    )
    _require_exact_count_mapping(
        value["target_switch_counts_by_trajectory"],
        expected=dict(sorted(switch_counts.items())),
        name="target_resolution.target_switch_counts_by_trajectory",
    )
    _require_exact_count_mapping(
        value["resolution_policy_counts"],
        expected=dict(sorted(policy_counts.items())),
        name="target_resolution.resolution_policy_counts",
    )
    if _mapping_integer(value, "nearest_court_tie_count", minimum=0) != tie_count:
        raise ValueError("Court nearest-court tie count is inconsistent.")


def _court_centers_from_published_projection(
    dataset: Mapping[str, object],
) -> dict[str, NDArray[np.float64]]:
    """Recover accepted court centres from the published fourteen-point geometry."""
    samples = _mapping_sequence(dataset.get("samples"), name="samples")
    if not samples:
        raise ValueError("Court v2 diagnostics require an accepted sample.")
    projection = samples[0].get("projection")
    if not isinstance(projection, Mapping):
        raise TypeError("Court accepted projection must be a mapping.")
    courts = _mapping_sequence(projection.get("courts"), name="projected courts")
    centers: dict[str, NDArray[np.float64]] = {}
    for court in courts:
        court_id = court.get("court_instance_id")
        if not isinstance(court_id, str) or not court_id or court_id in centers:
            raise ValueError("Court projected court inventory is invalid.")
        classes = _mapping_sequence(
            court.get("classes"),
            name="projected court classes",
        )
        points_by_index: dict[int, NDArray[np.float64]] = {}
        for semantic_class in classes:
            points = _mapping_sequence(
                semantic_class.get("points"),
                name="projected semantic points",
            )
            for point in points:
                physical_index = _record_integer(
                    point,
                    "physical_index",
                    minimum=0,
                )
                scene_xyz = point.get("scene_xyz_m")
                if not isinstance(scene_xyz, list) or len(scene_xyz) != 3:
                    raise TypeError(
                        "Court projected scene point must be a three-vector."
                    )
                if physical_index in points_by_index:
                    raise ValueError("Court projected physical indices are duplicated.")
                points_by_index[physical_index] = np.asarray(
                    [
                        _json_float(coordinate, name="scene_xyz_m")
                        for coordinate in scene_xyz
                    ],
                    dtype=np.float64,
                )
        if set(points_by_index) != set(range(14)):
            raise ValueError("Court projected physical index inventory is incomplete.")
        centers[court_id] = np.mean(
            np.asarray(
                [points_by_index[index] for index in range(14)],
                dtype=np.float64,
            ),
            axis=0,
        )
    if not centers:
        raise ValueError("Court projected court inventory must not be empty.")
    return centers


def _validate_parameter_table_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    definition: CourtSchemaDefinition,
) -> None:
    """Cross-check the exact versioned parameter rows against trajectory groups."""
    keys = {"schema", "rows"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court parameter-table diagnostic schema is invalid.")
    if value["schema"] != definition.parameter_table_schema:
        raise ValueError("Court parameter-table diagnostic version is invalid.")
    groups = _mapping_sequence(
        dataset["trajectory_groups"],
        name="trajectory_groups",
    )
    rows = _mapping_sequence(value["rows"], name="parameter rows")
    if len(rows) != len(groups):
        raise ValueError("Court parameter-table group inventory is incomplete.")
    for row, group in zip(rows, groups, strict=True):
        trajectory = group["trajectory"]
        views = _mapping_sequence(group["views"], name="group views")
        if not isinstance(trajectory, Mapping):
            raise TypeError("Court group trajectory must be a mapping.")
        expected_row: dict[str, object] = {
            **trajectory,
            "view_ids": [view["view_id"] for view in views],
            "split": group["split"],
            "shard_id": group["shard_id"],
            "sample_count_per_view": group["sample_count"],
        }
        if not _uses_resolved_target_version(definition.version):
            target = group["target_court"]
            if not isinstance(target, Mapping):
                raise TypeError("Court v1 group target must be a mapping.")
            expected_row.update(
                {
                    "target_court_instance_id": target["court_instance_id"],
                    "candidate_id": target["candidate_id"],
                }
            )
        else:
            policy = TargetCourtPolicyV2.from_mapping(group["target_court_policy"])
            expected_row["target_court_policy"] = policy.to_dict()
        if set(row) != set(expected_row):
            raise ValueError("Court parameter-table row schema is invalid.")
        trajectory_row = {key: row[key] for key in trajectory}
        (
            OrbitTrajectorySpecV4.from_mapping(trajectory_row)
            if definition.version is CourtDatasetSchemaVersion.V4
            else OrbitTrajectorySpec.from_mapping(trajectory_row)
        )
        view_ids = row["view_ids"]
        if not isinstance(view_ids, list) or any(
            not isinstance(view_id, str) for view_id in view_ids
        ):
            raise TypeError("Court parameter-table view_ids must be a string list.")
        if not isinstance(row["split"], str) or not isinstance(row["shard_id"], str):
            raise TypeError("Court parameter-table split/shard must be strings.")
        _mapping_integer(row, "sample_count_per_view", minimum=8)
        if not _uses_resolved_target_version(definition.version):
            if not isinstance(row["target_court_instance_id"], str) or not isinstance(
                row["candidate_id"], str
            ):
                raise TypeError("Court parameter-table target IDs must be strings.")
        else:
            TargetCourtPolicyV2.from_mapping(row["target_court_policy"])
        if dict(row) != expected_row:
            raise ValueError("Court parameter-table row semantics are inconsistent.")


def _validate_semantic_visibility_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    definition: CourtSchemaDefinition,
) -> None:
    """Validate the exact semantic-class inventory and its published counts."""
    keys = {
        "schema",
        "renderer_visible_points_by_class",
        "all_classes_visible",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court semantic-visibility diagnostic schema is invalid.")
    if value["schema"] != definition.semantic_visibility_diagnostics_schema:
        raise ValueError("Court semantic-visibility diagnostic version is invalid.")
    metrics = dataset["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("Court dataset metrics must be a mapping.")
    visible = metrics["renderer_visible_points_by_class"]
    if not isinstance(visible, Mapping):
        raise TypeError("Court renderer visibility metrics must be a mapping.")
    expected = {
        name: cast(int, visible[name]) for name in definition.semantic_class_names
    }
    _require_exact_count_mapping(
        value["renderer_visible_points_by_class"],
        expected=expected,
        name="semantic visibility",
    )
    all_classes_visible = value["all_classes_visible"]
    if not isinstance(all_classes_visible, bool):
        raise TypeError("Court all_classes_visible must be a boolean.")
    if all_classes_visible is not all(count > 0 for count in expected.values()):
        raise ValueError("Court semantic visibility summary is inconsistent.")


def _validate_acceptance_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    definition: CourtSchemaDefinition,
) -> None:
    """Cross-check the exact versioned acceptance payload and dispositions."""
    keys = {
        "schema",
        "proposal_count",
        "accepted_count",
        "rejected_count",
        "accepted_fraction",
        "accepted_sample_ids",
        "rejected",
        "coverage_counts",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court acceptance diagnostic schema is invalid.")
    if value["schema"] != definition.acceptance_diagnostics_schema:
        raise ValueError("Court acceptance diagnostic version is invalid.")
    metrics = dataset["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("Court dataset metrics must be a mapping.")
    proposal_count = _mapping_integer(value, "proposal_count", minimum=1)
    accepted_count = _mapping_integer(value, "accepted_count", minimum=1)
    rejected_count = _mapping_integer(value, "rejected_count", minimum=0)
    accepted_fraction = _mapping_float(value, "accepted_fraction")
    accepted = _mapping_sequence(dataset["samples"], name="samples")
    rejected = _mapping_sequence(dataset["rejected_samples"], name="rejected_samples")
    accepted_sample_ids = value["accepted_sample_ids"]
    if not isinstance(accepted_sample_ids, list) or any(
        not isinstance(sample_id, str) for sample_id in accepted_sample_ids
    ):
        raise TypeError("Court accepted diagnostic IDs must be a string list.")
    metric_coverage = metrics["coverage_counts"]
    if not isinstance(metric_coverage, Mapping) or any(
        not isinstance(name, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in metric_coverage.items()
    ):
        raise TypeError("Court metric coverage counts are invalid.")
    expected_coverage = {name: count for name, count in metric_coverage.items()}
    _require_exact_count_mapping(
        value["coverage_counts"],
        expected=expected_coverage,
        name="acceptance coverage_counts",
    )
    if (
        proposal_count != metrics["proposal_count"]
        or accepted_count != metrics["accepted_frame_count"]
        or rejected_count != metrics["rejected_frame_count"]
        or accepted_fraction != metrics["accepted_fraction"]
        or accepted_sample_ids != [record["sample_id"] for record in accepted]
        or value["rejected"] != list(rejected)
    ):
        raise ValueError(
            "Court acceptance diagnostic disagrees with the dataset inventory."
        )


def _validate_metric_inventories(
    metrics: Mapping[str, object],
    *,
    groups: Sequence[Mapping[str, object]],
    samples: Sequence[Mapping[str, object]],
    split_counts: Mapping[str, int],
    definition: CourtSchemaDefinition,
) -> None:
    """Recompute every version-specific Court count inventory."""
    expected_split_frames = dict(sorted(split_counts.items()))
    expected_split_groups = dict(
        sorted(Counter(str(group["split"]) for group in groups).items())
    )
    _require_exact_count_mapping(
        metrics["split_frame_counts"],
        expected=expected_split_frames,
        name="split_frame_counts",
    )
    _require_exact_count_mapping(
        metrics["split_group_counts"],
        expected=expected_split_groups,
        name="split_group_counts",
    )
    court_counts: Counter[str] = Counter()
    split_court_counts: dict[str, Counter[str]] = defaultdict(Counter)
    if not _uses_resolved_target_version(definition.version):
        for group in groups:
            target = group["target_court"]
            if not isinstance(target, Mapping):
                raise TypeError("Court v1 group target must be a mapping.")
            court_id = target["court_instance_id"]
            split = group["split"]
            if not isinstance(court_id, str) or not isinstance(split, str):
                raise TypeError("Court v1 group target/split inventory is invalid.")
            court_counts[court_id] += 1
            split_court_counts[split][court_id] += 1
        count_key = "court_group_counts"
        split_count_key = "split_court_group_counts"
    else:
        for sample in samples:
            target = sample["target_court"]
            if not isinstance(target, Mapping):
                raise TypeError("Court v2 sample target must be a mapping.")
            binding = target["binding"]
            split = sample["split"]
            if not isinstance(binding, Mapping) or not isinstance(split, str):
                raise TypeError("Court v2 sample target/split inventory is invalid.")
            court_id = binding["court_instance_id"]
            if not isinstance(court_id, str):
                raise TypeError("Court v2 sample target ID must be a string.")
            court_counts[court_id] += 1
            split_court_counts[split][court_id] += 1
        count_key = "court_sample_counts"
        split_count_key = "split_court_sample_counts"
    expected_court_counts = dict(sorted(court_counts.items()))
    expected_split_court_counts = {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(split_court_counts.items())
    }
    _require_exact_count_mapping(
        metrics[count_key],
        expected=expected_court_counts,
        name=count_key,
    )
    _require_exact_nested_count_mapping(
        metrics[split_count_key],
        expected=expected_split_court_counts,
        name=split_count_key,
    )


def _validate_metric_schema(
    metrics: Mapping[str, object],
    *,
    definition: CourtSchemaDefinition,
) -> None:
    """Require the exact metric fields owned by the selected dataset version."""
    expected_keys = _COURT_METRIC_KEYS_BY_VERSION[definition.version]
    if set(metrics) != expected_keys:
        raise ValueError(
            "Court metric schema contains missing, mixed-version, or unexpected fields."
        )


def _require_exact_count_mapping(
    value: object,
    *,
    expected: Mapping[str, int],
    name: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(expected)
        or any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in value.items()
        )
    ):
        raise ValueError(f"Court {name} metric inventory is invalid.")
    if dict(value) != dict(expected):
        raise ValueError(f"Court {name} metric inventory is inconsistent.")


def _require_exact_nested_count_mapping(
    value: object,
    *,
    expected: Mapping[str, Mapping[str, int]],
    name: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(expected)
        or any(not isinstance(key, str) for key in value)
    ):
        raise ValueError(f"Court {name} metric inventory is invalid.")
    for split, expected_counts in expected.items():
        _require_exact_count_mapping(
            value[split],
            expected=expected_counts,
            name=f"{name}.{split}",
        )


def _validate_trajectory_plan_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    definition: CourtSchemaDefinition,
    require_support_occupancy_identity: bool,
) -> None:
    """Cross-check the exact versioned plan against published dispositions."""
    keys = {"schema", "scene_id", "profile", "policy", "groups", "samples"}
    if definition.version is CourtDatasetSchemaVersion.V4:
        keys.update(
            {
                "support_policy",
                "support_summary",
                "candidate_safety_evaluations",
                "candidate_semantic_phase_evaluations",
                "semantic_phase_inventory_digest",
                "projected_semantic_valid_frame_count",
                "projected_semantic_valid_fraction",
                "required_coverage",
                "selected_coverage",
                "required_coverage_shortfall",
                "optional_candidate_coverage_shortfall",
            }
        )
        if require_support_occupancy_identity:
            keys.add("support_occupancy_identity")
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court trajectory-plan diagnostic schema is invalid.")
    plan_policy = _parse_canonical_sampling_policy(
        value["policy"], version=definition.version
    )
    dataset_policy = _parse_canonical_sampling_policy(
        dataset["sampling_policy"], version=definition.version
    )
    if (
        value["schema"] != definition.plan_schema
        or value["scene_id"] != dataset["scene_id"]
        or value["profile"] != dataset["profile"]
        or plan_policy.to_dict() != dataset_policy.to_dict()
        or value["groups"] != dataset["trajectory_groups"]
    ):
        raise ValueError("Court trajectory-plan diagnostic disagrees with dataset.")
    if definition.version is CourtDatasetSchemaVersion.V4:
        support_policy = TrajectorySupportPolicy.from_mapping(value["support_policy"])
        support_summary = SupportModelSummary.from_mapping(value["support_summary"])
        occupancy_identity = (
            CourtV4SupportOccupancyIdentity.from_mapping(
                value["support_occupancy_identity"]
            )
            if require_support_occupancy_identity
            else None
        )
        if occupancy_identity is not None and (
            occupancy_identity.coordinate_space != support_summary.coordinate_space
            or occupancy_identity.voxel_size_m
            != support_policy.occupancy_voxel_size_m
            or occupancy_identity.cell_count
            != support_summary.inflated_occupancy_cell_count
            or occupancy_identity.support_input_digest != support_summary.input_digest
            or occupancy_identity.policy_decision_id != support_policy.decision_id
        ):
            raise ValueError(
                "Court V4 trajectory-plan occupancy identity is inconsistent."
            )
        candidates = _mapping_sequence(
            value["candidate_safety_evaluations"],
            name="candidate_safety_evaluations",
        )
        evaluations = tuple(
            TrajectorySafetyEvaluation.from_mapping(item) for item in candidates
        )
        semantic_candidates = _mapping_sequence(
            value["candidate_semantic_phase_evaluations"],
            name="candidate_semantic_phase_evaluations",
        )
        semantic_evaluations = tuple(
            TrajectorySemanticPhaseEvaluation.from_mapping(item)
            for item in semantic_candidates
        )
        id_pairs = tuple(
            (item.trajectory_id, item.trajectory_group_id) for item in evaluations
        )
        expected_pairs = tuple(
            (f"trajectory-{index:05d}", f"group-{index:05d}")
            for index in range(len(evaluations))
        )
        required_coverage = RequiredTrajectoryCoverage.from_mapping(
            value["required_coverage"]
        )
        selected_coverage = SelectedTrajectoryCoverage.from_mapping(
            value["selected_coverage"]
        )
        groups = _mapping_sequence(value["groups"], name="trajectory groups")
        recomputed_coverage = _serialized_selected_coverage(
            groups,
            required_coverage=required_coverage,
        )
        required_shortfall = value["required_coverage_shortfall"]
        optional_shortfall = value["optional_candidate_coverage_shortfall"]
        if (
            support_policy.decision_id == ""
            or not evaluations
            or id_pairs != expected_pairs
            or any(
                item.support_input_digest != support_summary.input_digest
                for item in evaluations
            )
            or value["semantic_phase_inventory_digest"]
            != semantic_phase_inventory_digest(semantic_evaluations)
            or selected_coverage != recomputed_coverage
            or not isinstance(required_shortfall, list)
            or required_shortfall
            or required_shortfall
            != list(required_coverage_shortfall(required_coverage, recomputed_coverage))
            or not isinstance(optional_shortfall, list)
            or optional_shortfall != sorted(set(optional_shortfall))
        ):
            raise ValueError("Court V4 trajectory-plan safety authority is invalid.")
        evaluation_by_group = {item.trajectory_group_id: item for item in evaluations}
        semantic_evaluation_set = set(semantic_evaluations)
        projected_valid_count = 0
        projected_frame_count = 0
        for group in groups:
            group_safety = TrajectorySafetyEvaluation.from_mapping(
                group["safety_evaluation"]
            )
            if evaluation_by_group.get(_nested_group_id(group)) != group_safety:
                raise ValueError(
                    "Court V4 selected evaluation differs from its candidate record."
                )
            group_semantic = TrajectorySemanticPhaseEvaluation.from_mapping(
                group["semantic_phase_evaluation"]
            )
            if group_semantic not in semantic_evaluation_set:
                raise ValueError(
                    "Court V4 selected semantic phase differs from its candidate record."
                )
            projected_valid_count += group_semantic.expected_valid_frame_count
            projected_frame_count += group_semantic.expected_frame_count
        metrics = dataset["metrics"]
        projected_fraction = projected_valid_count / projected_frame_count
        if (
            not isinstance(metrics, Mapping)
            or metrics.get("support_input_digest") != support_summary.input_digest
            or value["projected_semantic_valid_frame_count"] != projected_valid_count
            or value["projected_semantic_valid_fraction"] != projected_fraction
            or metrics.get("semantic_phase_inventory_digest")
            != value["semantic_phase_inventory_digest"]
            or metrics.get("projected_semantic_valid_frame_count")
            != projected_valid_count
            or metrics.get("projected_semantic_valid_fraction") != projected_fraction
            or metrics.get("required_coverage") != required_coverage.to_dict()
            or metrics.get("selected_coverage") != selected_coverage.to_dict()
            or metrics.get("required_coverage_shortfall") != required_shortfall
            or metrics.get("optional_candidate_coverage_shortfall")
            != optional_shortfall
        ):
            raise ValueError("Court V4 plan support digest disagrees with metrics.")
    planned = _mapping_sequence(value["samples"], name="planned samples")
    accepted = _mapping_sequence(dataset["samples"], name="samples")
    rejected = _mapping_sequence(dataset["rejected_samples"], name="rejected_samples")
    records = tuple(
        sorted(
            (*accepted, *rejected),
            key=lambda record: _record_integer(record, "sample_index", minimum=0),
        )
    )
    if len(planned) != len(records):
        raise ValueError("Court trajectory-plan sample inventory is incomplete.")
    base_keys = {
        "sample_index",
        "sample_id",
        "trajectory_group_id",
        "trajectory_id",
        "view_id",
        "trajectory_frame_index",
        "split",
        "shard_id",
        "camera_center_scene_m",
        "camera",
    }
    if _uses_resolved_target_version(definition.version):
        base_keys.add("target_court")
    if definition.version is CourtDatasetSchemaVersion.V4:
        base_keys.update(
            {
                "safety_support_input_digest",
                "semantic_phase_index",
                "semantic_phase_disposition_digest",
            }
        )
    for planned_sample, record in zip(planned, records, strict=True):
        if set(planned_sample) != base_keys:
            raise ValueError("Court planned sample schema is invalid.")
        camera = SceneCamera.from_dict(record.get("camera"))
        expected = {
            key: record[key] for key in base_keys if key != "camera_center_scene_m"
        }
        expected["camera_center_scene_m"] = camera.camera_to_scene.matrix()[
            :3, 3
        ].tolist()
        if dict(planned_sample) != expected:
            raise ValueError("Court planned sample disagrees with published semantics.")


def _validate_safety_diagnostic(
    value: object,
    *,
    dataset: Mapping[str, object],
    trajectory_plan: object,
    definition: CourtSchemaDefinition,
) -> None:
    """Reject missing, unsafe, or tampered V4 safety evidence."""
    keys = {
        "schema",
        "support_policy",
        "support_summary",
        "candidate_safety_evaluations",
        "candidate_semantic_phase_evaluations",
        "semantic_phase_inventory_digest",
        "projected_semantic_valid_frame_count",
        "projected_semantic_valid_fraction",
        "selected_trajectory_group_ids",
        "required_coverage",
        "selected_coverage",
        "required_coverage_shortfall",
        "optional_candidate_coverage_shortfall",
        "selected_point_violation_count",
        "selected_segment_violation_count",
        "zero_selected_safety_violations",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court V4 safety diagnostic schema is invalid.")
    if value["schema"] != definition.safety_diagnostics_schema:
        raise ValueError("Court V4 safety diagnostic version is invalid.")
    support_policy = TrajectorySupportPolicy.from_mapping(value["support_policy"])
    summary = SupportModelSummary.from_mapping(value["support_summary"])
    candidates = _mapping_sequence(
        value["candidate_safety_evaluations"],
        name="candidate_safety_evaluations",
    )
    evaluations = tuple(
        TrajectorySafetyEvaluation.from_mapping(item) for item in candidates
    )
    semantic_candidates = _mapping_sequence(
        value["candidate_semantic_phase_evaluations"],
        name="candidate_semantic_phase_evaluations",
    )
    semantic_evaluations = tuple(
        TrajectorySemanticPhaseEvaluation.from_mapping(item)
        for item in semantic_candidates
    )
    if not isinstance(trajectory_plan, Mapping):
        raise TypeError("Court V4 trajectory plan must be a mapping.")
    planned_policy = TrajectorySupportPolicy.from_mapping(
        trajectory_plan["support_policy"]
    )
    planned_summary = SupportModelSummary.from_mapping(
        trajectory_plan["support_summary"]
    )
    planned_candidates = _mapping_sequence(
        trajectory_plan["candidate_safety_evaluations"],
        name="planned candidate_safety_evaluations",
    )
    planned_evaluations = tuple(
        TrajectorySafetyEvaluation.from_mapping(item) for item in planned_candidates
    )
    planned_semantic_candidates = _mapping_sequence(
        trajectory_plan["candidate_semantic_phase_evaluations"],
        name="planned candidate_semantic_phase_evaluations",
    )
    planned_semantic_evaluations = tuple(
        TrajectorySemanticPhaseEvaluation.from_mapping(item)
        for item in planned_semantic_candidates
    )
    required_coverage = RequiredTrajectoryCoverage.from_mapping(
        value["required_coverage"]
    )
    selected_coverage = SelectedTrajectoryCoverage.from_mapping(
        value["selected_coverage"]
    )
    if (
        support_policy != planned_policy
        or summary != planned_summary
        or evaluations != planned_evaluations
        or semantic_evaluations != planned_semantic_evaluations
        or value["semantic_phase_inventory_digest"]
        != semantic_phase_inventory_digest(semantic_evaluations)
        or value["semantic_phase_inventory_digest"]
        != trajectory_plan["semantic_phase_inventory_digest"]
        or value["projected_semantic_valid_frame_count"]
        != trajectory_plan["projected_semantic_valid_frame_count"]
        or value["projected_semantic_valid_fraction"]
        != trajectory_plan["projected_semantic_valid_fraction"]
        or value["required_coverage"] != trajectory_plan["required_coverage"]
        or value["selected_coverage"] != trajectory_plan["selected_coverage"]
        or value["required_coverage_shortfall"]
        != trajectory_plan["required_coverage_shortfall"]
        or value["optional_candidate_coverage_shortfall"]
        != trajectory_plan["optional_candidate_coverage_shortfall"]
        or value["required_coverage_shortfall"] != []
        or not isinstance(required_coverage, RequiredTrajectoryCoverage)
        or not isinstance(selected_coverage, SelectedTrajectoryCoverage)
    ):
        raise ValueError(
            "Court V4 safety diagnostic candidate inventory differs from the plan."
        )
    evaluation_by_group = {
        evaluation.trajectory_group_id: evaluation for evaluation in evaluations
    }
    if (
        not evaluations
        or len(evaluation_by_group) != len(evaluations)
        or len({item.trajectory_id for item in evaluations}) != len(evaluations)
    ):
        raise ValueError("Court V4 candidate safety IDs are duplicated or empty.")
    groups = _mapping_sequence(dataset["trajectory_groups"], name="trajectory_groups")
    selected_ids = [_nested_group_id(group) for group in groups]
    selected_evaluations = [
        evaluation_by_group.get(group_id) for group_id in selected_ids
    ]
    group_evaluations = [
        TrajectorySafetyEvaluation.from_mapping(group["safety_evaluation"])
        for group in groups
    ]
    semantic_evaluation_set = set(semantic_evaluations)
    group_semantic_evaluations = [
        TrajectorySemanticPhaseEvaluation.from_mapping(
            group["semantic_phase_evaluation"]
        )
        for group in groups
    ]
    metrics = dataset["metrics"]
    if (
        not isinstance(metrics, Mapping)
        or metrics.get("support_input_digest") != summary.input_digest
        or value["selected_trajectory_group_ids"] != selected_ids
        or any(item is None or not item.safe for item in selected_evaluations)
        or selected_evaluations != group_evaluations
        or any(
            item not in semantic_evaluation_set for item in group_semantic_evaluations
        )
        or value["selected_point_violation_count"] != 0
        or value["selected_segment_violation_count"] != 0
        or value["zero_selected_safety_violations"] is not True
        or metrics.get("selected_safety_violation_count") != 0
        or value["required_coverage"] != metrics.get("required_coverage")
        or value["selected_coverage"] != metrics.get("selected_coverage")
        or value["required_coverage_shortfall"]
        != metrics.get("required_coverage_shortfall")
        or value["required_coverage_shortfall"] != []
        or value["optional_candidate_coverage_shortfall"]
        != metrics.get("optional_candidate_coverage_shortfall")
        or metrics.get("semantic_phase_inventory_digest")
        != value["semantic_phase_inventory_digest"]
        or metrics.get("projected_semantic_valid_frame_count")
        != value["projected_semantic_valid_frame_count"]
        or metrics.get("projected_semantic_valid_fraction")
        != value["projected_semantic_valid_fraction"]
    ):
        raise ValueError("Court V4 selected safety evidence is inconsistent.")


def _contained_file(root: Path, relative_value: str) -> Path:
    if "\\" in relative_value:
        raise ValueError("Court dataset paths must use POSIX separators.")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe Court dataset path: {relative_value!r}.")
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if (
        not resolved.is_relative_to(resolved_root)
        or not resolved.is_file()
        or candidate.is_symlink()
    ):
        raise ValueError(
            f"Court dataset path is not a contained ordinary file: {relative_value}"
        )
    return resolved


def _validated_v2_target_policy(
    *,
    trajectory: OrbitTrajectorySpec,
    value: object,
) -> TargetCourtPolicyV2:
    """Parse a persisted policy and bind it to trajectory-centre authority."""
    persisted = TargetCourtPolicyV2.from_mapping(value)
    required = target_court_policy_for_trajectory(trajectory)
    if persisted != required:
        raise ValueError("Court v2 target policy disagrees with its trajectory centre.")
    return persisted


def _parse_canonical_sampling_policy(
    value: object,
    *,
    version: CourtDatasetSchemaVersion,
) -> OrbitSamplingPolicy:
    """Parse an exact policy and reject non-canonical persisted mappings."""
    policy = OrbitSamplingPolicy.from_mapping(value)
    expected_stable_fields = (
        V4_ORBIT_STABLE_FIELDS
        if version is CourtDatasetSchemaVersion.V4
        else LEGACY_ORBIT_STABLE_FIELDS
    )
    if {field.value for field in policy.stable_field_order} != {
        field.value for field in expected_stable_fields
    }:
        raise ValueError(
            "stable_field_order must list every OrbitStableField exactly once."
        )
    if set(policy.coverage_objective) != set(OrbitCoverageObjective):
        raise ValueError(
            "coverage_objective must list every OrbitCoverageObjective exactly once."
        )
    if value != policy.to_dict():
        raise ValueError("Court sampling policy is not canonical.")
    return policy


def _mapping_sequence(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        raise TypeError(f"{name} must be a list of mappings.")
    return tuple(value)


def _metric_integer(metrics: Mapping[str, object], key: str, *, minimum: int) -> int:
    return _mapping_integer(metrics, key, minimum=minimum)


def _mapping_integer(value: Mapping[str, object], key: str, *, minimum: int) -> int:
    if key not in value:
        raise TypeError(f"{key} is required.")
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
        raise TypeError(f"{key} must be an integer >= {minimum}.")
    return item


def _record_integer(value: Mapping[str, object], key: str, *, minimum: int) -> int:
    return _mapping_integer(value, key, minimum=minimum)


def _mapping_float(value: Mapping[str, object], key: str) -> float:
    if key not in value:
        raise TypeError(f"{key} is required.")
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{key} must be numeric.")
    result = float(item)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite.")
    return result


def _json_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain numeric values.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain finite values.")
    return result


def _validate_published_ambiguity_reason(
    reasons: Sequence[object],
    *,
    camera: SceneCamera,
    published_court_geometry: Mapping[str, RigidTransform],
) -> None:
    """Require the persisted mid-plane reason to match physical geometry."""
    if (
        len(reasons) != 1
        or not isinstance(reasons[0], str)
        or not _is_ambiguous_near_far_reason(reasons[0])
    ):
        raise ValueError(
            "A null Court v2/v3 projection requires exactly one ambiguity reason."
        )
    reason = reasons[0]
    court_id = reason.removeprefix(f"{AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON}:")
    try:
        scene_from_court = published_court_geometry[court_id]
    except KeyError as error:
        raise ValueError(
            "Court ambiguity reason references unpublished court geometry."
        ) from error
    local_y = camera_center_court_y(
        camera,
        scene_from_court=scene_from_court,
    )
    if abs(local_y) > CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M:
        raise ValueError("Court ambiguity reason disagrees with camera/court geometry.")


def _is_ambiguous_near_far_reason(reason: str) -> bool:
    prefix, separator, court_id = reason.partition(":")
    return (
        prefix == AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON
        and separator == ":"
        and bool(court_id)
        and court_id == court_id.strip()
    )


def _require_finite_json(value: object, *, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} contains a non-string key.")
            _require_finite_json(item, name=f"{name}.{key}")
    elif isinstance(value, list):
        for item in value:
            _require_finite_json(item, name=name)


__all__ = [
    "CourtArrayValidationMode",
    "CourtAssemblyReport",
    "assemble_court_dataset",
    "validate_court_dataset",
]
