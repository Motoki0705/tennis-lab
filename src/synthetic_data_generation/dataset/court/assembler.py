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
from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
    MultiCourtProjection,
    attach_renderer_visibility_from_validated_arrays,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    COURT_DATASET_SCHEMA,
    COURT_SAMPLE_SCHEMA,
    CourtDatasetPlan,
    OrbitTrajectorySpec,
    OrbitViewSpec,
    PlannedCourtSample,
    TrajectoryGroupPlan,
)
from src.synthetic_data_generation.dataset.court.diagnostics import (
    DIAGNOSTIC_FILES,
    write_court_diagnostics,
)
from src.synthetic_data_generation.dataset.court.performance import (
    CourtPerformanceEvidence,
)
from src.synthetic_data_generation.dataset.court.semantic_manifest import (
    COURT_SEMANTIC_MANIFEST_PATH,
    build_court_semantic_manifest,
    validate_court_semantic_manifest,
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
    projection: MultiCourtProjection
    accepted: bool
    rejection_reasons: tuple[str, ...]
    complete_array_scan_count: int


def assemble_court_dataset(
    staging_root: Path,
    *,
    plan: CourtDatasetPlan,
    layout: MultiCourtLayout,
    metric_adapter: MetricSceneAdapter,
    render_result: CourtRenderResult,
    configuration: CourtDatasetConfiguration,
    attempt_root: Path,
    performance_timer: PerformanceTimer,
) -> CourtAssemblyReport:
    """Stream staged samples through gates and assemble fixed outputs."""
    if not staging_root.is_absolute() or not staging_root.is_dir() or staging_root.is_symlink():
        raise ValueError("Court staging_root must be an existing absolute ordinary directory.")
    if not isinstance(render_result, CourtRenderResult):
        raise TypeError("Court assembly requires a complete CourtRenderResult.")
    if not isinstance(performance_timer, PerformanceTimer):
        raise TypeError("Court assembly requires an attempt-scoped PerformanceTimer.")
    rendered_tuple = render_result.samples
    _validate_render_inventory(
        plan,
        rendered_tuple,
        pre_render_rejected_sample_ids=render_result.pre_render_rejected_sample_ids,
    )
    projection_by_id = render_result.projection_by_sample_id
    if tuple(projection_by_id) != tuple(sample.sample_id for sample in plan.samples):
        raise ValueError("Court pre-render projection inventory changed before assembly.")
    expected_court_ids = tuple(court.court_instance_id for court in layout.courts)
    if any(
        tuple(court.court_instance_id for court in projection.courts)
        != expected_court_ids
        for projection in projection_by_id.values()
    ):
        raise ValueError("Court pre-render projections disagree with the alignment layout.")
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
            raise ValueError("NHT render output must stay inside Court attempt staging.")
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
            "schema": COURT_SAMPLE_SCHEMA,
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
        save_json_atomic(label_payload, label_path)
        relative_directory = destination.relative_to(staging_root).as_posix()
        accepted_records.append(
            {
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
        )
    accepted = tuple(item for item in evaluated if item.accepted)
    post_render_rejected = tuple(item for item in evaluated if not item.accepted)
    planned_by_id = {sample.sample_id: sample for sample in plan.samples}
    rejected_records = [
        _rejected_record(
            planned_by_id[sample_id],
            group=group_by_id[planned_by_id[sample_id].trajectory_group_id],
            projection=projection_by_id[sample_id],
            profile=plan.profile,
            metadata_fields=configuration.metadata_fields,
            reasons=("insufficient_pre_render_semantic_coverage",),
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
        court.coverage_mode
        for item in accepted
        for court in item.projection.courts
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
        visible_by_class={name: visible_by_class[name] for name in SEMANTIC_CLASS_NAMES},
    )
    metrics = _metrics(
        plan,
        accepted_records=accepted_records,
        rejected_count=len(rejected_records),
        coverage_counts=coverage_counts,
        visible_by_class=visible_by_class,
    )
    manifest = {
        "schema": COURT_DATASET_SCHEMA,
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
    )
    return validate_court_dataset(
        staging_root,
        expected_plan=plan,
        expected_configuration=configuration,
        array_validation=CourtArrayValidationMode.HEADERS_ONLY,
    )


def _validate_render_inventory(
    plan: CourtDatasetPlan,
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
    projection: MultiCourtProjection,
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
    plan: CourtDatasetPlan,
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
        raise ValueError("Court accepted-frame count is below the resolved quality gate.")
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
        name for name in SEMANTIC_CLASS_NAMES if visible_by_class.get(name, 0) <= 0
    ]
    if missing_classes:
        raise ValueError(
            f"Court output lacks renderer-visible semantic classes: {missing_classes}."
        )


def _sample_metadata(
    sample: PlannedCourtSample,
    *,
    group: TrajectoryGroupPlan,
    view_id: str,
    profile: str,
    metadata_fields: Sequence[str],
) -> dict[str, object]:
    available: dict[str, object] = {
        "target_court": group.target_court.court_instance_id,
        "candidate_id": group.target_court.candidate_id,
        "transform": group.target_court.scene_from_court.to_list(),
        "camera_profile": profile,
        "camera_parameters": {
            "view_id": view_id,
            "camera_center_scene_m": list(sample.camera_center_scene_m),
            "intrinsics": list(sample.camera.intrinsics),
            "camera_to_scene": sample.camera.camera_to_scene.to_list(),
        },
        "seed": group.target_court.selection_seed,
    }
    unknown = set(metadata_fields) - set(available)
    if unknown:
        raise ValueError(
            f"Court metadata fields lack a semantic source: {sorted(unknown)}."
        )
    return {field: available[field] for field in metadata_fields}


def _rejected_record(
    sample: PlannedCourtSample,
    *,
    group: TrajectoryGroupPlan,
    projection: MultiCourtProjection,
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
    if projection.camera_id != sample.sample_id:
        raise ValueError("Rejected Court projection disagrees with the planned sample.")
    return {
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
        "projection": projection.to_dict(),
        "metadata": _sample_metadata(
            sample,
            group=group,
            view_id=sample.view_id,
            profile=profile,
            metadata_fields=metadata_fields,
        ),
        "reasons": list(reason_tuple),
    }


def _metrics(
    plan: CourtDatasetPlan,
    *,
    accepted_records: Sequence[Mapping[str, object]],
    rejected_count: int,
    coverage_counts: Mapping[str, int],
    visible_by_class: Mapping[str, int],
) -> dict[str, object]:
    accepted_count = len(accepted_records)
    split_frame_counts = Counter(str(record["split"]) for record in accepted_records)
    split_group_counts = Counter(group.split.value for group in plan.groups)
    court_group_counts = Counter(
        group.target_court.court_instance_id for group in plan.groups
    )
    split_court_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for group in plan.groups:
        split_court_group_counts[group.split.value][
            group.target_court.court_instance_id
        ] += 1
    return {
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
        "court_group_counts": dict(sorted(court_group_counts.items())),
        "split_court_group_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_court_group_counts.items())
        },
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "renderer_visible_points_by_class": {
            name: visible_by_class.get(name, 0) for name in SEMANTIC_CLASS_NAMES
        },
        "split_leakage_count": 0,
    }


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
) -> CourtPerformanceEvidence:
    """Persist self-consistent measured bytes and fail closed on every budget."""
    wall_seconds, cpu_seconds, peak_rss_bytes = timer.elapsed()
    published_bytes = max(1, directory_size_bytes(root))
    performance_path = root / "diagnostics" / "performance.json"
    pre_render_rejected_sample_count = len(
        render_result.pre_render_rejected_sample_ids
    )
    renderable_sample_count = proposal_count - pre_render_rejected_sample_count
    post_render_rejected_sample_count = (
        rejected_frame_count - pre_render_rejected_sample_count
    )
    staged_complete_array_scans = (
        accepted_staged_complete_array_scans
        + post_render_rejected_staged_complete_array_scans
    )
    fresh_rendered_sample_count = render_result.nht_complete_array_scans
    reused_rendered_sample_count = (
        renderable_sample_count - fresh_rendered_sample_count
    )
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
                render_result.nht_complete_array_scans
                + staged_complete_array_scans
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
            nht_boundary_complete_array_scans=(
                render_result.nht_complete_array_scans
            ),
            accepted_staged_complete_array_scans=(
                accepted_staged_complete_array_scans
            ),
            post_render_rejected_staged_complete_array_scans=(
                post_render_rejected_staged_complete_array_scans
            ),
            staged_complete_array_scans=staged_complete_array_scans,
            fresh_run_complete_array_scan_requirement=(
                fresh_run_complete_array_scan_requirement
            ),
            complete_array_scan_budget_capacity=(
                complete_array_scan_budget_capacity
            ),
            scene_validation_count=render_result.scene_validation_count,
            preview_validation_count=render_result.preview_validation_count,
            loaded_array_bytes=render_result.loaded_array_bytes,
            maximum_nht_live_array_bytes=(
                render_result.maximum_nht_live_array_bytes
            ),
            retained_nht_array_bytes=render_result.retained_nht_array_bytes,
            external_nht_boundary_wall_seconds=(
                render_result.external_nht_boundary_wall_seconds
            ),
            shard_wall_seconds={
                timing.shard_id: timing.wall_seconds
                for timing in render_result.shard_timings
            },
            visible_points_by_class=visible_by_class,
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
    expected_plan: CourtDatasetPlan | None = None,
    expected_configuration: CourtDatasetConfiguration | None = None,
    array_validation: CourtArrayValidationMode = CourtArrayValidationMode.FULL,
) -> CourtAssemblyReport:
    """Validate the complete canonical output and reject every inventory mismatch."""
    if not isinstance(array_validation, CourtArrayValidationMode):
        raise TypeError("array_validation must be a CourtArrayValidationMode.")
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
    if raw["schema"] != COURT_DATASET_SCHEMA or raw["status"] != "completed":
        raise ValueError("Court dataset schema/status is invalid.")
    validate_court_semantic_manifest(
        raw,
        load_json(_contained_file(root, COURT_SEMANTIC_MANIFEST_PATH)),
    )
    groups = _mapping_sequence(raw["trajectory_groups"], name="trajectory_groups")
    samples = _mapping_sequence(raw["samples"], name="samples")
    rejected = _mapping_sequence(raw["rejected_samples"], name="rejected_samples")
    metrics = raw["metrics"]
    policy = raw["sampling_policy"]
    if not isinstance(metrics, Mapping) or not isinstance(policy, Mapping):
        raise TypeError("Court metrics and sampling_policy must be mappings.")
    for group in groups:
        _validate_group_record(group)
    group_ids = [_nested_group_id(group) for group in groups]
    if not group_ids or len(group_ids) != len(set(group_ids)):
        raise ValueError("Court trajectory group IDs must be non-empty and unique.")
    split_by_group = {
        group_id: group["split"] for group_id, group in zip(group_ids, groups, strict=True)
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
    for record in rejected:
        if set(record) != rejected_keys or not isinstance(record.get("reasons"), list):
            raise ValueError("Rejected Court sample record schema is invalid.")
        sample_id, sample_index = _validate_semantic_sample_record(
            record,
            group_by_id=group_by_id,
            metadata_fields=metadata_fields,
            profile=raw["profile"],
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
        if set(record) != expected_sample_keys:
            raise ValueError("Court sample record contains missing or unexpected fields.")
        if record["depth_coordinate_space"] != "metric_scene_metres":
            raise ValueError("Court sample depth must use metric scene metres.")
        sample_id, sample_index = _validate_semantic_sample_record(
            record,
            group_by_id=group_by_id,
            metadata_fields=metadata_fields,
            profile=raw["profile"],
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
        raise ValueError("Court proposal indices do not cover the full planned inventory.")
    budget = _mapping_integer(policy, "proposal_budget", minimum=1)
    minimum_groups = _mapping_integer(
        policy, "minimum_trajectory_groups", minimum=1
    )
    minimum_frames = _mapping_integer(policy, "minimum_accepted_frames", minimum=1)
    minimum_fraction = _mapping_float(policy, "minimum_accepted_fraction")
    max_allowed_step = _mapping_float(policy, "max_arc_step_m")
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
    coverage = metrics.get("coverage_counts")
    visible = metrics.get("renderer_visible_points_by_class")
    if not isinstance(coverage, Mapping) or not {"full", "near_full", "partial"}.issubset(
        {name for name, count in coverage.items() if isinstance(count, int) and count > 0}
    ):
        raise ValueError("Court coverage diversity gate failed.")
    if not isinstance(visible, Mapping) or set(visible) != set(SEMANTIC_CLASS_NAMES) or any(
        not isinstance(visible[name], int) or visible[name] <= 0
        for name in SEMANTIC_CLASS_NAMES
    ):
        raise ValueError("Court seven-class renderer visibility gate failed.")
    _validate_court_balance(groups)
    diagnostics = raw["diagnostics"]
    expected_diagnostics = [f"diagnostics/{name}" for name in DIAGNOSTIC_FILES]
    if diagnostics != expected_diagnostics:
        raise ValueError("Court diagnostic inventory is incomplete or unexpected.")
    for relative in expected_diagnostics:
        _contained_file(root, relative)
    performance = CourtPerformanceEvidence.from_dict(
        load_json(_contained_file(root, "diagnostics/performance.json"))
    )
    if (
        performance.proposal_count != proposal_count
        or performance.accepted_frame_count != accepted_count
        or performance.rejected_frame_count != rejected_count
        or performance.visible_points_by_class
        != {name: int(visible[name]) for name in SEMANTIC_CLASS_NAMES}
    ):
        raise ValueError("Court performance evidence disagrees with semantic metrics.")
    actual_published_bytes = directory_size_bytes(root)
    if performance.metrics.published_bytes != actual_published_bytes:
        raise ValueError("Court published-byte evidence disagrees with the dataset.")
    if expected_plan is not None:
        if raw["scene_id"] != expected_plan.scene_id or raw["profile"] != expected_plan.profile:
            raise ValueError("Published Court scene/profile disagrees with the resolved plan.")
        if proposal_count != expected_plan.proposal_count or group_ids != [
            group.trajectory_group_id for group in expected_plan.groups
        ]:
            raise ValueError("Published Court inventory disagrees with the resolved plan.")
        if list(groups) != [group.to_dict() for group in expected_plan.groups]:
            raise ValueError("Published Court trajectory semantics changed after planning.")
        if policy != expected_plan.policy.to_dict():
            raise ValueError("Published Court sampling policy changed after planning.")
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
    if expected_configuration is not None and metadata_fields != list(
        expected_configuration.metadata_fields
    ):
        raise ValueError("Published Court metadata fields changed after configuration.")
    if (
        expected_configuration is not None
        and performance.budget != expected_configuration.performance
    ):
        raise ValueError("Published Court performance budget changed after configuration.")
    court_counts_raw = metrics.get("court_group_counts")
    if not isinstance(court_counts_raw, Mapping):
        raise TypeError("court_group_counts must be a mapping.")
    return CourtAssemblyReport(
        proposal_count=proposal_count,
        accepted_frame_count=accepted_count,
        rejected_frame_count=rejected_count,
        trajectory_group_count=group_count,
        maximum_adjacent_step_m=observed_step,
        accepted_fraction=observed_fraction,
        split_frame_counts=dict(split_counts),
        court_group_counts={str(key): int(value) for key, value in court_counts_raw.items()},
        performance=performance,
    )


def _validate_semantic_sample_record(
    record: Mapping[str, object],
    *,
    group_by_id: Mapping[str, Mapping[str, object]],
    metadata_fields: Sequence[str],
    profile: object,
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
        raise ValueError("Court sample references an unknown trajectory group.") from error
    trajectory = group.get("trajectory")
    views = group.get("views")
    target = group.get("target_court")
    if (
        not isinstance(trajectory, Mapping)
        or not isinstance(views, list)
        or not isinstance(target, Mapping)
    ):
        raise TypeError("Court trajectory group semantics are incomplete.")
    view_ids = {
        view.get("view_id")
        for view in views
        if isinstance(view, Mapping)
    }
    if (
        trajectory.get("trajectory_id") != trajectory_id
        or group.get("split") != split
        or group.get("shard_id") != shard_id
        or view_id not in view_ids
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
        raise ValueError("Court embedded camera disagrees with sample identity/resolution.")
    projection = record.get("projection")
    if not isinstance(projection, Mapping):
        raise TypeError("Court sample projection must be a mapping.")
    if (
        projection.get("camera_id") != sample_id
        or projection.get("resolution") != [width, height]
    ):
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
    expected_metadata = {
        "target_court": target.get("court_instance_id"),
        "candidate_id": target.get("candidate_id"),
        "transform": target.get("scene_from_court"),
        "camera_profile": profile,
        "camera_parameters": expected_camera_parameters,
        "seed": target.get("selection_seed"),
    }
    if dict(metadata) != {
        field: expected_metadata[field]
        for field in metadata_fields
        if field in expected_metadata
    } or any(field not in expected_metadata for field in metadata_fields):
        raise ValueError("Court sample metadata disagrees with canonical semantic sources.")
    return sample_id, sample_index


def _validate_published_sample(
    root: Path,
    record: Mapping[str, object],
    *,
    array_validation: CourtArrayValidationMode,
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
    if (
        not isinstance(label_payload, Mapping)
        or set(label_payload) != label_keys
        or label_payload["schema"] != COURT_SAMPLE_SCHEMA
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
    _require_finite_json(label_payload, name="labels")
    if array_validation is CourtArrayValidationMode.FULL:
        _validate_renderer_visibility_payload(
            label_payload["projection"],
            alpha=loaded_arrays["alpha"],
            depth=loaded_arrays["depth"],
        )


def _validate_renderer_visibility_payload(
    value: object,
    *,
    alpha: NDArray[np.float32],
    depth: NDArray[np.float32],
) -> None:
    """Recompute every stored visibility bit from published renderer arrays."""
    if not isinstance(value, Mapping):
        raise TypeError("Court semantic projection must be a mapping.")
    courts = value.get("courts")
    if not isinstance(courts, list):
        raise TypeError("Court semantic projection courts must be a list.")
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
                if not isinstance(in_frame, bool) or not isinstance(uv, list) or len(uv) != 2:
                    raise TypeError("Court semantic point visibility inputs are invalid.")
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
    expected_names = [name for name in SEMANTIC_CLASS_NAMES if name in visible_names]
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


def _validate_group_record(group: Mapping[str, object]) -> None:
    keys = {
        "trajectory",
        "center",
        "views",
        "split",
        "shard_id",
        "target_court",
        "sample_count",
        "maximum_adjacent_step_m",
        "total_arc_length_m",
    }
    if set(group) != keys:
        raise ValueError("Court trajectory group schema is invalid.")
    trajectory = group["trajectory"]
    OrbitTrajectorySpec.from_mapping(trajectory)
    views = group["views"]
    if not isinstance(views, list) or not views:
        raise TypeError("Court group views must be a non-empty list.")
    for view in views:
        OrbitViewSpec.from_mapping(view)
    center = group["center"]
    center_keys = {
        "center_kind",
        "court_instance_id",
        "reference_court_instance_id",
        "scene_from_center",
        "center_scene_m",
        "base_radius_m",
        "captured_offset_median_m",
        "captured_offset_q90_m",
        "captured_camera_count",
    }
    if not isinstance(center, Mapping) or set(center) != center_keys:
        raise ValueError("Court resolved centre schema is invalid.")
    transform = center["scene_from_center"]
    if not isinstance(transform, list):
        raise TypeError("Court centre transform must be a list.")
    RigidTransform(tuple(_json_float(value, name="scene_from_center") for value in transform))
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
        tuple(_json_float(value, name="scene_from_court") for value in target_transform)
    )
    if group["split"] not in {"train", "validation", "test"}:
        raise ValueError("Court group split is invalid.")
    if not isinstance(group["shard_id"], str) or not group["shard_id"]:
        raise TypeError("Court group shard_id must be non-empty.")
    _mapping_integer(group, "sample_count", minimum=8)
    _mapping_float(group, "maximum_adjacent_step_m")
    _mapping_float(group, "total_arc_length_m")


def _nested_group_id(group: Mapping[str, object]) -> str:
    trajectory = group.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise TypeError("Court group trajectory must be a mapping.")
    group_id = trajectory.get("trajectory_group_id")
    if not isinstance(group_id, str) or not group_id:
        raise TypeError("trajectory_group_id must be a non-empty string.")
    return group_id


def _contained_file(root: Path, relative_value: str) -> Path:
    if "\\" in relative_value:
        raise ValueError("Court dataset paths must use POSIX separators.")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe Court dataset path: {relative_value!r}.")
    candidate = root.joinpath(*relative.parts)
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file() or candidate.is_symlink():
        raise ValueError(f"Court dataset path is not a contained ordinary file: {relative_value}")
    return resolved


def _mapping_sequence(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
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
