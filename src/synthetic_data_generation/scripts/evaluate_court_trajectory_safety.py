"""Render, observation-lock, and finalize the frozen Court safety benchmark.

Usage:
    python -m src.synthetic_data_generation.scripts.evaluate_court_trajectory_safety action=validate_complete_evidence scene_path=/absolute/path/to/scene.json alignment_path=/absolute/path/to/alignment.json

Notes:
    - Hydra loads benchmark authority from `src/synthetic_data_generation/configs/evaluate_court_trajectory_safety.yaml`.
    - Render actions use only `StandardSceneExport` and `NHTRenderClient`; finalization joins already-complete private blind annotations by opaque ID.
    - `validate_complete_evidence` is read-only: it never renders, creates directories, regenerates contact sheets, or writes evidence.
    - The final V4 dataset action is explicit and non-default; it never mutates the existing B00 V1-V3 scene dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.configuration import CourtDatasetConfiguration
from src.synthetic_data_generation.dataset.court.assembler import (
    CourtArrayValidationMode,
    assemble_court_dataset,
    validate_court_dataset,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.selection import (
    build_court_dataset_plan,
)
from src.synthetic_data_generation.dataset.court.components.camera_sampling.support import (
    TrajectorySupportModel,
    build_trajectory_support_model,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    CourtDatasetPlanV3,
    CourtDatasetPlanV4,
    RequiredTrajectoryCoverage,
    SelectedTrajectoryCoverage,
    required_coverage_shortfall,
)
from src.synthetic_data_generation.dataset.court.evaluation.quality import (
    BLIND_ADJUDICATION_SCHEMA,
    BLIND_ANNOTATION_SCHEMA,
    MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE,
    MINIMUM_HELD_OUT_NEGATIVE_LABELS,
    MINIMUM_HELD_OUT_POSITIVE_LABELS,
    MINIMUM_HELD_OUT_PRECISION,
    MINIMUM_HELD_OUT_RECALL,
    QUALITY_FEATURE_DEFINITION_ID,
    AnnotationConsensus,
    BenchmarkSplit,
    BlindAnnotation,
    PublicQualityFeatures,
    QualityDecision,
    QualityDecisionReport,
    QualityObservation,
    QualityRuleCalibration,
    assign_group_held_out_splits,
    calibrate_quality_only_rule,
    derive_annotation_consensus,
    evaluate_quality_only_rule,
    extract_public_quality_features,
    opaque_review_ids,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import (
    CourtNHTRenderer,
)
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.dataset.runtime import PerformanceTimer
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.rendering.nht import (
    NHTRenderCamera,
    NHTRenderClient,
    NHTRenderCommandRequest,
    NHTRenderRequest,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.hydra import hydra_main
from src.utils.io import load_json, save_json_atomic
from src.utils.paths import PROJECT_ROOT

PILOT_MANIFEST_SCHEMA = "court_trajectory_safety_pilot_manifest_v1"
PILOT_FEATURES_SCHEMA = "court_trajectory_safety_pilot_features_v1"
BLIND_REVIEW_MANIFEST_SCHEMA = "court_trajectory_blind_review_manifest_v1"
TRACKED_EVIDENCE_SCHEMA = "court_trajectory_safety_evidence_v2"
DERIVED_EVIDENCE_SCHEMA = "court_trajectory_safety_derived_evidence_v2"
FROZEN_CONFIG_SCHEMA = "court_trajectory_safety_frozen_config_v2"
_OPAQUE_ID = re.compile(r"^review-[0-9a-f]{16}$")
_REQUIRED_STRATA = frozenset(
    {
        "captured_control",
        "legacy_orbit",
        "support_interior",
        "support_boundary",
        "support_exterior",
        "safe_v4_candidate",
    }
)
_MINIMUM_VIEWS_PER_STRATUM = 12
_REQUIRED_PROPOSAL_BUDGET = 4_800
_TRACKED_EVIDENCE_PATH = (
    "experiments/synthetic_data_generation/court_trajectory_safety/"
    "pilot-manifest.json"
)
_FINAL_EVIDENCE_ROOT = (
    "outputs/court_trajectory_safety/B00-required-coverage-final"
)
_PILOT_DECISION_INPUT_KEYS = {
    "feature_definition_id",
    "legacy_plan_seed",
    "candidate_plan_seed",
    "legacy_proposal_budget",
    "candidate_proposal_budget",
    "equal_proposal_budget",
    "candidate_count",
    "safe_candidate_count",
    "selected_group_count",
    "planned_frame_count",
    "selected_support_violation_count",
    "selected_trajectory_group_ids",
    "required_coverage",
    "selected_coverage",
    "required_coverage_shortfall",
    "optional_candidate_coverage_shortfall",
    "semantic_phase_evaluation_count",
    "semantic_phase_count",
    "semantic_phase_inventory_digest",
    "projected_semantic_valid_frame_count",
    "projected_semantic_rejected_frame_count",
    "projected_semantic_valid_fraction",
    "selected_semantic_phases",
}
_T = TypeVar("_T")


class BenchmarkAction(StrEnum):
    """Explicit non-render preparation or frozen pilot-render action."""

    PREPARE_MANIFEST = "prepare_manifest"
    RENDER_FROZEN_PILOT = "render_frozen_pilot"
    FREEZE_PILOT_OBSERVATIONS = "freeze_pilot_observations"
    FINALIZE_PILOT_EVIDENCE = "finalize_pilot_evidence"
    RENDER_FINAL_V4 = "render_final_v4"
    FINALIZE_COMPLETE_EVIDENCE = "finalize_complete_evidence"
    VALIDATE_COMPLETE_EVIDENCE = "validate_complete_evidence"


@dataclass(frozen=True, slots=True)
class ObservationLock:
    """Hash-only binding for one explicitly frozen pilot observation set."""

    pilot_manifest_sha256: str
    pilot_features_sha256: str
    blind_review_manifest_sha256: str
    reviewer_a_sha256: str
    reviewer_b_sha256: str
    adjudication_sha256: str
    rgb_preview_inventory_sha256: str

    def __post_init__(self) -> None:
        if any(
            not _is_sha256(getattr(self, name))
            for name in self.__dataclass_fields__
        ):
            raise ValueError("Observation lock fields must be SHA-256 values.")

    @classmethod
    def from_mapping(cls, value: object) -> ObservationLock:
        keys = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError("Observation lock schema is invalid.")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkAuthority:
    """Machine-read immutable protocol plus optional explicit observation lock."""

    scene_id: str
    pilot_seed: int
    minimum_pilot_views: int
    required_strata: tuple[str, ...]
    calibration_fraction: float
    quality_only_thresholds: Mapping[str, object]
    geometry_release_gates: Mapping[str, object]
    observation_lock: ObservationLock | None
    normalized_payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    action: BenchmarkAction
    scene_path: Path
    alignment_path: Path
    pilot_manifest_path: Path
    pilot_output_root: Path
    final_evidence_root: Path
    annotation_root: Path
    audit_output_root: Path
    source_video_path: Path
    frozen_config_path: Path
    frozen_authority: FrozenBenchmarkAuthority
    render_executable: str | Path
    timeout_seconds: float
    environment: Mapping[str, str]
    minimum_view_count: int
    split_seed: int
    calibration_fraction: float
    candidate_court: CourtDatasetConfiguration
    legacy_court: CourtDatasetConfiguration

    @classmethod
    def from_config(cls, config: DictConfig) -> BenchmarkConfiguration:
        raw = OmegaConf.to_container(config, resolve=True)
        if not isinstance(raw, Mapping):
            raise TypeError("Court safety benchmark config must be a mapping.")
        keys = {
            "action",
            "scene_path",
            "alignment_path",
            "pilot_manifest_path",
            "pilot_output_root",
            "final_evidence_root",
            "annotation_root",
            "audit_output_root",
            "source_video_path",
            "frozen_config_path",
            "render_executable",
            "timeout_seconds",
            "environment",
            "candidate_court",
            "legacy_court",
        }
        if set(raw) != keys:
            raise ValueError("Court safety benchmark config keys are invalid.")
        try:
            action = BenchmarkAction(str(raw["action"]))
        except ValueError as error:
            raise ValueError("Court safety benchmark action is invalid.") from error
        scene_path = _path(raw["scene_path"], name="scene_path", must_exist=True)
        alignment_path = _path(
            raw["alignment_path"], name="alignment_path", must_exist=True
        )
        manifest_path = _path(
            raw["pilot_manifest_path"],
            name="pilot_manifest_path",
            must_exist=action is not BenchmarkAction.PREPARE_MANIFEST,
        )
        pilot_output_root = _path(
            raw["pilot_output_root"], name="pilot_output_root", must_exist=False
        )
        final_evidence_root = _path(
            raw["final_evidence_root"], name="final_evidence_root", must_exist=False
        )
        annotation_root = _path(
            raw["annotation_root"], name="annotation_root", must_exist=False
        )
        audit_output_root = _path(
            raw["audit_output_root"], name="audit_output_root", must_exist=False
        )
        source_video_path = _path(
            raw["source_video_path"], name="source_video_path", must_exist=False
        )
        frozen_config_path = _path(
            raw["frozen_config_path"],
            name="frozen_config_path",
            must_exist=True,
        )
        frozen_authority = _load_frozen_benchmark_authority(
            frozen_config_path,
            allow_unfrozen_observation_lock=(
                action
                in {
                    BenchmarkAction.PREPARE_MANIFEST,
                    BenchmarkAction.RENDER_FROZEN_PILOT,
                    BenchmarkAction.FREEZE_PILOT_OBSERVATIONS,
                }
            ),
        )
        if (
            audit_output_root.parent != annotation_root
            or audit_output_root.name != "evidence"
        ):
            raise ValueError("audit_output_root must be annotation_root/evidence.")
        executable_raw = raw["render_executable"]
        if not isinstance(executable_raw, str) or not executable_raw:
            raise TypeError("render_executable must be a non-empty string.")
        executable: str | Path = (
            executable_raw
            if executable_raw == "nht-render"
            else _path(executable_raw, name="render_executable", must_exist=True)
        )
        environment = raw["environment"]
        if not isinstance(environment, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        ):
            raise TypeError("environment must map strings to strings.")
        timeout = _finite(raw["timeout_seconds"], name="timeout_seconds")
        if timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        candidate_court = CourtDatasetConfiguration.from_mapping(raw["candidate_court"])
        legacy_court = CourtDatasetConfiguration.from_mapping(raw["legacy_court"])
        if candidate_court.schema_version is not CourtDatasetSchemaVersion.V4:
            raise ValueError("candidate_court must be strict V4.")
        if legacy_court.schema_version is not CourtDatasetSchemaVersion.V3:
            raise ValueError("legacy_court must be strict V3.")
        return cls(
            action=action,
            scene_path=scene_path,
            alignment_path=alignment_path,
            pilot_manifest_path=manifest_path,
            pilot_output_root=pilot_output_root,
            final_evidence_root=final_evidence_root,
            annotation_root=annotation_root,
            audit_output_root=audit_output_root,
            source_video_path=source_video_path,
            frozen_config_path=frozen_config_path,
            frozen_authority=frozen_authority,
            render_executable=executable,
            timeout_seconds=timeout,
            environment=dict(environment),
            minimum_view_count=frozen_authority.minimum_pilot_views,
            split_seed=frozen_authority.pilot_seed,
            calibration_fraction=frozen_authority.calibration_fraction,
            candidate_court=candidate_court,
            legacy_court=legacy_court,
        )


@dataclass(frozen=True, slots=True)
class PilotEntry:
    opaque_id: str
    trajectory_group_id: str
    stratum: str
    valid_control: bool
    support_margin_m: float
    obstacle_clearance_m: float
    captured_camera_distance_m: float
    camera: NHTRenderCamera


@dataclass(frozen=True, slots=True)
class _ValidatedObservationInputs:
    entries: tuple[PilotEntry, ...]
    features: tuple[_PilotFeatureRecord, ...]
    reviewer_a: BlindAnnotation
    reviewer_b: BlindAnnotation
    adjudication: BlindAnnotation
    consensus: AnnotationConsensus
    lock: ObservationLock


@dataclass(frozen=True, slots=True)
class ValidatedPilotEvidence:
    """Complete frozen pilot join kept in memory across finalization steps."""

    entries: tuple[PilotEntry, ...]
    observations: tuple[QualityObservation, ...]
    reviewer_a: BlindAnnotation
    reviewer_b: BlindAnnotation
    adjudication: BlindAnnotation
    consensus: AnnotationConsensus
    calibration: QualityRuleCalibration
    decision: QualityDecisionReport
    annotation_sha256: Mapping[str, str]
    pilot_features_sha256: str
    blind_manifest_sha256: str
    candidate_plan: CourtDatasetPlanV4

    @property
    def consensus_by_id(self) -> Mapping[str, bool]:
        return {
            record.opaque_id: record.artifact_heavy for record in self.consensus.records
        }


@dataclass(frozen=True, slots=True)
class _PilotFeatureRecord:
    opaque_id: str
    trajectory_group_id: str
    stratum: str
    split: BenchmarkSplit
    valid_control: bool
    features: PublicQualityFeatures


@hydra_main(
    config_path="../configs",
    config_name="evaluate_court_trajectory_safety",
    version_base="1.3",
)
def main(config: DictConfig) -> int:  # pragma: no cover - Hydra CLI boundary
    """Render the frozen pilot and publish feature records without human labels."""
    runtime = BenchmarkConfiguration.from_config(config)
    client = NHTRenderClient()
    scene = client.validate_scene(runtime.scene_path)
    if runtime.action is BenchmarkAction.PREPARE_MANIFEST:
        manifest = prepare_pilot_manifest(scene=scene, runtime=runtime)
        save_json_atomic(manifest, runtime.pilot_manifest_path)
        manifest_records = manifest["records"]
        if not isinstance(manifest_records, list):
            raise TypeError("Prepared pilot manifest records are invalid.")
        print(f"scene={scene.scene_id}")
        print(f"pilot_views={len(manifest_records)}")
        print(f"manifest={runtime.pilot_manifest_path}")
        print("decision=pending_gpu_pilot_and_blind_annotations")
        return 0
    if runtime.action is BenchmarkAction.FREEZE_PILOT_OBSERVATIONS:
        lock = freeze_pilot_observations(scene=scene, runtime=runtime)
        print(f"scene={scene.scene_id}")
        print(f"observation_lock={runtime.frozen_config_path}")
        print(f"rgb_preview_inventory_sha256={lock.rgb_preview_inventory_sha256}")
        print("status=pilot_observations_frozen")
        return 0
    if runtime.action is not BenchmarkAction.RENDER_FROZEN_PILOT:
        _require_frozen_pilot_manifest(runtime)
    if runtime.action is BenchmarkAction.VALIDATE_COMPLETE_EVIDENCE:
        summary = validate_complete_evidence(scene=scene, runtime=runtime)
        print(f"scene={scene.scene_id}")
        print(f"decision={summary['decision']}")
        print(f"status={summary['status']}")
        print("validation=complete_evidence_read_only")
        return 0
    if runtime.action in {
        BenchmarkAction.FINALIZE_PILOT_EVIDENCE,
        BenchmarkAction.FINALIZE_COMPLETE_EVIDENCE,
    }:
        summary = finalize_evidence(
            scene=scene,
            runtime=runtime,
            require_final_dataset=(
                runtime.action is BenchmarkAction.FINALIZE_COMPLETE_EVIDENCE
            ),
        )
        print(f"scene={scene.scene_id}")
        print(f"decision={summary['decision']}")
        print(f"status={summary['status']}")
        print(f"summary={runtime.pilot_manifest_path.parent / 'summary.json'}")
        return 0
    if runtime.action is BenchmarkAction.RENDER_FINAL_V4:
        evidence = validate_pilot_evidence(scene=scene, runtime=runtime)
        render_final_v4(scene=scene, runtime=runtime, evidence=evidence)
        print(f"scene={scene.scene_id}")
        print(f"final_dataset={runtime.final_evidence_root}")
        print("status=final_v4_render_complete_pending_evidence_finalization")
        return 0
    if runtime.action is not BenchmarkAction.RENDER_FROZEN_PILOT:
        raise ValueError(f"Unsupported benchmark action: {runtime.action.value}")
    entries = _load_pilot_manifest(
        runtime.pilot_manifest_path,
        expected_scene_id=scene.scene_id,
        expected_seed=runtime.split_seed,
        minimum_view_count=runtime.minimum_view_count,
    )
    split_by_group = assign_group_held_out_splits(
        tuple(sorted({entry.trajectory_group_id for entry in entries})),
        seed=runtime.split_seed,
        calibration_fraction=runtime.calibration_fraction,
    )
    output_root = runtime.pilot_output_root
    if output_root.is_symlink() or output_root.exists():
        raise FileExistsError(
            f"Court safety pilot output must be a new ordinary directory: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    request = NHTRenderRequest(cameras=tuple(entry.camera for entry in entries))
    result = client.render(
        NHTRenderCommandRequest(
            scene_path=scene.scene_path,
            output_directory=output_root / "renders",
            arbitrary_cameras=request,
            arbitrary_request_path=output_root / "requests" / "pilot.json",
            executable=runtime.render_executable,
        ),
        environment=runtime.environment,
        timeout_seconds=runtime.timeout_seconds,
    )
    entry_by_id = {entry.opaque_id: entry for entry in entries}
    feature_records: list[dict[str, object]] = []
    blind_records: list[dict[str, str]] = []
    for record in result.records:
        entry = entry_by_id[record.camera_id]
        arrays = record.load_arrays()
        features = extract_public_quality_features(
            rgb=arrays.rgb,
            alpha=arrays.alpha,
            depth=arrays.depth,
            support_margin_m=entry.support_margin_m,
            obstacle_clearance_m=entry.obstacle_clearance_m,
            captured_camera_distance_m=entry.captured_camera_distance_m,
        )
        feature_records.append(
            {
                "opaque_id": entry.opaque_id,
                "trajectory_group_id": entry.trajectory_group_id,
                "stratum": entry.stratum,
                "split": split_by_group[entry.trajectory_group_id].value,
                "valid_control": entry.valid_control,
                "features": features.to_dict(),
            }
        )
        blind_records.append(
            {
                "opaque_id": entry.opaque_id,
                "rgb_preview": record.rgb_preview_path.relative_to(
                    output_root
                ).as_posix(),
            }
        )
    save_json_atomic(
        {
            "schema": PILOT_FEATURES_SCHEMA,
            "scene_id": scene.scene_id,
            "feature_definition_id": QUALITY_FEATURE_DEFINITION_ID,
            "pilot_manifest_sha256": _sha256(runtime.pilot_manifest_path),
            "records": feature_records,
        },
        output_root / "features.json",
    )
    save_json_atomic(
        {
            "schema": "court_trajectory_blind_review_manifest_v1",
            "scene_id": scene.scene_id,
            "records": blind_records,
        },
        output_root / "blind-review-manifest.json",
    )
    print(f"scene={scene.scene_id}")
    print(f"pilot_views={len(entries)}")
    print(f"features={output_root / 'features.json'}")
    print("decision=pending_two_blind_labels_and_adjudication")
    return 0


@dataclass(frozen=True, slots=True)
class _PilotSource:
    source_id: str
    trajectory_group_id: str
    stratum: str
    valid_control: bool
    camera_metric: SceneCamera


def prepare_pilot_manifest(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
) -> dict[str, object]:
    """Build and freeze 128 honest B00 decision inputs without rendering or labels."""
    alignment = load_alignment_result(runtime.alignment_path)
    candidate = build_court_dataset_plan(
        scene_id=scene.scene_id,
        profile="trajectory-safety-candidate",
        cameras=scene.cameras,
        layout=alignment.layout,
        configuration=runtime.candidate_court,
        metric_adapter=alignment.metric_adapter,
        points_scene=scene.points_scene,
    )
    legacy = build_court_dataset_plan(
        scene_id=scene.scene_id,
        profile="trajectory-safety-legacy",
        cameras=scene.cameras,
        layout=alignment.layout,
        configuration=runtime.legacy_court,
        metric_adapter=alignment.metric_adapter,
    )
    if not isinstance(candidate, CourtDatasetPlanV4):
        raise TypeError("Pilot candidate planning did not produce strict V4.")
    if not isinstance(legacy, CourtDatasetPlanV3):
        raise TypeError("Pilot legacy planning did not produce strict V3.")
    support_policy = runtime.candidate_court.support
    if support_policy is None:
        raise TypeError("Pilot candidate support policy is missing.")
    metric_cameras = tuple(
        SceneCamera(
            camera_id=camera.camera_id,
            source_frame_index=camera.source_frame_index,
            width=camera.width,
            height=camera.height,
            intrinsics=camera.intrinsics,
            camera_to_scene=alignment.metric_adapter.metric_from_nht_camera(
                camera.camera_to_scene
            ),
            image_path=camera.image_path,
        )
        for camera in scene.cameras
    )
    support_model = build_trajectory_support_model(
        cameras=metric_cameras,
        points_scene_m=alignment.metric_adapter.metric_from_nht_points(
            scene.points_scene[:, :3]
        ),
        policy=support_policy,
    )
    counts = {
        stratum: 22 if index < 2 else 21
        for index, stratum in enumerate(sorted(_REQUIRED_STRATA))
    }
    sources: list[_PilotSource] = []
    captured = _uniform_items(metric_cameras, counts["captured_control"])
    sources.extend(
        _PilotSource(
            source_id=f"captured:{camera.camera_id}",
            trajectory_group_id=(f"captured-run-{camera.source_frame_index // 32:03d}"),
            stratum="captured_control",
            valid_control=True,
            camera_metric=camera,
        )
        for camera in captured
    )
    legacy_samples = _uniform_items(legacy.samples, counts["legacy_orbit"])
    sources.extend(
        _PilotSource(
            source_id=f"legacy:{sample.sample_id}",
            trajectory_group_id=f"legacy:{sample.trajectory_group_id}",
            stratum="legacy_orbit",
            valid_control=False,
            camera_metric=sample.camera,
        )
        for sample in legacy_samples
    )
    safe_samples = _uniform_items(candidate.samples, counts["safe_v4_candidate"])
    sources.extend(
        _PilotSource(
            source_id=f"candidate:{sample.sample_id}",
            trajectory_group_id=f"candidate:{sample.trajectory_group_id}",
            stratum="safe_v4_candidate",
            valid_control=False,
            camera_metric=sample.camera,
        )
        for sample in safe_samples
    )
    interior_samples = _uniform_items(
        tuple(reversed(candidate.samples)), counts["support_interior"]
    )
    sources.extend(
        _PilotSource(
            source_id=f"interior:{sample.sample_id}",
            trajectory_group_id=f"interior:{sample.trajectory_group_id}",
            stratum="support_interior",
            valid_control=False,
            camera_metric=sample.camera,
        )
        for sample in interior_samples
    )
    boundary_bases = _uniform_items(metric_cameras, counts["support_boundary"])
    vertical = alignment.layout.courts[0].scene_from_court.matrix()[:3, 2]
    for index, base in enumerate(boundary_bases):
        boundary, exterior = _support_boundary_cameras(
            base,
            vertical_scene=vertical,
            support_model=support_model,
        )
        sources.append(
            _PilotSource(
                source_id=f"boundary:{base.camera_id}",
                trajectory_group_id=f"boundary-ray-{index // 3:03d}",
                stratum="support_boundary",
                valid_control=False,
                camera_metric=boundary,
            )
        )
        sources.append(
            _PilotSource(
                source_id=f"exterior:{base.camera_id}",
                trajectory_group_id=f"exterior-ray-{index // 3:03d}",
                stratum="support_exterior",
                valid_control=False,
                camera_metric=exterior,
            )
        )
    if len(sources) != 128:
        raise RuntimeError("Pilot preparation did not produce exactly 128 records.")
    source_by_id = {source.source_id: source for source in sources}
    opaque = opaque_review_ids(
        {
            stratum: tuple(
                source.source_id for source in sources if source.stratum == stratum
            )
            for stratum in sorted(_REQUIRED_STRATA)
        },
        seed=runtime.split_seed,
    )
    captured_centers = support_model.captured_camera_centers_m
    records: list[dict[str, object]] = []
    for opaque_id, stratum, source_id in opaque:
        source = source_by_id[source_id]
        center = source.camera_metric.camera_to_scene.matrix()[:3, 3]
        margin, clearance, _supported, _occupied = support_model.evaluate_point(center)
        nht_camera = SceneCamera(
            camera_id=opaque_id,
            source_frame_index=source.camera_metric.source_frame_index,
            width=source.camera_metric.width,
            height=source.camera_metric.height,
            intrinsics=source.camera_metric.intrinsics,
            camera_to_scene=alignment.metric_adapter.nht_from_metric_camera(
                source.camera_metric.camera_to_scene
            ),
            image_path="request-only",
        )
        records.append(
            {
                "opaque_id": opaque_id,
                "trajectory_group_id": source.trajectory_group_id,
                "stratum": stratum,
                "valid_control": source.valid_control,
                "support_margin_m": margin,
                "obstacle_clearance_m": clearance,
                "captured_camera_distance_m": float(
                    np.min(np.linalg.norm(captured_centers - center, axis=1))
                ),
                "camera": NHTRenderCamera.from_scene_camera(nht_camera).to_dict(),
            }
        )
    selected_violations = sum(
        len(group.safety_evaluation.violating_point_indices)
        + len(group.safety_evaluation.violating_segment_indices)
        for group in candidate.groups
    )
    return {
        "schema": PILOT_MANIFEST_SCHEMA,
        "status": "frozen",
        "scene_id": scene.scene_id,
        "seed": runtime.split_seed,
        "required_strata": sorted(_REQUIRED_STRATA),
        "tracked_evidence_path": _project_relative(runtime.pilot_manifest_path),
        "final_evidence_root": _project_relative(runtime.final_evidence_root),
        "provenance": {
            "scene_export_schema": "nht_standard_scene_v1",
            "captured_camera_count": len(scene.cameras),
            "public_point_count": scene.point_count,
            "nht_scene_units_per_metre": (
                alignment.metric_adapter.nht_scene_units_per_metre
            ),
            "metric_metres_per_nht_scene_unit": (
                1.0 / alignment.metric_adapter.nht_scene_units_per_metre
            ),
            "support_input_digest": support_model.summary.input_digest,
            "captured_camera_occupied_count": (
                support_model.summary.captured_camera_occupied_count
            ),
        },
        "decision_inputs": {
            "feature_definition_id": QUALITY_FEATURE_DEFINITION_ID,
            "legacy_plan_seed": legacy.policy.seed,
            "candidate_plan_seed": candidate.policy.seed,
            "candidate_count": len(candidate.candidate_safety_evaluations),
            "safe_candidate_count": sum(
                evaluation.safe for evaluation in candidate.candidate_safety_evaluations
            ),
            "selected_group_count": len(candidate.groups),
            "planned_frame_count": candidate.proposal_count,
            "selected_support_violation_count": selected_violations,
            "selected_trajectory_group_ids": [
                group.trajectory_group_id for group in candidate.groups
            ],
            **_release_plan_evidence(
                candidate,
                legacy_proposal_budget=legacy.policy.proposal_budget,
            ),
            **_semantic_phase_plan_evidence(candidate),
        },
        "records": records,
    }


def freeze_pilot_observations(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
) -> ObservationLock:
    """Explicitly replace only the tracked hash-only pilot observation lock."""
    validated = _validate_observation_inputs(
        runtime=runtime,
        scene_id=scene.scene_id,
        expected_lock=None,
    )
    payload = dict(runtime.frozen_authority.normalized_payload)
    payload["observation_lock"] = validated.lock.to_dict()
    save_json_atomic(payload, runtime.frozen_config_path)
    return validated.lock


def _validate_observation_inputs(
    *,
    runtime: BenchmarkConfiguration,
    scene_id: str,
    expected_lock: ObservationLock | None,
) -> _ValidatedObservationInputs:
    """Validate manifest-bound pilot files before freezing or consuming hashes."""
    _require_ordinary_file(runtime.pilot_manifest_path, name="pilot manifest")
    entries = _load_pilot_manifest(
        runtime.pilot_manifest_path,
        expected_scene_id=scene_id,
        expected_seed=runtime.split_seed,
        minimum_view_count=runtime.minimum_view_count,
    )
    manifest_sha256 = _sha256(runtime.pilot_manifest_path)
    feature_path = runtime.pilot_output_root / "features.json"
    blind_manifest_path = runtime.pilot_output_root / "blind-review-manifest.json"
    features = _load_pilot_features(
        feature_path,
        entries=entries,
        scene_id=scene_id,
        manifest_sha256=manifest_sha256,
        split_seed=runtime.split_seed,
        calibration_fraction=runtime.calibration_fraction,
    )
    preview_inventory_sha256 = _validate_blind_review_manifest(
        blind_manifest_path,
        pilot_root=runtime.pilot_output_root,
        entries=entries,
        scene_id=scene_id,
    )
    annotation_paths = _annotation_paths(runtime.annotation_root)
    raw_annotations = {
        name: _load_ordinary_json(path, name=f"{name} annotation")
        for name, path in annotation_paths.items()
    }
    opaque_ids = tuple(entry.opaque_id for entry in entries)
    reviewer_a = BlindAnnotation.from_mapping(
        raw_annotations["reviewer_a"],
        expected_schema=BLIND_ANNOTATION_SCHEMA,
        expected_manifest_sha256=manifest_sha256,
        expected_opaque_ids=opaque_ids,
    )
    reviewer_b = BlindAnnotation.from_mapping(
        raw_annotations["reviewer_b"],
        expected_schema=BLIND_ANNOTATION_SCHEMA,
        expected_manifest_sha256=manifest_sha256,
        expected_opaque_ids=opaque_ids,
    )
    decisions_a = {
        record.opaque_id: record.artifact_heavy for record in reviewer_a.records
    }
    decisions_b = {
        record.opaque_id: record.artifact_heavy for record in reviewer_b.records
    }
    disagreement_ids = tuple(
        opaque_id
        for opaque_id in opaque_ids
        if decisions_a[opaque_id] != decisions_b[opaque_id]
    )
    adjudication = BlindAnnotation.from_mapping(
        raw_annotations["adjudication"],
        expected_schema=BLIND_ADJUDICATION_SCHEMA,
        expected_manifest_sha256=manifest_sha256,
        expected_opaque_ids=disagreement_ids,
    )
    consensus = derive_annotation_consensus(
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        adjudication=adjudication,
    )
    lock = ObservationLock(
        pilot_manifest_sha256=manifest_sha256,
        pilot_features_sha256=_sha256(feature_path),
        blind_review_manifest_sha256=_sha256(blind_manifest_path),
        reviewer_a_sha256=_sha256(annotation_paths["reviewer_a"]),
        reviewer_b_sha256=_sha256(annotation_paths["reviewer_b"]),
        adjudication_sha256=_sha256(annotation_paths["adjudication"]),
        rgb_preview_inventory_sha256=preview_inventory_sha256,
    )
    if expected_lock is not None:
        _require_matching_observation_lock(observed=lock, expected=expected_lock)
    return _ValidatedObservationInputs(
        entries=entries,
        features=features,
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        adjudication=adjudication,
        consensus=consensus,
        lock=lock,
    )


def _require_matching_observation_lock(
    *,
    observed: ObservationLock,
    expected: ObservationLock,
) -> None:
    if observed != expected:
        raise ValueError(
            "Pilot observations changed from the explicit tracked observation lock."
        )


def validate_pilot_evidence(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
) -> ValidatedPilotEvidence:
    """Validate and join every frozen pilot input without trusting copied counts."""
    lock = _required_observation_lock(runtime)
    validated = _validate_observation_inputs(
        runtime=runtime,
        scene_id=scene.scene_id,
        expected_lock=lock,
    )
    consensus_by_id = {
        record.opaque_id: record.artifact_heavy
        for record in validated.consensus.records
    }
    observations = tuple(
        QualityObservation(
            opaque_id=record.opaque_id,
            trajectory_group_id=record.trajectory_group_id,
            stratum=record.stratum,
            split=record.split,
            artifact_heavy=consensus_by_id[record.opaque_id],
            valid_control=record.valid_control,
            features=record.features,
        )
        for record in validated.features
    )
    calibration = calibrate_quality_only_rule(observations)
    decision = evaluate_quality_only_rule(observations, rule=calibration.rule)
    candidate_plan = _rebuild_frozen_candidate_plan(scene=scene, runtime=runtime)
    evidence = ValidatedPilotEvidence(
        entries=validated.entries,
        observations=observations,
        reviewer_a=validated.reviewer_a,
        reviewer_b=validated.reviewer_b,
        adjudication=validated.adjudication,
        consensus=validated.consensus,
        calibration=calibration,
        decision=decision,
        annotation_sha256={
            "reviewer_a": lock.reviewer_a_sha256,
            "reviewer_b": lock.reviewer_b_sha256,
            "adjudication": lock.adjudication_sha256,
        },
        pilot_features_sha256=lock.pilot_features_sha256,
        blind_manifest_sha256=lock.blind_review_manifest_sha256,
        candidate_plan=candidate_plan,
    )
    return evidence


def finalize_evidence(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
    require_final_dataset: bool,
) -> dict[str, object]:
    """Publish canonical compact evidence and RGB-only audit contact sheets."""
    evidence = validate_pilot_evidence(scene=scene, runtime=runtime)
    audit_root = runtime.audit_output_root
    if audit_root.is_symlink() or (audit_root.exists() and not audit_root.is_dir()):
        raise ValueError("Audit output root must be an ordinary directory.")
    audit_root.mkdir(parents=True, exist_ok=True)
    contact_sheets = _write_contact_sheets(
        audit_root / "contact-sheets",
        pilot_root=runtime.pilot_output_root,
        evidence=evidence,
    )
    lock = _required_observation_lock(runtime)
    consensus_payload = evidence.consensus.to_dict()
    consensus_payload["pilot_manifest_sha256"] = lock.pilot_manifest_sha256
    decision_payload = {
        "schema": DERIVED_EVIDENCE_SCHEMA,
        "pilot_manifest_sha256": lock.pilot_manifest_sha256,
        "calibration": evidence.calibration.to_dict(),
        "held_out_decision": evidence.decision.to_dict(),
    }
    save_json_atomic(consensus_payload, audit_root / "consensus.json")
    save_json_atomic(decision_payload, audit_root / "quality-decision.json")
    source_authority = _validate_source_video_authority(
        scene=scene,
        source_video_path=runtime.source_video_path,
    )
    final_dataset = (
        _final_dataset_evidence(runtime=runtime, evidence=evidence)
        if require_final_dataset
        else None
    )
    summary = _tracked_summary(
        runtime=runtime,
        evidence=evidence,
        contact_sheets=contact_sheets,
        source_authority=source_authority,
        final_dataset=final_dataset,
        consensus_sha256=_canonical_sha256(consensus_payload),
        decision_sha256=_canonical_sha256(decision_payload),
    )
    derived_payload = {
        "schema": DERIVED_EVIDENCE_SCHEMA,
        "tracked_summary": summary,
        "consensus": consensus_payload,
        "quality_decision": decision_payload,
    }
    save_json_atomic(derived_payload, audit_root / "evidence.json")
    tracked_root = runtime.pilot_manifest_path.parent
    save_json_atomic(summary, tracked_root / "summary.json")
    (tracked_root / "report.md").write_text(
        _evidence_report(summary),
        encoding="utf-8",
    )
    return summary


def validate_complete_evidence(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
) -> dict[str, object]:
    """Validate the immutable complete evidence without rendering or writing."""
    evidence = validate_pilot_evidence(scene=scene, runtime=runtime)
    source_authority = _validate_source_video_authority(
        scene=scene,
        source_video_path=runtime.source_video_path,
    )
    final_dataset = _final_dataset_evidence(runtime=runtime, evidence=evidence)
    tracked_root = runtime.pilot_manifest_path.parent
    summary = _load_ordinary_json(
        tracked_root / "summary.json",
        name="tracked complete evidence summary",
    )
    contact_sheets = _validate_existing_contact_sheets(
        summary.get("representative_contact_sheets"),
        runtime=runtime,
        evidence=evidence,
    )
    lock = _required_observation_lock(runtime)
    consensus_payload = evidence.consensus.to_dict()
    consensus_payload["pilot_manifest_sha256"] = lock.pilot_manifest_sha256
    decision_payload = {
        "schema": DERIVED_EVIDENCE_SCHEMA,
        "pilot_manifest_sha256": lock.pilot_manifest_sha256,
        "calibration": evidence.calibration.to_dict(),
        "held_out_decision": evidence.decision.to_dict(),
    }
    audit_root = runtime.audit_output_root
    observed_consensus = _load_ordinary_json(
        audit_root / "consensus.json",
        name="derived consensus evidence",
    )
    observed_decision = _load_ordinary_json(
        audit_root / "quality-decision.json",
        name="derived quality decision evidence",
    )
    if observed_consensus != consensus_payload or observed_decision != decision_payload:
        raise ValueError("Derived complete evidence changed from frozen inputs.")
    expected_summary = _tracked_summary(
        runtime=runtime,
        evidence=evidence,
        contact_sheets=contact_sheets,
        source_authority=source_authority,
        final_dataset=final_dataset,
        consensus_sha256=_canonical_sha256(consensus_payload),
        decision_sha256=_canonical_sha256(decision_payload),
    )
    if summary != expected_summary:
        raise ValueError("Tracked complete evidence summary is inconsistent.")
    expected_derived = {
        "schema": DERIVED_EVIDENCE_SCHEMA,
        "tracked_summary": expected_summary,
        "consensus": consensus_payload,
        "quality_decision": decision_payload,
    }
    observed_derived = _load_ordinary_json(
        audit_root / "evidence.json",
        name="derived complete evidence",
    )
    if observed_derived != expected_derived:
        raise ValueError("Derived complete evidence payload is inconsistent.")
    report_path = tracked_root / "report.md"
    _require_ordinary_file(report_path, name="tracked complete evidence report")
    if report_path.read_text(encoding="utf-8") != _evidence_report(expected_summary):
        raise ValueError("Tracked complete evidence report is inconsistent.")
    return expected_summary


def render_final_v4(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
    evidence: ValidatedPilotEvidence,
) -> None:
    """Generate, render, assemble, and validate one fresh frozen B00 V4 dataset."""
    if evidence.decision.decision is not QualityDecision.QUALITY_ONLY_REJECTED:
        raise ValueError("Final V4 route requires the frozen quality-only rejection.")
    _validate_source_video_authority(
        scene=scene,
        source_video_path=runtime.source_video_path,
    )
    output_root = runtime.final_evidence_root
    staging_root = output_root.with_name(output_root.name + ".staging")
    for name, path in (("final output", output_root), ("final staging", staging_root)):
        if path.is_symlink() or path.exists():
            raise FileExistsError(f"Court safety {name} must be absent: {path}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=False, exist_ok=False)
    alignment = load_alignment_result(runtime.alignment_path)
    renderer = CourtNHTRenderer(
        executable=runtime.render_executable,
        client=NHTRenderClient(),
        environment=runtime.environment,
        timeout_seconds=runtime.timeout_seconds,
    )
    validated_scene = renderer.preflight(scene.scene_path)
    timer = PerformanceTimer()
    attempt_root = staging_root / "_attempt"
    rendered = renderer.render(
        plan=evidence.candidate_plan,
        scene=validated_scene,
        attempt_root=attempt_root,
        attempt_token=uuid.uuid4().hex,
        alignment=alignment,
    )
    assemble_court_dataset(
        staging_root,
        plan=evidence.candidate_plan,
        layout=alignment.layout,
        metric_adapter=alignment.metric_adapter,
        render_result=rendered,
        configuration=runtime.candidate_court,
        attempt_root=attempt_root,
        performance_timer=timer,
    )
    validate_court_dataset(
        staging_root,
        expected_plan=evidence.candidate_plan,
        expected_configuration=runtime.candidate_court,
        array_validation=CourtArrayValidationMode.HEADERS_ONLY,
    )
    staging_root.replace(output_root)


def _rebuild_frozen_candidate_plan(
    *,
    scene: StandardSceneExport,
    runtime: BenchmarkConfiguration,
) -> CourtDatasetPlanV4:
    alignment = load_alignment_result(runtime.alignment_path)
    candidate = build_court_dataset_plan(
        scene_id=scene.scene_id,
        profile="trajectory-safety-candidate",
        cameras=scene.cameras,
        layout=alignment.layout,
        configuration=runtime.candidate_court,
        metric_adapter=alignment.metric_adapter,
        points_scene=scene.points_scene,
    )
    if not isinstance(candidate, CourtDatasetPlanV4):
        raise TypeError("Frozen candidate plan did not produce strict V4.")
    manifest = _load_ordinary_json(
        runtime.pilot_manifest_path,
        name="frozen pilot manifest",
    )
    decision_inputs = manifest.get("decision_inputs")
    if not isinstance(decision_inputs, Mapping):
        raise ValueError("Frozen pilot decision inputs are invalid.")
    selected_violations = sum(
        len(group.safety_evaluation.violating_point_indices)
        + len(group.safety_evaluation.violating_segment_indices)
        for group in candidate.groups
    )
    observed = {
        "candidate_count": len(candidate.candidate_safety_evaluations),
        "safe_candidate_count": sum(
            evaluation.safe for evaluation in candidate.candidate_safety_evaluations
        ),
        "selected_group_count": len(candidate.groups),
        "planned_frame_count": candidate.proposal_count,
        "selected_support_violation_count": selected_violations,
        "selected_trajectory_group_ids": [
            group.trajectory_group_id for group in candidate.groups
        ],
        **_release_plan_evidence(
            candidate,
            legacy_proposal_budget=runtime.legacy_court.sampling.proposal_budget,
        ),
        **_semantic_phase_plan_evidence(candidate),
    }
    if any(decision_inputs.get(key) != value for key, value in observed.items()):
        raise ValueError(
            "Rebuilt V4 candidate plan changed from frozen pilot authority."
        )
    split_by_group: dict[str, str] = {}
    for sample in candidate.samples:
        split = sample.split.value
        previous = split_by_group.setdefault(sample.trajectory_group_id, split)
        if previous != split:
            raise ValueError("Rebuilt V4 candidate plan leaks a group across splits.")
    if len(split_by_group) != len(candidate.groups):
        raise ValueError("Rebuilt V4 candidate group inventory is incomplete.")
    return candidate


def _release_plan_evidence(
    plan: CourtDatasetPlanV4,
    *,
    legacy_proposal_budget: int,
) -> dict[str, object]:
    """Bind equal proposal budgets and typed release coverage without aliases."""
    candidate_budget = plan.policy.proposal_budget
    equal_budget = candidate_budget == legacy_proposal_budget
    if (
        candidate_budget != _REQUIRED_PROPOSAL_BUDGET
        or legacy_proposal_budget != _REQUIRED_PROPOSAL_BUDGET
        or not equal_budget
        or plan.required_coverage_shortfall
    ):
        raise ValueError("Frozen benchmark budget or required coverage is invalid.")
    return {
        "legacy_proposal_budget": legacy_proposal_budget,
        "candidate_proposal_budget": candidate_budget,
        "equal_proposal_budget": equal_budget,
        "required_coverage": plan.required_coverage.to_dict(),
        "selected_coverage": plan.selected_coverage.to_dict(),
        "required_coverage_shortfall": list(plan.required_coverage_shortfall),
        "optional_candidate_coverage_shortfall": list(
            plan.optional_candidate_coverage_shortfall
        ),
    }


def _semantic_phase_plan_evidence(
    plan: CourtDatasetPlanV4,
) -> dict[str, object]:
    """Return exact selection-time semantic authority for tracked evidence."""
    first = plan.candidate_semantic_phase_evaluations[0]
    return {
        "semantic_phase_evaluation_count": len(
            plan.candidate_semantic_phase_evaluations
        ),
        "semantic_phase_count": first.phase_count,
        "semantic_phase_inventory_digest": plan.semantic_phase_inventory_digest,
        "projected_semantic_valid_frame_count": (
            plan.projected_semantic_valid_frame_count
        ),
        "projected_semantic_rejected_frame_count": (
            plan.proposal_count - plan.projected_semantic_valid_frame_count
        ),
        "projected_semantic_valid_fraction": (plan.projected_semantic_valid_fraction),
        "selected_semantic_phases": [
            {
                "trajectory_group_id": group.trajectory_group_id,
                "phase_index": group.semantic_phase_evaluation.phase_index,
                "phase_count": group.semantic_phase_evaluation.phase_count,
                "coverage_mode": (
                    group.semantic_phase_evaluation.view.coverage_mode.value
                ),
                "look_at_height_m": (
                    group.semantic_phase_evaluation.view.look_at_height_m
                ),
                "expected_frame_count": (
                    group.semantic_phase_evaluation.expected_frame_count
                ),
                "expected_valid_frame_count": (
                    group.semantic_phase_evaluation.expected_valid_frame_count
                ),
                "rejection_counts": [
                    {"reason": reason, "count": count}
                    for reason, count in (
                        group.semantic_phase_evaluation.rejection_counts
                    )
                ],
                "disposition_digest": (
                    group.semantic_phase_evaluation.disposition_digest
                ),
            }
            for group in plan.groups
        ],
    }


def _load_pilot_features(
    path: Path,
    *,
    entries: Sequence[PilotEntry],
    scene_id: str,
    manifest_sha256: str,
    split_seed: int,
    calibration_fraction: float,
) -> tuple[_PilotFeatureRecord, ...]:
    raw = _load_ordinary_json(path, name="pilot features")
    if set(raw) != {
        "schema",
        "scene_id",
        "feature_definition_id",
        "pilot_manifest_sha256",
        "records",
    }:
        raise ValueError("Pilot feature file schema is invalid.")
    if (
        raw["schema"] != PILOT_FEATURES_SCHEMA
        or raw["scene_id"] != scene_id
        or raw["feature_definition_id"] != QUALITY_FEATURE_DEFINITION_ID
        or raw["pilot_manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("Pilot feature identity or manifest hash changed.")
    records_raw = raw["records"]
    if not isinstance(records_raw, Sequence) or isinstance(records_raw, (str, bytes)):
        raise TypeError("Pilot feature records must be a sequence.")
    if len(records_raw) != len(entries):
        raise ValueError("Pilot feature coverage changed.")
    expected_split = assign_group_held_out_splits(
        tuple(sorted({entry.trajectory_group_id for entry in entries})),
        seed=split_seed,
        calibration_fraction=calibration_fraction,
    )
    parsed: list[_PilotFeatureRecord] = []
    observed_ids: set[str] = set()
    for raw_record, entry in zip(records_raw, entries, strict=True):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "opaque_id",
            "trajectory_group_id",
            "stratum",
            "split",
            "valid_control",
            "features",
        }:
            raise ValueError("Pilot feature record schema is invalid.")
        opaque_id = raw_record["opaque_id"]
        if not isinstance(opaque_id, str):
            raise TypeError("Pilot feature opaque_id must be a string.")
        if opaque_id in observed_ids:
            raise ValueError("Pilot feature records contain duplicate opaque IDs.")
        observed_ids.add(opaque_id)
        if (
            opaque_id != entry.opaque_id
            or raw_record["trajectory_group_id"] != entry.trajectory_group_id
            or raw_record["stratum"] != entry.stratum
            or raw_record["valid_control"] is not entry.valid_control
        ):
            raise ValueError("Pilot feature join metadata changed from the manifest.")
        try:
            split = BenchmarkSplit(str(raw_record["split"]))
        except ValueError as error:
            raise ValueError("Pilot feature split is invalid.") from error
        if split is not expected_split[entry.trajectory_group_id]:
            raise ValueError(
                "Pilot feature group split changed from frozen assignment."
            )
        features = PublicQualityFeatures.from_mapping(raw_record["features"])
        if (
            features.support_margin_m != entry.support_margin_m
            or features.obstacle_clearance_m != entry.obstacle_clearance_m
            or features.captured_camera_distance_m != entry.captured_camera_distance_m
        ):
            raise ValueError(
                "Pilot support features changed from frozen manifest values."
            )
        parsed.append(
            _PilotFeatureRecord(
                opaque_id=opaque_id,
                trajectory_group_id=entry.trajectory_group_id,
                stratum=entry.stratum,
                split=split,
                valid_control=entry.valid_control,
                features=features,
            )
        )
    return tuple(parsed)


def _validate_blind_review_manifest(
    path: Path,
    *,
    pilot_root: Path,
    entries: Sequence[PilotEntry],
    scene_id: str,
) -> str:
    raw = _load_ordinary_json(path, name="blind-review manifest")
    if set(raw) != {"schema", "scene_id", "records"} or (
        raw["schema"] != BLIND_REVIEW_MANIFEST_SCHEMA or raw["scene_id"] != scene_id
    ):
        raise ValueError("Blind-review manifest schema or identity changed.")
    records = raw["records"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("Blind-review manifest records must be a sequence.")
    if len(records) != len(entries):
        raise ValueError("Blind-review manifest coverage changed.")
    preview_inventory: list[dict[str, object]] = []
    for value, entry in zip(records, entries, strict=True):
        expected_preview = f"renders/{entry.opaque_id}/rgb.png"
        if (
            not isinstance(value, Mapping)
            or set(value) != {"opaque_id", "rgb_preview"}
            or value["opaque_id"] != entry.opaque_id
            or value["rgb_preview"] != expected_preview
        ):
            raise ValueError(
                "Blind-review manifest order, schema, or reviewer blinding changed."
            )
        preview = pilot_root / expected_preview
        _require_ordinary_file_beneath(
            preview,
            root=pilot_root,
            name="blind RGB preview",
        )
        with Image.open(preview) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            content = hashlib.sha256()
            content.update(f"RGB:{width}:{height}:".encode("ascii"))
            content.update(rgb.tobytes())
        preview_inventory.append(
            {
                "opaque_id": entry.opaque_id,
                "width": width,
                "height": height,
                "rgb_content_sha256": content.hexdigest(),
            }
        )
    return _canonical_sha256(
        {
            "schema": "court_trajectory_rgb_preview_inventory_v1",
            "records": preview_inventory,
        }
    )


def _annotation_paths(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(
            f"Annotation root must be an ordinary directory: {root}"
        )
    paths = {
        "reviewer_a": root / "reviewer-a" / "annotation.json",
        "reviewer_b": root / "reviewer-b" / "annotation.json",
        "adjudication": root / "adjudicator" / "annotation.json",
    }
    for name, path in paths.items():
        _require_ordinary_file_beneath(path, root=root, name=f"{name} annotation")
    return paths


def _write_contact_sheets(
    root: Path,
    *,
    pilot_root: Path,
    evidence: ValidatedPilotEvidence,
) -> list[dict[str, object]]:
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("Contact-sheet root must be an ordinary directory.")
    root.mkdir(parents=True, exist_ok=True)
    consensus = evidence.consensus_by_id
    entry_by_id = {entry.opaque_id: entry for entry in evidence.entries}
    decisions_a = {
        record.opaque_id: record.artifact_heavy
        for record in evidence.reviewer_a.records
    }
    decisions_b = {
        record.opaque_id: record.artifact_heavy
        for record in evidence.reviewer_b.records
    }
    adjudicated = {
        record.opaque_id: record.artifact_heavy
        for record in evidence.adjudication.records
    }
    specifications = (
        (
            "consensus_artifact_heavy",
            tuple(sorted(opaque_id for opaque_id, label in consensus.items() if label)),
            lambda opaque_id: f"{opaque_id} consensus=artifact-heavy",
        ),
        (
            "non_artifact_controls",
            tuple(
                sorted(
                    opaque_id
                    for opaque_id, label in consensus.items()
                    if not label and entry_by_id[opaque_id].valid_control
                )
            ),
            lambda opaque_id: f"{opaque_id} consensus=control-valid",
        ),
        (
            "disagreements_adjudication",
            evidence.consensus.disagreement_ids,
            lambda opaque_id: (
                f"{opaque_id} A={int(decisions_a[opaque_id])} "
                f"B={int(decisions_b[opaque_id])} "
                f"C={int(adjudicated[opaque_id])}"
            ),
        ),
    )
    result: list[dict[str, object]] = []
    for kind, opaque_ids, caption in specifications:
        if not opaque_ids:
            raise ValueError(f"Contact-sheet category {kind!r} is empty.")
        path = root / f"{kind}.png"
        width, height = _render_contact_sheet(
            path,
            pilot_root=pilot_root,
            opaque_ids=opaque_ids,
            caption=caption,
        )
        result.append(
            {
                "kind": kind,
                "path": _stable_output_location(
                    path,
                    marker="outputs/court_trajectory_safety/",
                ),
                "sha256": _sha256(path),
                "image_count": len(opaque_ids),
                "mode": "RGB",
                "width": width,
                "height": height,
            }
        )
    return result


def _validate_existing_contact_sheets(
    value: object,
    *,
    runtime: BenchmarkConfiguration,
    evidence: ValidatedPilotEvidence,
) -> list[dict[str, object]]:
    """Read and validate finalized contact-sheet metadata without regenerating it."""
    if not isinstance(value, list):
        raise TypeError("Tracked contact-sheet inventory must be a list.")
    consensus = evidence.consensus_by_id
    expected_counts = {
        "consensus_artifact_heavy": sum(consensus.values()),
        "non_artifact_controls": sum(
            not consensus[entry.opaque_id] and entry.valid_control
            for entry in evidence.entries
        ),
        "disagreements_adjudication": len(evidence.consensus.disagreement_ids),
    }
    expected_kinds = tuple(expected_counts)
    if len(value) != len(expected_kinds):
        raise ValueError("Tracked contact-sheet inventory is incomplete.")
    root = runtime.audit_output_root / "contact-sheets"
    result: list[dict[str, object]] = []
    for expected_kind, raw in zip(expected_kinds, value, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != {
            "kind",
            "path",
            "sha256",
            "image_count",
            "mode",
            "width",
            "height",
        }:
            raise ValueError("Tracked contact-sheet record schema is invalid.")
        path = root / f"{expected_kind}.png"
        _require_ordinary_file_beneath(path, root=root, name="audit contact sheet")
        with Image.open(path) as image:
            observed_mode = image.mode
            observed_width, observed_height = image.size
        expected_path = _stable_output_location(
            path,
            marker="outputs/court_trajectory_safety/",
        )
        if (
            raw["kind"] != expected_kind
            or raw["path"] != expected_path
            or raw["sha256"] != _sha256(path)
            or raw["image_count"] != expected_counts[expected_kind]
            or raw["mode"] != observed_mode
            or raw["width"] != observed_width
            or raw["height"] != observed_height
            or observed_mode != "RGB"
        ):
            raise ValueError("Tracked contact-sheet evidence is inconsistent.")
        result.append(dict(raw))
    return result


def _render_contact_sheet(
    path: Path,
    *,
    pilot_root: Path,
    opaque_ids: Sequence[str],
    caption: Callable[[str], str],
) -> tuple[int, int]:
    columns = 4
    tile_width = 320
    image_height = 180
    label_height = 20
    rows = math.ceil(len(opaque_ids) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (image_height + label_height)),
        color=(0, 0, 0),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, opaque_id in enumerate(opaque_ids):
        source = pilot_root / "renders" / opaque_id / "rgb.png"
        _require_ordinary_file_beneath(source, root=pilot_root, name="contact RGB")
        with Image.open(source) as image:
            tile = image.convert("RGB").resize(
                (tile_width, image_height),
                resample=Image.Resampling.BILINEAR,
            )
        left = (index % columns) * tile_width
        top = (index // columns) * (image_height + label_height)
        sheet.paste(tile, (left, top))
        draw.rectangle(
            (
                left,
                top + image_height,
                left + tile_width,
                top + image_height + label_height,
            ),
            fill=(0, 0, 0),
        )
        draw.text(
            (left + 3, top + image_height + 3),
            str(caption(opaque_id)),
            fill=(255, 255, 255),
            font=font,
        )
    temporary = path.with_name(path.name + ".tmp")
    sheet.save(
        temporary,
        format="PNG",
        compress_level=9,
        optimize=False,
    )
    temporary.replace(path)
    with Image.open(path) as validated:
        if validated.mode != "RGB" or validated.size != sheet.size:
            raise ValueError("Contact sheet changed mode or dimensions after save.")
    return int(sheet.width), int(sheet.height)


def _tracked_summary(
    *,
    runtime: BenchmarkConfiguration,
    evidence: ValidatedPilotEvidence,
    contact_sheets: list[dict[str, object]],
    source_authority: Mapping[str, object],
    final_dataset: Mapping[str, object] | None,
    consensus_sha256: str,
    decision_sha256: str,
) -> dict[str, object]:
    observations = evidence.observations
    consensus = evidence.consensus_by_id
    calibration_count = sum(
        item.split is BenchmarkSplit.CALIBRATION for item in observations
    )
    held_out_count = len(observations) - calibration_count
    stratum_counts = Counter(item.stratum for item in observations)
    calibration_positive_count = sum(
        item.artifact_heavy
        for item in observations
        if item.split is BenchmarkSplit.CALIBRATION
    )
    held_out_positive_count = sum(
        item.artifact_heavy
        for item in observations
        if item.split is BenchmarkSplit.HELD_OUT
    )
    split_group_counts = Counter(
        group.split.value for group in evidence.candidate_plan.groups
    )
    candidate_artifact_count, candidate_count = _stratum_artifact_count(
        observations,
        stratum="safe_v4_candidate",
        split=BenchmarkSplit.HELD_OUT,
    )
    legacy_artifact_count, legacy_count = _stratum_artifact_count(
        observations,
        stratum="legacy_orbit",
        split=BenchmarkSplit.HELD_OUT,
    )
    candidate_rate = candidate_artifact_count / candidate_count
    legacy_rate = legacy_artifact_count / legacy_count
    gates = runtime.frozen_authority.geometry_release_gates
    maximum_rate_ratio = _finite(
        gates["maximum_candidate_to_legacy_artifact_rate_ratio"],
        name="maximum_candidate_to_legacy_artifact_rate_ratio",
    )
    rate_ratio = candidate_rate / legacy_rate if legacy_rate > 0.0 else None
    artifact_gate_passed = (
        candidate_rate == 0.0
        if legacy_rate == 0.0
        else candidate_rate <= maximum_rate_ratio * legacy_rate
    )
    geometry_metrics: dict[str, object] = {
        "candidate_count": len(evidence.candidate_plan.candidate_safety_evaluations),
        "safe_candidate_count": sum(
            item.safe for item in evidence.candidate_plan.candidate_safety_evaluations
        ),
        "planned_frame_count": evidence.candidate_plan.proposal_count,
        "trajectory_group_count": len(evidence.candidate_plan.groups),
        "selected_support_violation_count": sum(
            len(group.safety_evaluation.violating_point_indices)
            + len(group.safety_evaluation.violating_segment_indices)
            for group in evidence.candidate_plan.groups
        ),
        **_release_plan_evidence(
            evidence.candidate_plan,
            legacy_proposal_budget=runtime.legacy_court.sampling.proposal_budget,
        ),
        **_semantic_phase_plan_evidence(evidence.candidate_plan),
        "group_disjoint_splits": True,
        "split_group_counts": dict(sorted(split_group_counts.items())),
        "final_release_status": (
            "passed" if final_dataset is not None else "pending_final_gpu_dataset"
        ),
    }
    if final_dataset is not None:
        geometry_metrics["accepted_frame_count"] = final_dataset["accepted_frame_count"]
    summary: dict[str, object] = {
        "schema": TRACKED_EVIDENCE_SCHEMA,
        "status": (
            "complete"
            if final_dataset is not None
            else "quality_evidence_complete_final_v4_pending"
        ),
        "scene_id": runtime.frozen_authority.scene_id,
        "decision": evidence.decision.decision.value,
        "production_authority": "geometry_only",
        "pilot": {
            "record_count": len(evidence.entries),
            "manifest_sha256": _required_observation_lock(
                runtime
            ).pilot_manifest_sha256,
            "features_sha256": evidence.pilot_features_sha256,
            "blind_review_manifest_sha256": evidence.blind_manifest_sha256,
            "stratum_counts": dict(sorted(stratum_counts.items())),
            "calibration_record_count": calibration_count,
            "held_out_record_count": held_out_count,
            "calibration_group_count": len(evidence.decision.calibration_group_ids),
            "held_out_group_count": len(evidence.decision.held_out_group_ids),
            "calibration_group_ids": list(evidence.decision.calibration_group_ids),
            "held_out_group_ids": list(evidence.decision.held_out_group_ids),
        },
        "annotations": {
            "reviewer_a": {
                "reviewer_id": evidence.reviewer_a.reviewer_id,
                "record_count": len(evidence.reviewer_a.records),
                "positive_count": sum(
                    record.artifact_heavy for record in evidence.reviewer_a.records
                ),
                "sha256": evidence.annotation_sha256["reviewer_a"],
            },
            "reviewer_b": {
                "reviewer_id": evidence.reviewer_b.reviewer_id,
                "record_count": len(evidence.reviewer_b.records),
                "positive_count": sum(
                    record.artifact_heavy for record in evidence.reviewer_b.records
                ),
                "sha256": evidence.annotation_sha256["reviewer_b"],
            },
            "adjudication": {
                "reviewer_id": evidence.adjudication.reviewer_id,
                "record_count": len(evidence.adjudication.records),
                "sha256": evidence.annotation_sha256["adjudication"],
            },
            "disagreement_count": len(evidence.consensus.disagreement_ids),
            "consensus_positive_count": sum(consensus.values()),
            "calibration_positive_count": calibration_positive_count,
            "held_out_positive_count": held_out_positive_count,
            "label_inventory": {
                "calibration": {
                    "positive_count": calibration_positive_count,
                    "negative_count": calibration_count - calibration_positive_count,
                    "record_count": calibration_count,
                },
                "held_out": {
                    "positive_count": held_out_positive_count,
                    "negative_count": held_out_count - held_out_positive_count,
                    "record_count": held_out_count,
                },
            },
            "consensus_sha256": consensus_sha256,
        },
        "quality_calibration": evidence.calibration.to_dict(),
        "quality_decision": {
            **evidence.decision.to_dict(),
            "decision_sha256": decision_sha256,
        },
        "artifact_comparison": {
            "split": BenchmarkSplit.HELD_OUT.value,
            "candidate": {
                "stratum": "safe_v4_candidate",
                "artifact_heavy_count": candidate_artifact_count,
                "record_count": candidate_count,
                "artifact_heavy_rate": candidate_rate,
            },
            "legacy": {
                "stratum": "legacy_orbit",
                "artifact_heavy_count": legacy_artifact_count,
                "record_count": legacy_count,
                "artifact_heavy_rate": legacy_rate,
            },
            "candidate_to_legacy_rate_ratio": rate_ratio,
            "maximum_allowed_ratio": maximum_rate_ratio,
            "passed": artifact_gate_passed,
        },
        "geometry_metrics": geometry_metrics,
        "source_authority": dict(source_authority),
        "final_dataset": dict(final_dataset) if final_dataset is not None else None,
        "representative_contact_sheets": contact_sheets,
    }
    summary["canonical_evidence_sha256"] = _canonical_sha256(summary)
    return summary


def _stratum_artifact_count(
    observations: Sequence[QualityObservation],
    *,
    stratum: str,
    split: BenchmarkSplit,
) -> tuple[int, int]:
    values = tuple(
        item for item in observations if item.stratum == stratum and item.split is split
    )
    if not values:
        raise ValueError(f"Quality evidence stratum is empty: {stratum}.")
    return sum(item.artifact_heavy for item in values), len(values)


def _validate_source_video_authority(
    *,
    scene: StandardSceneExport,
    source_video_path: Path,
) -> dict[str, object]:
    if source_video_path.name != "B00.mp4":
        raise ValueError("Frozen B00 source authority must use the B00.mp4 name.")
    _require_ordinary_file(source_video_path, name="authoritative B00 source video")
    scene_root = scene.scene_path.parents[2]
    ingested_video = scene_root / "source" / "video.mp4"
    metadata_path = scene_root / "source" / "metadata.json"
    _require_ordinary_file(ingested_video, name="immutable ingested B00 video")
    metadata = _load_ordinary_json(metadata_path, name="B00 source metadata")
    if metadata.get("schema") != "canonical_scene_source_v1" or (
        metadata.get("scene_id") != "B00"
    ):
        raise ValueError("B00 source metadata identity changed.")
    prior_authority = metadata.get("configured_source_video")
    if not isinstance(prior_authority, str) or Path(prior_authority).name != (
        "tennis_court.mp4"
    ):
        raise ValueError("Expected stale B00 source authority was not found.")
    current_hash = _sha256(source_video_path)
    ingested_hash = _sha256(ingested_video)
    if current_hash != ingested_hash or source_video_path.stat().st_size != (
        ingested_video.stat().st_size
    ):
        raise ValueError("B00.mp4 does not match the immutable ingested scene source.")
    return {
        "status": "authoritative_b00_matches_immutable_ingest_copy",
        "current_source": "data/synthetic_data_generation/raw/B00.mp4",
        "stale_recorded_source_basename": "tennis_court.mp4",
        "sha256": current_hash,
        "size_bytes": source_video_path.stat().st_size,
    }


def _final_dataset_evidence(
    *,
    runtime: BenchmarkConfiguration,
    evidence: ValidatedPilotEvidence,
) -> dict[str, object]:
    root = runtime.final_evidence_root
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"Final V4 dataset is unavailable: {root}")
    report = validate_court_dataset(
        root,
        expected_configuration=runtime.candidate_court,
        array_validation=CourtArrayValidationMode.HEADERS_ONLY,
    )
    manifest_path = root / "dataset.json"
    manifest = _load_ordinary_json(manifest_path, name="final V4 dataset manifest")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Final V4 dataset metrics are invalid.")
    gates = runtime.frozen_authority.geometry_release_gates
    minimum_frames = _integer(gates["minimum_frames"], name="minimum_frames")
    maximum_frames = _integer(gates["maximum_frames"], name="maximum_frames")
    minimum_groups = _integer(
        gates["minimum_trajectory_groups"], name="minimum_trajectory_groups"
    )
    minimum_accepted_fraction = _finite(
        gates["minimum_accepted_fraction"], name="minimum_accepted_fraction"
    )
    split_leakage_count = _integer(
        metrics.get("split_leakage_count"), name="split_leakage_count"
    )
    performance = report.performance
    renderer_error_count = performance.post_render_rejected_sample_count
    renderer_invocation_count = performance.metrics.nht_invocations
    if (
        manifest.get("schema") != "canonical_court_dataset_v4"
        or report.proposal_count != evidence.candidate_plan.proposal_count
        or not minimum_frames <= report.accepted_frame_count <= maximum_frames
        or report.accepted_frame_count
        < math.ceil(minimum_accepted_fraction * report.proposal_count)
        or report.trajectory_group_count < minimum_groups
        or metrics.get("selected_safety_violation_count") != 0
        or split_leakage_count != 0
        or renderer_error_count != 0
        or renderer_invocation_count != performance.resolved_shard_count
    ):
        raise ValueError("Final V4 dataset failed a frozen geometry/release gate.")
    compact_files = [manifest_path]
    diagnostics = root / "diagnostics"
    if diagnostics.is_symlink() or not diagnostics.is_dir():
        raise FileNotFoundError("Final V4 diagnostics directory is missing.")
    compact_files.extend(
        path for path in sorted(diagnostics.rglob("*")) if path.is_file()
    )
    hashes = {
        path.relative_to(root).as_posix(): _sha256(path) for path in compact_files
    }
    return {
        "schema": "canonical_court_dataset_v4",
        "status": "complete",
        "path": _stable_output_location(
            root,
            marker="outputs/court_trajectory_safety/",
        ),
        "proposal_count": report.proposal_count,
        "accepted_frame_count": report.accepted_frame_count,
        "rejected_frame_count": report.rejected_frame_count,
        "accepted_fraction": report.accepted_fraction,
        "trajectory_group_count": report.trajectory_group_count,
        "resolved_shard_count": performance.resolved_shard_count,
        "split_leakage_count": split_leakage_count,
        "selected_support_violation_count": 0,
        "group_disjoint_splits": True,
        "renderer_error_count": renderer_error_count,
        "renderer_invocation_count": renderer_invocation_count,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "compact_evidence_sha256": _canonical_sha256(hashes),
        "compact_evidence_file_count": len(hashes),
    }


def _evidence_report(summary: Mapping[str, object]) -> str:
    annotations = _required_mapping(summary, "annotations")
    calibration = _required_mapping(summary, "quality_calibration")
    quality = _required_mapping(summary, "quality_decision")
    comparison = _required_mapping(summary, "artifact_comparison")
    geometry = _required_mapping(summary, "geometry_metrics")
    source = _required_mapping(summary, "source_authority")
    candidate = _required_mapping(comparison, "candidate")
    legacy = _required_mapping(comparison, "legacy")
    reviewer_a = _required_mapping(annotations, "reviewer_a")
    reviewer_b = _required_mapping(annotations, "reviewer_b")
    failure_reasons_raw = quality.get("failure_reasons")
    if (
        not isinstance(failure_reasons_raw, Sequence)
        or isinstance(failure_reasons_raw, (str, bytes))
        or any(not isinstance(reason, str) for reason in failure_reasons_raw)
    ):
        raise TypeError("Quality failure reasons must be a string sequence.")
    failure_reasons = tuple(failure_reasons_raw)
    predictive_raw = quality.get("predictive_metrics")
    if predictive_raw is None:
        quality_result = (
            f"Calibration evaluated {calibration['evaluated_candidate_count']} "
            "adjacent-midpoint/operator candidates, but no threshold family passed "
            "all frozen calibration gates. No rule, threshold, recall, precision, or "
            "other predictive metric was selected or reported. The explicit rejection "
            f"reasons are `{', '.join(failure_reasons)}`."
        )
    elif isinstance(predictive_raw, Mapping):
        rule = _required_mapping(quality, "rule")
        quality_result = (
            f"Calibration selected `{rule['feature_name']}` with operator "
            f"`{rule['operator']}` and threshold {rule['threshold']}. Held-out "
            f"TP={predictive_raw['true_positive']}, "
            f"FP={predictive_raw['false_positive']}, "
            f"TN={predictive_raw['true_negative']}, "
            f"FN={predictive_raw['false_negative']}; "
            f"recall={predictive_raw['recall']}, "
            f"precision={predictive_raw['precision']}, and valid-control "
            f"FPR={predictive_raw['valid_control_false_positive_rate']}. "
            f"The decision reasons are `{', '.join(failure_reasons)}`."
        )
    else:
        raise TypeError("Quality predictive_metrics must be a mapping or null.")
    final_dataset = summary["final_dataset"]
    final_line = (
        "- Final V4 dataset: pending the queued GPU generation/render/assembly run."
        if final_dataset is None
        else (
            "- Final V4 dataset: complete, "
            f"{_required_mapping(summary, 'final_dataset')['accepted_frame_count']}/"
            f"{_required_mapping(summary, 'final_dataset')['proposal_count']} "
            "accepted frames, "
            f"{_required_mapping(summary, 'final_dataset')['trajectory_group_count']} "
            "trajectory groups, "
            f"{_required_mapping(summary, 'final_dataset')['resolved_shard_count']} "
            "shards, "
            f"{_required_mapping(summary, 'final_dataset')['split_leakage_count']} "
            "split leaks, "
            f"{_required_mapping(summary, 'final_dataset')['selected_support_violation_count']} "
            "safety violations, and "
            f"{_required_mapping(summary, 'final_dataset')['renderer_error_count']} "
            "renderer errors across "
            f"{_required_mapping(summary, 'final_dataset')['renderer_invocation_count']} "
            "invocations."
        )
    )
    return (
        "# B00 Court trajectory safety decision report\n\n"
        f"Status: **{summary['status']}**. The quality-only decision is "
        f"**{summary['decision']}**; geometry remains authoritative.\n\n"
        "## Blind pilot evidence\n\n"
        f"- Reviewer A: {reviewer_a['record_count']} records, "
        f"{reviewer_a['positive_count']} artifact-heavy.\n"
        f"- Reviewer B: {reviewer_b['record_count']} records, "
        f"{reviewer_b['positive_count']} artifact-heavy.\n"
        f"- Disagreements/adjudications: {annotations['disagreement_count']}; "
        f"consensus positives: {annotations['consensus_positive_count']} "
        f"(calibration {annotations['calibration_positive_count']}, held-out "
        f"{annotations['held_out_positive_count']}).\n\n"
        "## Frozen quality-only result\n\n"
        f"{quality_result}\n\n"
        f"Held-out safe V4 candidates were {candidate['artifact_heavy_count']}/"
        f"{candidate['record_count']} artifact-heavy; held-out legacy views were "
        f"{legacy['artifact_heavy_count']}/{legacy['record_count']}.\n\n"
        "## Geometry and final route\n\n"
        f"- Frozen V4 plan: {geometry['planned_frame_count']} frames, "
        f"{geometry['trajectory_group_count']} trajectory groups, "
        f"{geometry['selected_support_violation_count']} support violations, and "
        f"group-disjoint splits={geometry['group_disjoint_splits']}.\n"
        f"- Source authority: `{source['current_source']}` matches the immutable "
        f"ingested copy at SHA-256 `{source['sha256']}`; the stale recorded "
        "`tennis_court.mp4` path is not used.\n"
        f"{final_line}\n"
    )


def _uniform_items(values: Sequence[_T], count: int) -> tuple[_T, ...]:
    if len(values) < count:
        raise ValueError("Pilot stratum source inventory is too small.")
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return tuple(values[int(index)] for index in indices)


def _support_boundary_cameras(
    camera: SceneCamera,
    *,
    vertical_scene: np.ndarray,
    support_model: TrajectorySupportModel,
) -> tuple[SceneCamera, SceneCamera]:
    direction = np.asarray(vertical_scene, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    start = camera.camera_to_scene.matrix()[:3, 3]
    lower = 0.0
    upper = 0.25
    while upper <= 16.0 and support_model.evaluate_point(start + direction * upper)[2]:
        lower = upper
        upper *= 2.0
    if upper > 16.0:
        raise ValueError("Could not find a bounded public-camera support boundary.")
    for _ in range(40):
        midpoint = (lower + upper) / 2.0
        if support_model.evaluate_point(start + direction * midpoint)[2]:
            lower = midpoint
        else:
            upper = midpoint
    boundary_center = start + direction * lower
    exterior_center = start + direction * (upper + 0.5)
    return (
        _translated_camera(camera, boundary_center, suffix="boundary"),
        _translated_camera(camera, exterior_center, suffix="exterior"),
    )


def _translated_camera(
    camera: SceneCamera,
    center: np.ndarray,
    *,
    suffix: str,
) -> SceneCamera:
    matrix = camera.camera_to_scene.matrix()
    matrix[:3, 3] = center
    return SceneCamera(
        camera_id=f"{camera.camera_id}-{suffix}",
        source_frame_index=camera.source_frame_index,
        width=camera.width,
        height=camera.height,
        intrinsics=camera.intrinsics,
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="request-only",
    )


def _project_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError("Evidence paths must stay beneath PROJECT_ROOT.") from error


def _load_pilot_manifest(
    path: Path,
    *,
    expected_scene_id: str,
    expected_seed: int,
    minimum_view_count: int,
) -> tuple[PilotEntry, ...]:
    raw = load_json(path)
    keys = {
        "schema",
        "status",
        "scene_id",
        "seed",
        "required_strata",
        "tracked_evidence_path",
        "final_evidence_root",
        "provenance",
        "decision_inputs",
        "records",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise ValueError("Court safety pilot manifest schema is invalid.")
    if (
        raw["schema"] != PILOT_MANIFEST_SCHEMA
        or raw["scene_id"] != expected_scene_id
        or raw["seed"] != expected_seed
    ):
        raise ValueError("Court safety pilot manifest identity is invalid.")
    if raw["status"] != "frozen":
        raise ValueError(
            "Court safety pilot manifest is not frozen; generate/freeze pilot cameras first."
        )
    if raw["required_strata"] != sorted(_REQUIRED_STRATA):
        raise ValueError("Court safety pilot required strata changed.")
    if (
        raw["tracked_evidence_path"] != _TRACKED_EVIDENCE_PATH
        or raw["final_evidence_root"] != _FINAL_EVIDENCE_ROOT
    ):
        raise ValueError("Court safety pilot evidence paths changed.")
    provenance = raw["provenance"]
    decision_inputs = raw["decision_inputs"]
    if (
        not isinstance(decision_inputs, Mapping)
        or set(decision_inputs) != _PILOT_DECISION_INPUT_KEYS
    ):
        raise ValueError("Court safety pilot decision input schema is invalid.")
    required_coverage = RequiredTrajectoryCoverage.from_mapping(
        decision_inputs["required_coverage"]
    )
    selected_coverage = SelectedTrajectoryCoverage.from_mapping(
        decision_inputs["selected_coverage"]
    )
    required_shortfall = decision_inputs["required_coverage_shortfall"]
    optional_shortfall = decision_inputs["optional_candidate_coverage_shortfall"]
    selected_group_ids = decision_inputs["selected_trajectory_group_ids"]
    nht_units_per_metre = (
        provenance.get("nht_scene_units_per_metre")
        if isinstance(provenance, Mapping)
        else None
    )
    metres_per_nht_unit = (
        provenance.get("metric_metres_per_nht_scene_unit")
        if isinstance(provenance, Mapping)
        else None
    )
    selected_phases = (
        decision_inputs.get("selected_semantic_phases")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    phase_count = (
        decision_inputs.get("semantic_phase_count")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    group_count = (
        decision_inputs.get("selected_group_count")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    planned_count = (
        decision_inputs.get("planned_frame_count")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    projected_valid_count = (
        decision_inputs.get("projected_semantic_valid_frame_count")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    projected_rejected_count = (
        decision_inputs.get("projected_semantic_rejected_frame_count")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    projected_fraction = (
        decision_inputs.get("projected_semantic_valid_fraction")
        if isinstance(decision_inputs, Mapping)
        else None
    )
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("captured_camera_occupied_count") != 0
        or isinstance(nht_units_per_metre, bool)
        or not isinstance(nht_units_per_metre, int | float)
        or float(nht_units_per_metre) <= 0.0
        or isinstance(metres_per_nht_unit, bool)
        or not isinstance(metres_per_nht_unit, int | float)
        or not math.isclose(
            float(nht_units_per_metre) * float(metres_per_nht_unit),
            1.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or decision_inputs.get("feature_definition_id")
        != QUALITY_FEATURE_DEFINITION_ID
        or decision_inputs.get("legacy_proposal_budget")
        != _REQUIRED_PROPOSAL_BUDGET
        or decision_inputs.get("candidate_proposal_budget")
        != _REQUIRED_PROPOSAL_BUDGET
        or decision_inputs.get("equal_proposal_budget") is not True
        or isinstance(planned_count, bool)
        or not isinstance(planned_count, int)
        or planned_count < 2_000
        or planned_count > 5_000
        or isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count < 24
        or selected_coverage.total_group_count != group_count
        or selected_coverage.total_frame_count != planned_count
        or decision_inputs.get("required_coverage") != required_coverage.to_dict()
        or decision_inputs.get("selected_coverage") != selected_coverage.to_dict()
        or not isinstance(required_shortfall, list)
        or required_shortfall
        or required_shortfall
        != list(required_coverage_shortfall(required_coverage, selected_coverage))
        or not isinstance(optional_shortfall, list)
        or optional_shortfall != sorted(set(optional_shortfall))
        or not isinstance(selected_group_ids, list)
        or len(selected_group_ids) != group_count
        or any(not isinstance(group_id, str) or not group_id for group_id in selected_group_ids)
        or len(set(selected_group_ids)) != group_count
        or decision_inputs.get("selected_support_violation_count") != 0
        or isinstance(phase_count, bool)
        or not isinstance(phase_count, int)
        or phase_count < 1
        or decision_inputs.get("semantic_phase_evaluation_count")
        != decision_inputs.get("safe_candidate_count", 0) * phase_count
        or not isinstance(selected_phases, list)
        or len(selected_phases) != group_count
        or not _valid_selected_semantic_phases(
            selected_phases,
            selected_trajectory_group_ids=selected_group_ids,
            phase_count=phase_count,
            planned_frame_count=planned_count,
            projected_valid_frame_count=(
                projected_valid_count
                if isinstance(projected_valid_count, int)
                and not isinstance(projected_valid_count, bool)
                else -1
            ),
        )
        or not isinstance(projected_valid_count, int)
        or isinstance(projected_valid_count, bool)
        or projected_valid_count < 2_000
        or not isinstance(projected_rejected_count, int)
        or isinstance(projected_rejected_count, bool)
        or projected_valid_count + projected_rejected_count != planned_count
        or isinstance(projected_fraction, bool)
        or not isinstance(projected_fraction, int | float)
        or float(projected_fraction) < 0.9
        or not math.isclose(
            float(projected_fraction),
            projected_valid_count / planned_count,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        )
        or not _is_sha256(decision_inputs.get("semantic_phase_inventory_digest"))
    ):
        raise ValueError("Court safety pilot decision inputs are not feasible.")
    records = raw["records"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("Court safety pilot records must be a sequence.")
    entries = tuple(_pilot_entry(record) for record in records)
    if len(entries) < minimum_view_count:
        raise ValueError("Court safety pilot contains too few views.")
    if len({entry.opaque_id for entry in entries}) != len(entries):
        raise ValueError("Court safety pilot opaque IDs must be unique.")
    if tuple(entry.opaque_id for entry in entries) != tuple(
        sorted(entry.opaque_id for entry in entries)
    ):
        raise ValueError("Court safety pilot records must use opaque-ID order.")
    if {entry.stratum for entry in entries} != _REQUIRED_STRATA:
        raise ValueError("Court safety pilot does not cover every frozen stratum.")
    if any(
        sum(entry.stratum == stratum for entry in entries) < _MINIMUM_VIEWS_PER_STRATUM
        for stratum in _REQUIRED_STRATA
    ):
        raise ValueError("Court safety pilot strata are under-sampled.")
    if len({entry.trajectory_group_id for entry in entries}) < 2:
        raise ValueError("Court safety pilot requires at least two trajectory groups.")
    return entries


def _valid_selected_semantic_phases(
    value: Sequence[object],
    *,
    selected_trajectory_group_ids: Sequence[object],
    phase_count: int,
    planned_frame_count: int,
    projected_valid_frame_count: int,
) -> bool:
    expected_keys = {
        "trajectory_group_id",
        "phase_index",
        "phase_count",
        "coverage_mode",
        "look_at_height_m",
        "expected_frame_count",
        "expected_valid_frame_count",
        "rejection_counts",
        "disposition_digest",
    }
    groups: set[str] = set()
    ordered_groups: list[str] = []
    coverage_modes: set[str] = set()
    observed_frames = 0
    observed_valid = 0
    for item in value:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            return False
        group_id = item["trajectory_group_id"]
        phase_index = item["phase_index"]
        frame_count = item["expected_frame_count"]
        valid_count = item["expected_valid_frame_count"]
        rejection_counts = item["rejection_counts"]
        coverage_mode = item["coverage_mode"]
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in groups
            or isinstance(phase_index, bool)
            or not isinstance(phase_index, int)
            or not 0 <= phase_index < phase_count
            or item["phase_count"] != phase_count
            or isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 8
            or isinstance(valid_count, bool)
            or not isinstance(valid_count, int)
            or not 0 < valid_count <= frame_count
            or coverage_mode not in {"full", "near_full", "partial"}
            or isinstance(item["look_at_height_m"], bool)
            or not isinstance(item["look_at_height_m"], int | float)
            or not math.isfinite(float(item["look_at_height_m"]))
            or not _is_sha256(item["disposition_digest"])
            or not isinstance(rejection_counts, list)
        ):
            return False
        parsed_rejections: dict[str, int] = {}
        for rejection in rejection_counts:
            if (
                not isinstance(rejection, Mapping)
                or set(rejection) != {"reason", "count"}
                or not isinstance(rejection["reason"], str)
                or not rejection["reason"]
                or rejection["reason"] in parsed_rejections
                or isinstance(rejection["count"], bool)
                or not isinstance(rejection["count"], int)
                or rejection["count"] <= 0
            ):
                return False
            parsed_rejections[rejection["reason"]] = rejection["count"]
        if sum(parsed_rejections.values()) != frame_count - valid_count:
            return False
        groups.add(group_id)
        ordered_groups.append(group_id)
        coverage_modes.add(str(coverage_mode))
        observed_frames += frame_count
        observed_valid += valid_count
    return (
        coverage_modes == {"full", "near_full", "partial"}
        and observed_frames == planned_frame_count
        and observed_valid == projected_valid_frame_count
        and tuple(ordered_groups) == tuple(selected_trajectory_group_ids)
    )


def _pilot_entry(value: object) -> PilotEntry:
    keys = {
        "opaque_id",
        "trajectory_group_id",
        "stratum",
        "valid_control",
        "support_margin_m",
        "obstacle_clearance_m",
        "captured_camera_distance_m",
        "camera",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Court safety pilot record keys are invalid.")
    opaque_id = value["opaque_id"]
    group_id = value["trajectory_group_id"]
    stratum = value["stratum"]
    if not isinstance(opaque_id, str) or _OPAQUE_ID.fullmatch(opaque_id) is None:
        raise ValueError("Pilot opaque_id is invalid.")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("Pilot trajectory_group_id is invalid.")
    if not isinstance(stratum, str) or stratum not in _REQUIRED_STRATA:
        raise ValueError("Pilot stratum is invalid.")
    if not isinstance(value["valid_control"], bool):
        raise TypeError("Pilot valid_control must be boolean.")
    if value["valid_control"] != (stratum == "captured_control"):
        raise ValueError("Pilot valid_control must identify captured controls exactly.")
    captured_distance = _finite(
        value["captured_camera_distance_m"],
        name="captured_camera_distance_m",
    )
    if captured_distance < 0.0:
        raise ValueError("captured_camera_distance_m must be non-negative.")
    return PilotEntry(
        opaque_id=opaque_id,
        trajectory_group_id=group_id,
        stratum=stratum,
        valid_control=value["valid_control"],
        support_margin_m=_finite(value["support_margin_m"], name="support_margin_m"),
        obstacle_clearance_m=_finite(
            value["obstacle_clearance_m"], name="obstacle_clearance_m"
        ),
        captured_camera_distance_m=captured_distance,
        camera=_render_camera(value["camera"], opaque_id=opaque_id),
    )


def _render_camera(value: object, *, opaque_id: str) -> NHTRenderCamera:
    if not isinstance(value, Mapping) or set(value) != {
        "camera_id",
        "width",
        "height",
        "intrinsics",
        "camera_to_scene",
    }:
        raise ValueError("Pilot NHT camera schema is invalid.")
    if value["camera_id"] != opaque_id:
        raise ValueError("Pilot NHT camera_id must equal opaque_id.")
    intrinsics = value["intrinsics"]
    if not isinstance(intrinsics, Mapping) or set(intrinsics) != {
        "model",
        "distortion_model",
        "params",
        "matrix",
    }:
        raise ValueError("Pilot NHT intrinsics schema is invalid.")
    matrix = np.asarray(intrinsics["matrix"], dtype=np.float64)
    params = np.asarray(intrinsics["params"], dtype=np.float64)
    camera_matrix = np.asarray(value["camera_to_scene"], dtype=np.float64)
    width = value["width"]
    height = value["height"]
    if (
        intrinsics["model"] != "PINHOLE"
        or intrinsics["distortion_model"] != "NONE"
        or matrix.shape != (3, 3)
        or params.shape != (4,)
        or not np.isfinite(params).all()
        or not np.allclose(
            params,
            (matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]),
            atol=1.0e-9,
            rtol=0.0,
        )
        or camera_matrix.shape != (4, 4)
        or isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
    ):
        raise ValueError("Pilot NHT camera values are invalid.")
    return NHTRenderCamera(
        camera_id=opaque_id,
        width=width,
        height=height,
        intrinsics=tuple(float(item) for item in matrix.ravel()),
        camera_to_scene=RigidTransform.from_matrix(camera_matrix),
    )


def _require_frozen_pilot_manifest(runtime: BenchmarkConfiguration) -> None:
    lock = _required_observation_lock(runtime)
    _require_ordinary_file(
        runtime.pilot_manifest_path,
        name="frozen pilot manifest",
    )
    observed = _sha256(runtime.pilot_manifest_path)
    if observed != lock.pilot_manifest_sha256:
        raise ValueError(
            "Frozen pilot-manifest SHA-256 changed; refusing benchmark finalization."
        )


def _required_observation_lock(runtime: BenchmarkConfiguration) -> ObservationLock:
    lock = runtime.frozen_authority.observation_lock
    if lock is None:
        raise ValueError(
            "Pilot observations are not frozen; run action=freeze_pilot_observations."
        )
    return lock


def _load_frozen_benchmark_authority(
    path: Path,
    *,
    allow_unfrozen_observation_lock: bool,
) -> FrozenBenchmarkAuthority:
    raw = _load_ordinary_json(path, name="frozen benchmark config")
    immutable_keys = {
        "scene_id",
        "pilot_seed",
        "minimum_pilot_views",
        "required_strata",
        "feature_definition_id",
        "group_split",
        "quality_only_thresholds",
        "geometry_release_gates",
    }
    current_keys = {"schema", "observation_lock", *immutable_keys}
    observation_lock: ObservationLock | None
    if set(raw) == current_keys and raw["schema"] == FROZEN_CONFIG_SCHEMA:
        if raw["observation_lock"] is None and allow_unfrozen_observation_lock:
            observation_lock = None
        else:
            observation_lock = ObservationLock.from_mapping(raw["observation_lock"])
    else:
        raise ValueError("Frozen benchmark config schema is invalid.")
    group_split = raw["group_split"]
    thresholds = raw["quality_only_thresholds"]
    gates = raw["geometry_release_gates"]
    expected_thresholds = {
        "minimum_recall": MINIMUM_HELD_OUT_RECALL,
        "minimum_precision": MINIMUM_HELD_OUT_PRECISION,
        "maximum_valid_control_false_positive_rate": (
            MAXIMUM_VALID_CONTROL_FALSE_POSITIVE_RATE
        ),
        "minimum_positive_labels": MINIMUM_HELD_OUT_POSITIVE_LABELS,
        "minimum_negative_labels": MINIMUM_HELD_OUT_NEGATIVE_LABELS,
    }
    expected_gates = {
        "minimum_frames": 2_000,
        "maximum_frames": 5_000,
        "minimum_accepted_fraction": 0.9,
        "minimum_trajectory_groups": 24,
        "maximum_candidate_to_legacy_artifact_rate_ratio": 0.5,
        "required_selected_support_violations": 0,
        "require_group_disjoint_splits": True,
    }
    if (
        raw["scene_id"] != "B00"
        or raw["pilot_seed"] != 823
        or raw["minimum_pilot_views"] != 128
        or raw["required_strata"] != sorted(_REQUIRED_STRATA)
        or raw["feature_definition_id"] != QUALITY_FEATURE_DEFINITION_ID
        or group_split
        != {"calibration_fraction": 0.5, "held_out_fraction": 0.5}
        or thresholds != expected_thresholds
        or gates != expected_gates
    ):
        raise ValueError("Frozen benchmark protocol or release authority changed.")
    normalized_payload = {
        "schema": FROZEN_CONFIG_SCHEMA,
        "scene_id": raw["scene_id"],
        "observation_lock": (
            observation_lock.to_dict() if observation_lock is not None else None
        ),
        "pilot_seed": raw["pilot_seed"],
        "minimum_pilot_views": raw["minimum_pilot_views"],
        "required_strata": raw["required_strata"],
        "feature_definition_id": raw["feature_definition_id"],
        "group_split": group_split,
        "quality_only_thresholds": thresholds,
        "geometry_release_gates": gates,
    }
    return FrozenBenchmarkAuthority(
        scene_id="B00",
        pilot_seed=823,
        minimum_pilot_views=128,
        required_strata=tuple(sorted(_REQUIRED_STRATA)),
        calibration_fraction=0.5,
        quality_only_thresholds=dict(expected_thresholds),
        geometry_release_gates=dict(expected_gates),
        observation_lock=observation_lock,
        normalized_payload=normalized_payload,
    )


def _load_ordinary_json(path: Path, *, name: str) -> dict[str, object]:
    _require_ordinary_file(path, name=name)
    value = load_json(path)
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed JSON object.")
    return value


def _require_ordinary_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} must be an ordinary file: {path}")


def _require_ordinary_file_beneath(path: Path, *, root: Path, name: str) -> None:
    _require_ordinary_file(path, name=name)
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_path == resolved_root or not resolved_path.is_relative_to(
        resolved_root
    ):
        raise ValueError(f"{name} escaped its frozen evidence root.")


def _required_mapping(
    value: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    result = value.get(key)
    if not isinstance(result, Mapping) or any(
        not isinstance(item, str) for item in result
    ):
        raise TypeError(f"Evidence field {key!r} must be a string-keyed mapping.")
    return result


def _stable_output_location(path: Path, *, marker: str) -> str:
    rendered = path.as_posix()
    index = rendered.find(marker)
    if index < 0:
        raise ValueError(f"Evidence path is outside the frozen output location: {path}")
    return rendered[index:]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path(value: object, *, name: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty path string.")
    requested = Path(value)
    if not requested.is_absolute():
        requested = PROJECT_ROOT / requested
    if requested.is_symlink():
        raise FileNotFoundError(f"{name} must not be a symbolic link: {requested}")
    resolved = requested.resolve(strict=must_exist)
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"{name} must be an ordinary file: {resolved}")
    return resolved


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
