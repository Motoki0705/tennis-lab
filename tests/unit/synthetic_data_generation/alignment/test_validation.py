"""Tests for fixed-path output, projection, binding, and stage validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    CameraEvidencePartition,
    CameraExclusionReason,
    CandidateEvidence,
    CorrespondenceSet,
    EvaluatedAlignment,
    ExcludedCameraDiagnostics,
    FixedCameraSelectionDiagnostics,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
)
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
from src.synthetic_data_generation.alignment.validation import (
    load_accepted_layout,
    validate_alignment_outputs,
    validate_court_transform_binding,
    validate_projection_equivalence,
    write_alignment_outputs,
)
from src.synthetic_data_generation.alignment.whole_court import (
    sample_court_line_template,
)
from src.synthetic_data_generation.pipeline.contracts import (
    DatasetTarget,
    ScenePipelineRequest,
    StageDefinition,
    StageExecutionContext,
    StageExecutionSummary,
    StageInput,
    StageName,
)
from src.synthetic_data_generation.pipeline.publication import (
    AtomicDirectoryPublication,
)
from src.synthetic_data_generation.pipeline.reuse import (
    RequiredOutputsReusablePublicationValidator,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


def test_fixed_outputs_round_trip_and_reject_cross_file_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir(parents=True)
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )

    validated = validate_alignment_outputs(staging)
    layout = load_accepted_layout(staging)
    assert validated.to_dict() == result.to_dict()
    assert layout.to_dict() == result.layout.to_dict()
    assert {path.name for path in staging.iterdir()} == {
        "ground-line-map.npz",
        "court-geometry.json",
        "alignment.json",
        "diagnostics",
        "line-heatmaps",
    }

    geometry_path = staging / "court-geometry.json"
    original_geometry = geometry_path.read_text(encoding="utf-8")
    geometry = json.loads(original_geometry)
    geometry["fit_camera_ids"] = ["tampered"]
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="court-geometry.json disagrees"):
        validate_alignment_outputs(staging)

    geometry_path.write_text(original_geometry, encoding="utf-8")
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["fit_points_scene"] = arrays["fit_points_scene"].copy()
    arrays["fit_points_scene"][0, 0] += 0.1
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Ground-line evidence"):
        validate_alignment_outputs(staging)


def test_ground_line_archive_rejects_wrong_dtype(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment" / "staging"
    staging.mkdir(parents=True)
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["fit_points_scene"] = arrays["fit_points_scene"].astype(np.float32)
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dtype float64"):
        validate_alignment_outputs(staging)


def test_current_ground_line_archive_requires_whole_court_policy(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment" / "staging"
    staging.mkdir(parents=True)
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    del arrays["whole_court_minimum_matches_per_offset_level"]
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="archive keys do not match"):
        validate_alignment_outputs(staging)


def test_excluded_camera_diagnostics_round_trip_and_reject_reason_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    excluded = ExcludedCameraDiagnostics(
        camera_id="excluded-holdout",
        original_partition=CameraEvidencePartition.HOLDOUT,
        selected_line_pixel_count=17,
        projected_line_point_count=4,
        reason=CameraExclusionReason.INSUFFICIENT_PROJECTED_POINTS,
    )
    no_lines = ExcludedCameraDiagnostics(
        camera_id="no-lines-holdout",
        original_partition=CameraEvidencePartition.HOLDOUT,
        selected_line_pixel_count=0,
        projected_line_point_count=0,
        reason=CameraExclusionReason.NO_DETECTED_LINE_PIXELS,
    )
    exclusions = (excluded, no_lines)
    evidence = replace(
        alignment_evidence,
        diagnostics=replace(
            alignment_evidence.diagnostics,
            selection=replace(
                alignment_evidence.diagnostics.selection,
                requested_camera_count=6,
                available_camera_count=6,
                holdout_cameras_per_unit=2,
                camera_prefix_ids=(
                    "holdout-0",
                    "fit-0",
                    excluded.camera_id,
                    "holdout-1",
                    "fit-1",
                    no_lines.camera_id,
                ),
                holdout_camera_ids=(
                    "holdout-0",
                    excluded.camera_id,
                    "holdout-1",
                    no_lines.camera_id,
                ),
                observed_camera_ids=(
                    "holdout-0",
                    "fit-0",
                    "holdout-1",
                    "fit-1",
                ),
                excluded_cameras=exclusions,
            ),
            excluded_cameras=exclusions,
        ),
    )
    result = fit_alignment(evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir()

    write_alignment_outputs(
        staging,
        evidence=evidence,
        result=result,
        heatmaps=_line_heatmaps(evidence),
    )
    validated = validate_alignment_outputs(staging)
    persisted_evidence = json.loads(
        (staging / "diagnostics/evidence.json").read_text(encoding="utf-8")
    )

    assert validated.to_dict() == result.to_dict()
    assert persisted_evidence["schema"] == "alignment_measured_evidence_v11"
    assert persisted_evidence["excluded_cameras"] == [
        item.to_dict() for item in exclusions
    ]

    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    arrays["diagnostic_excluded_camera_reasons"][0] = "unclassified"
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Excluded camera reasons are invalid"):
        validate_alignment_outputs(staging)


def test_fixed_selection_archive_rejects_requested_count_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    selection = alignment_evidence.diagnostics.selection
    assert FixedCameraSelectionDiagnostics.from_dict(selection.to_dict()) == selection
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir()
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload = json.loads(str(arrays["diagnostic_fixed_selection_json"].item()))
    payload["requested_camera_count"] = 5
    payload["available_camera_count"] = 5
    arrays["diagnostic_fixed_selection_json"] = np.asarray(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Requested camera count"):
        validate_alignment_outputs(staging)


def test_fixed_selection_archive_rejects_contiguous_partition_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir()
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )
    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload = json.loads(str(arrays["diagnostic_fixed_selection_json"].item()))
    payload["camera_prefix_ids"] = ["c00", "c01", "c02", "c03"]
    payload["fit_camera_ids"] = ["c00", "c01"]
    payload["holdout_camera_ids"] = ["c02", "c03"]
    payload["observed_camera_ids"] = ["c00", "c01", "c02", "c03"]
    arrays["diagnostic_fixed_selection_json"] = np.asarray(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unit slot rule"):
        validate_alignment_outputs(staging)


def test_proposal_search_archive_round_trip_rejects_weighted_gate_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir()
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )
    assert validate_alignment_outputs(staging).to_dict() == result.to_dict()

    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload = json.loads(str(arrays["diagnostic_proposal_search_json"].item()))
    assert payload == alignment_evidence.diagnostics.proposal_search.to_dict()
    payload["selected_candidate_explained_evidence_fractions"][0] = 0.01
    arrays["diagnostic_proposal_search_json"] = np.asarray(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="explained-evidence gate"):
        validate_alignment_outputs(staging)


def test_proposal_search_archive_rejects_refinement_rank_tampering(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir()
    write_alignment_outputs(
        staging,
        evidence=alignment_evidence,
        result=result,
        heatmaps=_line_heatmaps(alignment_evidence),
    )

    archive_path = staging / "ground-line-map.npz"
    with np.load(archive_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    payload = json.loads(str(arrays["diagnostic_proposal_search_json"].item()))
    payload["refinement_attempt_count"] = 2
    arrays["diagnostic_proposal_search_json"] = np.asarray(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    np.savez_compressed(archive_path, **arrays)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="attempts must equal rejected states"):
        validate_alignment_outputs(staging)


def test_proposal_search_rejects_inconsistent_frontier_history(
    alignment_evidence: AlignmentEvidence,
) -> None:
    proposal = alignment_evidence.diagnostics.proposal_search

    with pytest.raises(ValueError, match="disagree with their total"):
        replace(
            proposal,
            feasible_complete_state_counts=(1, 1),
        )
    with pytest.raises(ValueError, match="rank disagrees with per-depth ordering"):
        replace(
            proposal,
            selected_complete_state_candidate_count=1,
        )
    with pytest.raises(ValueError, match="expanded and pruned state counts"):
        replace(
            proposal,
            frontier_state_counts=(1,),
            feasible_complete_state_counts=(1,),
            selected_complete_state_candidate_count=1,
        )


def test_whole_court_policy_and_recomputed_metrics_round_trip(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    settings = WholeCourtEvidenceSettings(
        required_court_count=2,
        maximum_common_scale_relative_deviation=0.07290400972053463,
        maximum_center_refit_displacement_metres=(
            0.07290400972053463 * np.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH) + 0.3
        ),
        minimum_distinct_offset_levels=2,
        minimum_matches_per_offset_level=3,
        minimum_level_camera_count=2,
        minimum_secondary_tangential_span_metres=0.6,
        minimum_longitudinal_offset_span_metres=8.23,
        minimum_longitudinal_tangential_span_metres=12.8,
        minimum_transverse_offset_span_metres=12.8,
        minimum_transverse_tangential_span_metres=8.23,
        samples_per_metre=3.0,
        inlier_distance_metres=0.3,
        minimum_inlier_fraction=0.9,
        maximum_q95_error_metres=0.1,
        minimum_semantic_segment_inlier_fraction=0.8,
        minimum_center_separation_metres=10.97,
        maximum_footprint_overlap_fraction=1.0e-9,
    )
    template = sample_court_line_template(settings.samples_per_metre)
    points_court = np.column_stack((template, np.zeros(len(template))))
    baseline = fit_alignment(alignment_evidence, policy=alignment_policy)
    measured_metric = np.concatenate(
        [court.scene_from_court.apply(points_court) for court in baseline.candidates]
    )
    measured_nht = alignment_evidence.metric_adapter.nht_from_metric_points(
        measured_metric
    )
    measured_lines = tuple(
        replace(item, points_nht_scene=measured_nht)
        for item in alignment_evidence.measured_camera_lines
    )
    diagnostics = replace(
        alignment_evidence.diagnostics,
        cameras=tuple(
            replace(item, projected_line_point_count=len(measured_nht))
            for item in alignment_evidence.diagnostics.cameras
        ),
    )
    candidates = tuple(
        CandidateEvidence(
            court_instance_id=source.court_instance_id,
            candidate_id=source.candidate_id,
            fit=_exact_correspondences(
                points_court,
                scene_from_court=fitted.scene_from_court,
                camera_ids=alignment_evidence.partitions.fit_camera_ids,
            ),
            holdout=_exact_correspondences(
                points_court,
                scene_from_court=fitted.scene_from_court,
                camera_ids=alignment_evidence.partitions.holdout_camera_ids,
            ),
        )
        for source, fitted in zip(
            alignment_evidence.candidates,
            baseline.candidates,
            strict=True,
        )
    )
    evidence = replace(
        alignment_evidence,
        candidates=candidates,
        measured_camera_lines=measured_lines,
        diagnostics=diagnostics,
        whole_court_settings=settings,
    )
    result = fit_alignment(evidence, policy=alignment_policy)
    staging = tmp_path / "alignment"
    staging.mkdir()

    write_alignment_outputs(
        staging,
        evidence=evidence,
        result=result,
        heatmaps=_line_heatmaps(evidence),
    )
    validated = validate_alignment_outputs(staging)
    metrics = json.loads(
        (staging / "diagnostics/candidate-metrics.json").read_text(encoding="utf-8")
    )

    assert validated.to_dict() == result.to_dict()
    assert metrics["schema"] == "alignment_candidate_metrics_v6"
    assert metrics["whole_court"]["required_court_count_check"] is True
    assert all(
        candidate["common_scale_refit"]["accepted"]
        for candidate in metrics["whole_court"]["candidates"]
    )
    assert all(
        candidate["fit"]["identifiability"]["accepted"]
        and candidate["holdout"]["identifiability"]["accepted"]
        for candidate in metrics["whole_court"]["candidates"]
    )


def _exact_correspondences(
    points_court: np.ndarray,
    *,
    scene_from_court: RigidTransform,
    camera_ids: tuple[str, ...],
) -> CorrespondenceSet:
    repeated = np.concatenate([points_court for _camera_id in camera_ids])
    return CorrespondenceSet(
        points_court=repeated,
        points_scene=scene_from_court.apply(repeated),
        camera_ids=tuple(
            camera_id for camera_id in camera_ids for _point in points_court
        ),
    )


def test_projection_and_target_binding_share_the_exact_court_transform(
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    result = fit_alignment(alignment_evidence, policy=alignment_policy)
    court = result.layout.courts[0]
    camera_matrix = np.eye(4, dtype=np.float64)
    camera_matrix[:3, 3] = (0.0, 0.0, -10.0)
    camera_to_court = RigidTransform.from_matrix(camera_matrix)
    points = np.asarray(
        [[-2.0, -3.0, 0.0], [2.0, -3.0, 0.0], [0.0, 3.0, 0.0]],
        dtype=np.float64,
    )
    error = validate_projection_equivalence(
        court,
        camera_to_court=camera_to_court,
        intrinsics=(1000.0, 0.0, 640.0, 0.0, 1000.0, 360.0, 0.0, 0.0, 1.0),
        points_court=points,
    )
    assert error <= 1.0e-7

    selected = validate_court_transform_binding(
        result.layout,
        court_instance_id=court.court_instance_id,
        candidate_id=court.candidate_id,
        transforms={
            "camera": court.scene_from_court,
            "ball": court.scene_from_court,
            "player": court.scene_from_court,
        },
    )
    assert selected == court

    mismatched = RigidTransform.from_matrix(
        court.scene_from_court.matrix()
        @ np.asarray(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    with pytest.raises(ValueError, match="transform mismatch"):
        validate_court_transform_binding(
            result.layout,
            court_instance_id=court.court_instance_id,
            candidate_id=court.candidate_id,
            transforms={"camera": court.scene_from_court, "trajectory": mismatched},
        )


@dataclass
class _EvidenceSource:
    evidence: AlignmentEvidence
    policy: AlignmentAcceptancePolicy
    preflight_calls: int = 0
    evaluation_calls: int = 0

    def preflight(self, scene: StandardSceneExport) -> None:
        assert scene.scene_path.name == "scene.json"
        assert scene.scene_id == "scene-a"
        self.preflight_calls += 1

    def collect_evaluated(self, scene: StandardSceneExport) -> EvaluatedAlignment:
        assert scene.scene_path.name == "scene.json"
        assert scene.scene_id == "scene-a"
        self.evaluation_calls += 1
        return EvaluatedAlignment(
            evidence=self.evidence,
            result=fit_alignment(self.evidence, policy=self.policy),
            heatmaps=_line_heatmaps(self.evidence),
        )


@dataclass(frozen=True)
class _Context:
    request: ScenePipelineRequest
    stage: StageDefinition[StageExecutionSummary]
    owner_path: Path
    staging_path: Path


@dataclass(frozen=True)
class _UnusedLifecycle:
    def preflight(self, context: StageExecutionContext) -> None:
        pass

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        return StageExecutionSummary({})

    def validate(self, context: StageExecutionContext) -> None:
        pass


def test_stage_handler_consumes_fixed_export_and_writes_only_to_staging(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    context = _context(tmp_path)
    source = _EvidenceSource(alignment_evidence, alignment_policy)
    handler = AlignmentStageHandler(
        evidence_source=source,
        policy=alignment_policy,
        scene_loader=_load_test_scene,
    )

    handler.preflight(context)
    handler.preflight(context)
    summary = handler.execute(context)
    handler.validate(context)

    assert source.preflight_calls == 1
    assert source.evaluation_calls == 1
    assert summary.values["accepted_court_count"] == 2
    assert not context.owner_path.exists()
    assert (context.staging_path / "alignment.json").is_file()

    wrong_context = _Context(
        request=context.request,
        stage=context.stage,
        owner_path=context.owner_path,
        staging_path=tmp_path / "fallback",
    )
    with pytest.raises(ValueError, match="transaction snapshot"):
        handler.execute(wrong_context)


def test_stage_handler_leaves_no_outputs_when_holdout_gate_fails(
    tmp_path: Path,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    first = alignment_evidence.candidates[0]
    rejected = CandidateEvidence(
        court_instance_id=first.court_instance_id,
        candidate_id=first.candidate_id,
        fit=first.fit,
        holdout=CorrespondenceSet(
            points_court=first.holdout.points_court,
            points_scene=first.holdout.points_scene + 2.0,
            camera_ids=first.holdout.camera_ids,
        ),
    )
    second = alignment_evidence.candidates[1]
    second_rejected = replace(
        second,
        holdout=CorrespondenceSet(
            points_court=second.holdout.points_court,
            points_scene=second.holdout.points_scene + 2.0,
            camera_ids=second.holdout.camera_ids,
        ),
    )
    failed_evidence = AlignmentEvidence(
        partitions=alignment_evidence.partitions,
        candidates=(rejected, second_rejected),
        measured_camera_lines=alignment_evidence.measured_camera_lines,
        complex_points_scene=alignment_evidence.complex_points_scene,
        primary_candidate_id=None,
        metric_adapter=alignment_evidence.metric_adapter,
        diagnostics=alignment_evidence.diagnostics,
        whole_court_settings=alignment_evidence.whole_court_settings,
    )
    context = _context(tmp_path)
    handler = AlignmentStageHandler(
        evidence_source=_EvidenceSource(failed_evidence, alignment_policy),
        policy=alignment_policy,
        scene_loader=_load_test_scene,
    )

    with pytest.raises(ValueError, match='selected_correspondence_accepted":false'):
        handler.preflight(context)
    assert not any(context.staging_path.iterdir())


def _context(tmp_path: Path) -> _Context:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    owner = tmp_path / "scene-a" / "alignment"
    staging = owner.parent / ".transactions" / "alignment" / "snapshot"
    staging.mkdir(parents=True)
    export = owner.parent / "reconstruction" / "export"
    export.mkdir(parents=True)
    (export / "scene.json").write_text("{}", encoding="utf-8")
    (export / "cameras.json").write_text("{}", encoding="utf-8")
    np.save(export / "points_scene.npy", np.zeros((1, 6), dtype=np.float32))
    (export / "images").mkdir()
    (export / "model").mkdir()
    request = ScenePipelineRequest(
        scene_id="scene-a",
        source_video=video,
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.ALIGNMENT,
        config_schema="test-v1",
    )
    definition = StageDefinition(
        name=StageName.ALIGNMENT,
        dependencies=(StageName.RECONSTRUCTION,),
        owner_relative_path=Path("alignment"),
        required_inputs=(
            StageInput.resolved_configuration(),
            StageInput.stage_output(StageName.RECONSTRUCTION, "export"),
        ),
        required_outputs=(
            Path("ground-line-map.npz"),
            Path("court-geometry.json"),
            Path("alignment.json"),
            Path("diagnostics"),
            Path("line-heatmaps"),
        ),
        handler=_UnusedLifecycle(),
        publication=AtomicDirectoryPublication(),
        reusable_publication_validator=RequiredOutputsReusablePublicationValidator(),
        summary_type=StageExecutionSummary,
    )
    return _Context(
        request=request,
        stage=definition,
        owner_path=owner,
        staging_path=staging,
    )


def _load_test_scene(scene_path: str | Path) -> StandardSceneExport:
    path = Path(scene_path)
    return StandardSceneExport(
        scene_id="scene-a",
        export_root=path.parent,
        scene_path=path,
        cameras=(),
        points_scene=np.zeros((1, 6), dtype=np.float32),
        scene_from_sfm=tuple(float(value) for value in np.eye(4).ravel()),
        sfm_from_scene=tuple(float(value) for value in np.eye(4).ravel()),
        checkpoint_path=path.parent / "model" / "checkpoint.pt",
        runtime_config_path=path.parent / "model" / "config.json",
    )


def _line_heatmaps(evidence: AlignmentEvidence) -> AlignmentLineHeatmaps:
    selection = evidence.diagnostics.selection
    projected_counts = {
        item.camera_id: item.projected_line_point_count
        for item in evidence.diagnostics.cameras
    }
    projected_counts.update(
        {
            item.camera_id: item.projected_line_point_count
            for item in selection.excluded_cameras
        }
    )
    fit_ids = set(evidence.diagnostics.evaluation.fit_camera_ids)
    return AlignmentLineHeatmaps(
        bounds_uv=(-1.0, 1.0, -1.0, 1.0),
        grid_spacing=0.25,
        proximity_scale=0.35,
        proximity_power=2.0,
        views=tuple(
            AlignmentLineHeatmapView(
                camera_id=camera_id,
                probability=np.asarray(
                    [[0.0, 0.25], [0.5, 1.0]],
                    dtype=np.float32,
                ),
                points_uv=np.column_stack(
                    (
                        np.linspace(-0.9, 0.9, projected_counts[camera_id]),
                        np.linspace(0.9, -0.9, projected_counts[camera_id]),
                    )
                ).astype(np.float64),
                projected_probabilities=np.linspace(
                    0.5,
                    1.0,
                    projected_counts[camera_id],
                    dtype=np.float32,
                ),
                proximity_weights=np.full(
                    projected_counts[camera_id],
                    0.8,
                    dtype=np.float64,
                ),
                included_in_aggregate=camera_id in fit_ids,
            )
            for camera_id in selection.camera_prefix_ids
        ),
    )
