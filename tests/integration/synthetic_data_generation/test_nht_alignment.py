"""Integration of the public NHT export boundary and alignment holdout gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvaluationDiagnostics,
    AlignmentEvaluationOutcome,
    AlignmentEvaluationPolicy,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    CameraLineDiagnostics,
    CameraOwnershipRule,
    CameraSelectionPolicy,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    EvaluatedAlignment,
    FixedCameraSelectionDiagnostics,
    LineInferenceDeterminismDiagnostics,
    MeasuredCameraLines,
    MetricSceneAdapter,
    PartitionThresholds,
    ProposalSearchDiagnostics,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.settings import WholeCourtEvidenceSettings
from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
)
from src.synthetic_data_generation.pipeline import (
    CanonicalStageHandlers,
    DatasetTarget,
    ScenePipelineRequest,
    StageDefinition,
    StageExecutionSummary,
    StageName,
    StageRegistry,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.contracts import StageExecutionContext
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.utils.schema.court import HALF_DOUBLES_WIDTH, HALF_LENGTH


@dataclass(frozen=True)
class _Context:
    request: ScenePipelineRequest
    stage: StageDefinition[StageExecutionSummary]
    owner_path: Path
    staging_path: Path


@dataclass(frozen=True)
class _NoopHandler:
    """Supply an explicit lifecycle binding for definition-only test contexts."""

    stage: StageName

    def preflight(self, context: StageExecutionContext) -> None:
        pass

    def execute(self, context: StageExecutionContext) -> StageExecutionSummary:
        return StageExecutionSummary({"stage": self.stage.value})

    def validate(self, context: StageExecutionContext) -> None:
        pass


def _definitions() -> StageRegistry:
    return canonical_registry(
        CanonicalStageHandlers(
            ingest=_NoopHandler(StageName.INGEST),
            reconstruction=_NoopHandler(StageName.RECONSTRUCTION),
            alignment=_NoopHandler(StageName.ALIGNMENT),
            court_dataset=_NoopHandler(StageName.COURT_DATASET),
            blcs_dataset=_NoopHandler(StageName.BLCS_DATASET),
            plcs_dataset=_NoopHandler(StageName.PLCS_DATASET),
            report=_NoopHandler(StageName.REPORT),
        )
    )


@dataclass
class _EvidenceSource:
    evidence: AlignmentEvidence
    policy: AlignmentAcceptancePolicy

    def preflight(self, scene: StandardSceneExport) -> None:
        if scene.scene_id != "B00":
            raise ValueError("wrong scene")

    def collect_evaluated(self, scene: StandardSceneExport) -> EvaluatedAlignment:
        return EvaluatedAlignment(
            evidence=self.evidence,
            result=fit_alignment(self.evidence, policy=self.policy),
        )


def _candidate(index: int) -> CandidateEvidence:
    points = _identifiable_court_points()
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = float(index * 30)
    transform = RigidTransform.from_matrix(matrix)
    repeated = np.concatenate((points, points))
    scene_points = transform.apply(repeated)
    return CandidateEvidence(
        court_instance_id=f"court-{index}",
        candidate_id=f"candidate-{index}",
        fit=CorrespondenceSet(
            points_court=repeated,
            points_scene=scene_points,
            camera_ids=("fit-0",) * len(points) + ("fit-1",) * len(points),
        ),
        holdout=CorrespondenceSet(
            points_court=repeated,
            points_scene=scene_points,
            camera_ids=("holdout-0",) * len(points) + ("holdout-1",) * len(points),
        ),
    )


def _evidence() -> AlignmentEvidence:
    camera_ids = ("fit-0", "fit-1", "holdout-0", "holdout-1")
    whole_court_settings = _whole_court_settings()
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=("fit-0", "fit-1"),
            holdout_camera_ids=("holdout-0", "holdout-1"),
        ),
        candidates=(_candidate(0), _candidate(1)),
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera_id,
                points_nht_scene=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
                ),
            )
            for camera_id in camera_ids
        ),
        complex_points_scene=np.asarray(
            [[-10.0, -20.0, -1.0], [40.0, 20.0, 5.0]], dtype=np.float64
        ),
        primary_candidate_id="candidate-0",
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
        diagnostics=AlignmentEvidenceDiagnostics(
            cameras=tuple(
                CameraLineDiagnostics(
                    camera_id=camera_id,
                    selected_line_pixel_count=10,
                    projected_line_point_count=2,
                )
                for camera_id in camera_ids
            ),
            candidate_scales=tuple(
                CandidateScaleDiagnostics(
                    candidate_id=f"candidate-{index}",
                    nht_scene_units_per_metre=1.0,
                    template_score=0.9 - index * 0.1,
                    common_scale_refit_center_displacement_metres=0.0,
                    maximum_common_scale_refit_center_displacement_metres=(
                        whole_court_settings.maximum_center_refit_displacement_metres
                    ),
                    proposal_orientation_band_minimum_radians=-0.5,
                    proposal_orientation_band_maximum_radians=0.5,
                    proposal_residual_point_count_before_suppression=100,
                    proposal_residual_point_count_after_suppression=50,
                    native_center_uv=(float(index * 30), 0.0),
                    native_orientation_radians=0.0,
                )
                for index in range(2)
            ),
            common_nht_scene_units_per_metre=1.0,
            maximum_relative_scale_deviation=0.0,
            selection=FixedCameraSelectionDiagnostics(
                policy=CameraSelectionPolicy.NESTED_UNIFORM_PREFIX_V1,
                ownership_rule=(CameraOwnershipRule.FIXED_UNIT_EVEN_HOLDOUT_SLOTS_V1),
                requested_camera_count=4,
                available_camera_count=4,
                candidate_count=2,
                orientation_family_count=1,
                fit_cameras_per_unit=1,
                holdout_cameras_per_unit=1,
                camera_prefix_ids=("fit-0", "holdout-0", "fit-1", "holdout-1"),
                fit_camera_ids=("fit-0", "fit-1"),
                holdout_camera_ids=("holdout-0", "holdout-1"),
                observed_camera_ids=(
                    "fit-0",
                    "holdout-0",
                    "fit-1",
                    "holdout-1",
                ),
                excluded_cameras=(),
            ),
            evaluation=AlignmentEvaluationDiagnostics(
                policy=(
                    AlignmentEvaluationPolicy.FIT_SELECT_ONCE_HOLDOUT_EVALUATE_ONCE_V1
                ),
                evaluation_index=0,
                fit_camera_ids=("fit-0", "fit-1"),
                holdout_camera_ids=("holdout-0", "holdout-1"),
                candidate_ids=("candidate-0", "candidate-1"),
                fit_evaluation_count=1,
                holdout_evaluation_count=1,
                outcome=AlignmentEvaluationOutcome.FULL_VALIDATION_PASS,
            ),
            determinism=LineInferenceDeterminismDiagnostics(
                seed=42,
                device="cpu",
                model_eval=True,
                inference_mode=True,
                deterministic_algorithms=True,
                deterministic_warn_only=False,
                cudnn_benchmark=False,
                cudnn_deterministic=True,
                cuda_matmul_allow_tf32=False,
                cudnn_allow_tf32=False,
                cublas_workspace_config=None,
                torch_version="test",
                cuda_version=None,
                device_name="cpu",
                cross_hardware_bit_identity_claimed=False,
            ),
            proposal_search=ProposalSearchDiagnostics(
                orientation_band_count=1,
                center_tile_count=1,
                maximum_center_tile_width_scene_units=1.0,
                maximum_complete_branch_count=1,
                maximum_tile_state_count=2,
                maximum_residual_state_count=2,
                residual_state_count=2,
                residual_tree_build_count=2,
                explored_tile_state_count=2,
                geometrically_impossible_tile_state_count=0,
                feasible_proposal_count_before_deduplication=2,
                duplicate_proposal_count=0,
                retained_proposal_count=2,
                expanded_state_count=2,
                feasible_complete_state_count=1,
                selected_orientation_band_indices=(0, 0),
                selected_center_tile_indices=(0, 0),
                original_point_count=100,
                selected_residual_point_count=25,
                selected_explained_point_count=75,
                selected_native_score_sum=1.7,
            ),
            excluded_cameras=(),
        ),
        whole_court_settings=whole_court_settings,
    )


def _identifiable_court_points() -> NDArray[np.float64]:
    longitudinal = np.asarray(
        [
            (offset, tangential, 0.0)
            for offset in (-HALF_DOUBLES_WIDTH, HALF_DOUBLES_WIDTH)
            for tangential in np.linspace(-10.0, 10.0, 41)
        ],
        dtype=np.float64,
    )
    transverse = np.asarray(
        [
            (tangential, offset, 0.0)
            for offset in (-HALF_LENGTH, HALF_LENGTH)
            for tangential in np.linspace(-4.63, 4.63, 31)
        ],
        dtype=np.float64,
    )
    points: NDArray[np.float64] = np.asarray(
        np.concatenate((longitudinal, transverse)), dtype=np.float64
    )
    return points


def _whole_court_settings() -> WholeCourtEvidenceSettings:
    scale_tolerance = 0.07290400972053462
    localization_tolerance = 0.3
    return WholeCourtEvidenceSettings(
        required_court_count=2,
        maximum_common_scale_relative_deviation=scale_tolerance,
        maximum_center_refit_displacement_metres=(
            scale_tolerance * np.hypot(HALF_DOUBLES_WIDTH, HALF_LENGTH)
            + localization_tolerance
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
        inlier_distance_metres=localization_tolerance,
        minimum_inlier_fraction=0.9,
        maximum_q95_error_metres=0.1,
        minimum_semantic_segment_inlier_fraction=0.8,
        minimum_center_separation_metres=10.97,
        maximum_footprint_overlap_fraction=1.0e-9,
    )


def _policy(*, holdout_rms: float = 0.01) -> AlignmentAcceptancePolicy:
    fit = PartitionThresholds(
        minimum_camera_count=2,
        minimum_correspondence_count=6,
        inlier_distance_m=0.01,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.01,
        maximum_q95_error_m=0.01,
    )
    holdout = PartitionThresholds(
        minimum_camera_count=2,
        minimum_correspondence_count=6,
        inlier_distance_m=holdout_rms,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=holdout_rms,
        maximum_q95_error_m=holdout_rms,
    )
    return AlignmentAcceptancePolicy(fit=fit, holdout=holdout)


def _context(tmp_path: Path) -> _Context:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    request = ScenePipelineRequest(
        scene_id="B00",
        source_video=video.resolve(),
        targets=frozenset({DatasetTarget.COURT}),
        from_stage=StageName.ALIGNMENT,
        config_schema="canonical_scene_pipeline_v1",
    )
    owner = tmp_path / "B00/alignment"
    staging = owner.parent / ".transactions/alignment/snapshot"
    staging.mkdir(parents=True)
    export = owner.parent / "reconstruction/export"
    (export / "images").mkdir(parents=True)
    (export / "model").mkdir()
    for name in ("scene.json", "cameras.json", "points_scene.npy"):
        (export / name).write_bytes(b"boundary")
    return _Context(
        request=request,
        stage=_definitions().definition(StageName.ALIGNMENT),
        owner_path=owner,
        staging_path=staging,
    )


def _scene(context: _Context) -> StandardSceneExport:
    export = context.owner_path.parent / "reconstruction/export"
    return StandardSceneExport(
        scene_id="B00",
        export_root=export,
        scene_path=export / "scene.json",
        cameras=(),
        points_scene=np.zeros((1, 6), dtype=np.float32),
        scene_from_sfm=tuple(np.eye(4, dtype=np.float64).reshape(-1)),
        sfm_from_scene=tuple(np.eye(4, dtype=np.float64).reshape(-1)),
        checkpoint_path=export / "model/checkpoint.pt",
        runtime_config_path=export / "model/runtime-config.json",
    )


def test_alignment_consumes_only_public_export_and_preserves_all_accepted_courts(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    handler = AlignmentStageHandler(
        evidence_source=_EvidenceSource(_evidence(), _policy()),
        policy=_policy(),
        scene_loader=lambda _path: _scene(context),
    )

    handler.preflight(context)
    summary = handler.execute(context)
    handler.validate(context)
    layout = validate_alignment_outputs(context.staging_path).layout

    assert summary.values["accepted_court_count"] == 2
    assert tuple(court.court_instance_id for court in layout.courts) == (
        "court-0",
        "court-1",
    )
    for court in layout.courts:
        product = court.scene_from_court.matrix() @ court.court_from_scene.matrix()
        np.testing.assert_allclose(product, np.eye(4), atol=1.0e-8)


def test_holdout_failure_is_rejected_before_alignment_output_mutation(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    evidence = _evidence()
    bad = evidence.candidates[0].holdout.points_scene.copy()
    bad[:, 2] += 0.2
    rejected = CandidateEvidence(
        court_instance_id=evidence.candidates[0].court_instance_id,
        candidate_id=evidence.candidates[0].candidate_id,
        fit=evidence.candidates[0].fit,
        holdout=CorrespondenceSet(
            points_court=evidence.candidates[0].holdout.points_court,
            points_scene=bad,
            camera_ids=evidence.candidates[0].holdout.camera_ids,
        ),
    )
    altered = AlignmentEvidence(
        partitions=evidence.partitions,
        candidates=(rejected, evidence.candidates[1]),
        measured_camera_lines=evidence.measured_camera_lines,
        complex_points_scene=evidence.complex_points_scene,
        primary_candidate_id=evidence.primary_candidate_id,
        metric_adapter=evidence.metric_adapter,
        diagnostics=evidence.diagnostics,
        whole_court_settings=evidence.whole_court_settings,
    )
    handler = AlignmentStageHandler(
        evidence_source=_EvidenceSource(altered, _policy()),
        policy=_policy(),
        scene_loader=lambda _path: _scene(context),
    )

    with pytest.raises(ValueError, match="failed acceptance"):
        handler.preflight(context)

    assert not tuple(context.staging_path.iterdir())
