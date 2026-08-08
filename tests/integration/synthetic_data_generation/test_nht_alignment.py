"""Integration of the public NHT export boundary and alignment holdout gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    CameraLineDiagnostics,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    MeasuredCameraLines,
    MetricSceneAdapter,
    PartitionThresholds,
)
from src.synthetic_data_generation.alignment.handler import AlignmentStageHandler
from src.synthetic_data_generation.alignment.validation import (
    validate_alignment_outputs,
)
from src.synthetic_data_generation.pipeline import (
    DatasetTarget,
    ScenePipelineRequest,
    StageName,
    canonical_registry,
)
from src.synthetic_data_generation.pipeline.contracts import StageSpec
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform


@dataclass(frozen=True)
class _Context:
    request: ScenePipelineRequest
    stage: StageSpec
    owner_path: Path
    staging_path: Path


@dataclass
class _EvidenceSource:
    evidence: AlignmentEvidence

    def preflight(self, scene: StandardSceneExport) -> None:
        if scene.scene_id != "B00":
            raise ValueError("wrong scene")

    def collect(self, scene: StandardSceneExport) -> AlignmentEvidence:
        self.preflight(scene)
        return self.evidence


def _candidate(index: int) -> CandidateEvidence:
    points = np.asarray(
        [
            [-5.0, -10.0, 0.0],
            [5.0, -10.0, 0.0],
            [5.0, 10.0, 0.0],
            [-5.0, 10.0, 0.0],
            [0.0, -6.0, 0.0],
            [0.0, 6.0, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = float(index * 30)
    transform = RigidTransform.from_matrix(matrix)
    scene_points = transform.apply(points)
    camera_ids = ("fit-0", "fit-1", "fit-0", "fit-1", "fit-0", "fit-1")
    holdout_ids = (
        "holdout-0",
        "holdout-1",
        "holdout-0",
        "holdout-1",
        "holdout-0",
        "holdout-1",
    )
    return CandidateEvidence(
        court_instance_id=f"court-{index}",
        candidate_id=f"candidate-{index}",
        fit=CorrespondenceSet(
            points_court=points,
            points_scene=scene_points,
            camera_ids=camera_ids,
        ),
        holdout=CorrespondenceSet(
            points_court=points,
            points_scene=scene_points,
            camera_ids=holdout_ids,
        ),
    )


def _evidence() -> AlignmentEvidence:
    camera_ids = ("fit-0", "fit-1", "holdout-0", "holdout-1")
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
                )
                for index in range(2)
            ),
            common_nht_scene_units_per_metre=1.0,
            maximum_relative_scale_deviation=0.0,
        ),
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
    staging = owner / "staging"
    staging.mkdir(parents=True)
    export = owner.parent / "reconstruction/export"
    (export / "images").mkdir(parents=True)
    (export / "model").mkdir()
    for name in ("scene.json", "cameras.json", "points_scene.npy"):
        (export / name).write_bytes(b"boundary")
    return _Context(
        request=request,
        stage=canonical_registry().spec(StageName.ALIGNMENT),
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
        evidence_source=_EvidenceSource(_evidence()),
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
    )
    handler = AlignmentStageHandler(
        evidence_source=_EvidenceSource(altered),
        policy=_policy(),
        scene_loader=lambda _path: _scene(context),
    )

    with pytest.raises(ValueError, match="failed acceptance"):
        handler.preflight(context)

    assert not tuple(context.staging_path.iterdir())
