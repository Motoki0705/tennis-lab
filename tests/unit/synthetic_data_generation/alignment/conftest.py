"""Focused semantic alignment fixtures."""

from __future__ import annotations

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
from src.synthetic_data_generation.scene_contract import RigidTransform


def _rigid(*, angle: float, translation: tuple[float, float, float]) -> RigidTransform:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix = np.asarray(
        [
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return RigidTransform.from_matrix(matrix)


def _candidate(
    *,
    court_id: str,
    candidate_id: str,
    transform: RigidTransform,
) -> CandidateEvidence:
    fit_court = np.asarray(
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
    holdout_court = np.asarray(
        [
            [-4.0, -8.0, 0.0],
            [4.0, -8.0, 0.0],
            [4.0, 8.0, 0.0],
            [-4.0, 8.0, 0.0],
            [-5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return CandidateEvidence(
        court_instance_id=court_id,
        candidate_id=candidate_id,
        fit=CorrespondenceSet(
            points_court=fit_court,
            points_scene=transform.apply(fit_court),
            camera_ids=("fit-0", "fit-1", "fit-0", "fit-1", "fit-0", "fit-1"),
        ),
        holdout=CorrespondenceSet(
            points_court=holdout_court,
            points_scene=transform.apply(holdout_court),
            camera_ids=(
                "holdout-0",
                "holdout-1",
                "holdout-0",
                "holdout-1",
                "holdout-0",
                "holdout-1",
            ),
        ),
    )


@pytest.fixture
def alignment_policy() -> AlignmentAcceptancePolicy:
    """Return separate strict fit and holdout gates."""
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
        inlier_distance_m=0.02,
        minimum_inlier_fraction=1.0,
        maximum_rms_error_m=0.02,
        maximum_q95_error_m=0.02,
    )
    return AlignmentAcceptancePolicy(fit=fit, holdout=holdout)


@pytest.fixture
def alignment_evidence() -> AlignmentEvidence:
    """Return two independently accepted physical courts."""
    first = _candidate(
        court_id="court-a",
        candidate_id="candidate-a",
        transform=_rigid(angle=0.2, translation=(1.0, 2.0, 0.5)),
    )
    second = _candidate(
        court_id="court-b",
        candidate_id="candidate-b",
        transform=_rigid(angle=-0.15, translation=(20.0, 1.0, 0.7)),
    )
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=("fit-0", "fit-1"),
            holdout_camera_ids=("holdout-0", "holdout-1"),
        ),
        candidates=(first, second),
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera_id,
                points_nht_scene=np.column_stack(
                    (
                        np.linspace(-1.0, 1.0, 80),
                        np.linspace(1.0, -1.0, 80),
                        np.full(80, index, dtype=np.float64),
                    )
                ),
            )
            for index, camera_id in enumerate(
                ("fit-0", "fit-1", "holdout-0", "holdout-1")
            )
        ),
        complex_points_scene=np.asarray(
            [[-8.0, -15.0, -1.0], [30.0, 16.0, 5.0]], dtype=np.float64
        ),
        primary_candidate_id="candidate-a",
        metric_adapter=MetricSceneAdapter.from_nht_scene_from_metric_scene(
            np.eye(4, dtype=np.float64)
        ),
        diagnostics=AlignmentEvidenceDiagnostics(
            cameras=tuple(
                CameraLineDiagnostics(
                    camera_id=camera_id,
                    selected_line_pixel_count=100,
                    projected_line_point_count=80,
                )
                for camera_id in ("fit-0", "fit-1", "holdout-0", "holdout-1")
            ),
            candidate_scales=(
                CandidateScaleDiagnostics(
                    candidate_id="candidate-a",
                    nht_scene_units_per_metre=1.0,
                    template_score=0.8,
                ),
                CandidateScaleDiagnostics(
                    candidate_id="candidate-b",
                    nht_scene_units_per_metre=1.0,
                    template_score=0.7,
                ),
            ),
            common_nht_scene_units_per_metre=1.0,
            maximum_relative_scale_deviation=0.0,
        ),
    )
