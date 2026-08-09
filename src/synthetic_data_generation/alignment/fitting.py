"""Fit rigid court transforms using fit evidence, then gate on holdout evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    AlignmentResult,
    CandidateAlignment,
    CorrespondenceSet,
    build_layout,
)
from src.synthetic_data_generation.alignment.evaluation import evaluate_partition
from src.synthetic_data_generation.scene_contract import RigidTransform


def fit_rigid_transform(evidence: CorrespondenceSet) -> RigidTransform:
    """Fit proper SE(3) by Kabsch using only the supplied correspondence set."""
    court = evidence.points_court
    scene = evidence.points_scene
    court_centred = court - np.mean(court, axis=0)
    scene_centred = scene - np.mean(scene, axis=0)
    if np.linalg.matrix_rank(court_centred, tol=1.0e-10) < 2:
        raise ValueError(
            "Fit court correspondences must contain non-collinear geometry."
        )
    if np.linalg.matrix_rank(scene_centred, tol=1.0e-10) < 2:
        raise ValueError(
            "Fit scene correspondences must contain non-collinear geometry."
        )

    covariance = court_centred.T @ scene_centred
    left, _singular_values, right_transposed = np.linalg.svd(covariance)
    rotation = right_transposed.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transposed[-1, :] *= -1.0
        rotation = right_transposed.T @ left.T
    translation = np.mean(scene, axis=0) - rotation @ np.mean(court, axis=0)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return RigidTransform.from_matrix(matrix)


def fit_alignment(
    evidence: AlignmentEvidence,
    *,
    policy: AlignmentAcceptancePolicy,
) -> AlignmentResult:
    """Fit every candidate on fit data and independently evaluate holdout data."""
    candidates: list[CandidateAlignment] = []
    for candidate_evidence in evidence.candidates:
        scene_from_court = fit_rigid_transform(candidate_evidence.fit)
        fit = evaluate_partition(
            candidate_evidence.fit,
            scene_from_court=scene_from_court,
            thresholds=policy.fit,
        )
        holdout = evaluate_partition(
            candidate_evidence.holdout,
            scene_from_court=scene_from_court,
            thresholds=policy.holdout,
        )
        candidates.append(
            CandidateAlignment(
                court_instance_id=candidate_evidence.court_instance_id,
                candidate_id=candidate_evidence.candidate_id,
                scene_from_court=scene_from_court,
                court_from_scene=scene_from_court.inverse(),
                fit=fit,
                holdout=holdout,
            )
        )
    candidate_tuple = tuple(candidates)
    layout = build_layout(
        candidate_tuple,
        complex_points_scene=evidence.complex_points_scene,
        primary_candidate_id=evidence.primary_candidate_id,
    )
    return AlignmentResult(
        partitions=evidence.partitions,
        policy=policy,
        candidates=candidate_tuple,
        layout=layout,
        metric_adapter=evidence.metric_adapter,
    )


def apply_transform(
    transform: RigidTransform,
    points: Sequence[Sequence[float]] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Apply a shared rigid transform through its validated public contract."""
    return transform.apply(np.asarray(points, dtype=np.float64))


__all__ = ["apply_transform", "fit_alignment", "fit_rigid_transform"]
