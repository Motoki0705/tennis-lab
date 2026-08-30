"""Fit rigid court transforms using fit evidence, then gate on holdout evidence."""

from __future__ import annotations

import json
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
    validate_alignment_trace_final_binding,
)
from src.synthetic_data_generation.alignment.evaluation import evaluate_partition
from src.synthetic_data_generation.alignment.whole_court import (
    evaluate_court_identifiability,
    evaluate_court_topology,
    evaluate_whole_template,
)
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
    validate_alignment_trace_final_binding(
        evidence.alignment_trace,
        ground_plane_frame=evidence.ground_plane_frame,
        candidates=candidate_tuple,
    )
    validate_whole_court_evidence(
        evidence,
        candidates=candidate_tuple,
        policy=policy,
    )
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


def whole_court_diagnostics(
    evidence: AlignmentEvidence,
    *,
    candidates: Sequence[CandidateAlignment],
    policy: AlignmentAcceptancePolicy,
) -> dict[str, object]:
    """Recompute identifiability, scale, template, and topology diagnostics."""
    settings = evidence.whole_court_settings
    candidate_tuple = tuple(candidates)
    fit_points = _measured_partition_points(
        evidence,
        camera_ids=evidence.partitions.fit_camera_ids,
    )
    holdout_points = _measured_partition_points(
        evidence,
        camera_ids=evidence.partitions.holdout_camera_ids,
    )
    evidence_by_candidate = {
        candidate.candidate_id: candidate for candidate in evidence.candidates
    }
    refit_by_candidate = {
        diagnostic.candidate_id: diagnostic
        for diagnostic in evidence.diagnostics.candidate_scales
    }
    candidate_payloads: list[dict[str, object]] = []
    for candidate in candidate_tuple:
        candidate_evidence = evidence_by_candidate.get(candidate.candidate_id)
        if candidate_evidence is None:
            raise ValueError(
                f"Missing source evidence for candidate {candidate.candidate_id!r}."
            )
        refit_diagnostic = refit_by_candidate.get(candidate.candidate_id)
        if refit_diagnostic is None:
            raise ValueError(
                f"Missing refit diagnostics for candidate {candidate.candidate_id!r}."
            )
        fit_template = evaluate_whole_template(
            scene_from_court=candidate.scene_from_court,
            measured_points_scene=fit_points,
            settings=settings,
        )
        holdout_template = evaluate_whole_template(
            scene_from_court=candidate.scene_from_court,
            measured_points_scene=holdout_points,
            settings=settings,
        )
        candidate_payloads.append(
            {
                "candidate_id": candidate.candidate_id,
                "court_instance_id": candidate.court_instance_id,
                "selected_correspondence_accepted": candidate.accepted,
                "common_scale_refit": {
                    "center_displacement_metres": (
                        refit_diagnostic.common_scale_refit_center_displacement_metres
                    ),
                    "maximum_center_displacement_metres": (
                        refit_diagnostic.maximum_common_scale_refit_center_displacement_metres
                    ),
                    "accepted": bool(
                        refit_diagnostic.common_scale_refit_center_displacement_metres
                        <= settings.maximum_center_refit_displacement_metres
                        and abs(
                            refit_diagnostic.maximum_common_scale_refit_center_displacement_metres
                            - settings.maximum_center_refit_displacement_metres
                        )
                        <= 1.0e-10
                    ),
                },
                "fit": {
                    "identifiability": evaluate_court_identifiability(
                        candidate_evidence.fit,
                        minimum_camera_count=policy.fit.minimum_camera_count,
                        settings=settings,
                    ).to_dict(
                        minimum_camera_count=policy.fit.minimum_camera_count,
                        settings=settings,
                    ),
                    "whole_template_diagnostic": fit_template.to_dict(
                        settings=settings
                    ),
                },
                "holdout": {
                    "identifiability": evaluate_court_identifiability(
                        candidate_evidence.holdout,
                        minimum_camera_count=policy.holdout.minimum_camera_count,
                        settings=settings,
                    ).to_dict(
                        minimum_camera_count=policy.holdout.minimum_camera_count,
                        settings=settings,
                    ),
                    "whole_template_diagnostic": holdout_template.to_dict(
                        settings=settings
                    ),
                },
            }
        )
    accepted = tuple(candidate for candidate in candidate_tuple if candidate.accepted)
    topology = evaluate_court_topology(
        tuple(
            (candidate.candidate_id, candidate.scene_from_court)
            for candidate in accepted
        )
    )
    return {
        "policy": settings.to_dict(),
        "common_scale": {
            "maximum_relative_deviation": (
                evidence.diagnostics.maximum_relative_scale_deviation
            ),
            "maximum_allowed_relative_deviation": (
                settings.maximum_common_scale_relative_deviation
            ),
            "accepted": bool(
                evidence.diagnostics.maximum_relative_scale_deviation
                <= settings.maximum_common_scale_relative_deviation
            ),
        },
        "required_court_count_check": (
            len(accepted) == settings.required_court_count
            and len(candidate_tuple) == settings.required_court_count
        ),
        "candidates": candidate_payloads,
        "topology": [item.to_dict(settings=settings) for item in topology],
    }


def validate_whole_court_evidence(
    evidence: AlignmentEvidence,
    *,
    candidates: Sequence[CandidateAlignment],
    policy: AlignmentAcceptancePolicy,
) -> None:
    """Fail closed on positive identifiability, scale, and topology gates."""
    diagnostics = whole_court_diagnostics(
        evidence,
        candidates=candidates,
        policy=policy,
    )
    candidate_payloads = diagnostics["candidates"]
    topology_payloads = diagnostics["topology"]
    if not isinstance(candidate_payloads, list) or not isinstance(
        topology_payloads, list
    ):
        raise RuntimeError("Whole-court diagnostics have an invalid internal shape.")
    candidates_accepted = all(
        isinstance(payload, dict)
        and payload.get("selected_correspondence_accepted") is True
        and isinstance(payload.get("common_scale_refit"), dict)
        and payload["common_scale_refit"].get("accepted") is True
        and isinstance(payload.get("fit"), dict)
        and isinstance(payload.get("holdout"), dict)
        and isinstance(payload["fit"].get("identifiability"), dict)
        and isinstance(payload["holdout"].get("identifiability"), dict)
        and payload["fit"]["identifiability"].get("accepted") is True
        and payload["holdout"]["identifiability"].get("accepted") is True
        for payload in candidate_payloads
    )
    topology_accepted = all(
        isinstance(payload, dict) and payload.get("accepted") is True
        for payload in topology_payloads
    )
    common_scale = diagnostics.get("common_scale")
    common_scale_accepted = (
        isinstance(common_scale, dict) and common_scale.get("accepted") is True
    )
    if (
        diagnostics["required_court_count_check"] is not True
        or not candidates_accepted
        or not topology_accepted
        or not common_scale_accepted
    ):
        raise ValueError(
            "Whole-court alignment evidence failed acceptance: "
            + json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
        )


def _measured_partition_points(
    evidence: AlignmentEvidence,
    *,
    camera_ids: tuple[str, ...],
) -> NDArray[np.float64]:
    selected = [
        evidence.metric_adapter.metric_from_nht_points(item.points_nht_scene)
        for item in evidence.measured_camera_lines
        if item.camera_id in set(camera_ids)
    ]
    if len(selected) != len(camera_ids):
        raise ValueError(
            "Measured line evidence is missing a declared camera partition."
        )
    return np.concatenate(selected)


def apply_transform(
    transform: RigidTransform,
    points: Sequence[Sequence[float]] | NDArray[np.floating[Any]],
) -> NDArray[np.float64]:
    """Apply a shared rigid transform through its validated public contract."""
    return transform.apply(np.asarray(points, dtype=np.float64))


__all__ = [
    "apply_transform",
    "fit_alignment",
    "fit_rigid_transform",
    "validate_whole_court_evidence",
    "whole_court_diagnostics",
]
