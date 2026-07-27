"""Tests for evidence-guided multi-start court fitting."""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.evidence_guided_court_fit import (
    CourtFitCluster,
    CourtFitRun,
    CourtMultiStartFitSettings,
    candidate_from_parameters,
    cluster_fit_runs,
    fit_unknown_number_of_courts,
    score_court_hypothesis,
    select_reliable_cluster,
    suppress_explained_evidence,
)
from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)


def _plane() -> GroundPlaneEstimate:
    return GroundPlaneEstimate(
        normal=(0.0, 0.0, 1.0),
        offset=0.0,
        origin=(0.0, 0.0, 0.0),
        basis_u=(1.0, 0.0, 0.0),
        basis_v=(0.0, 1.0, 0.0),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
        metrics={},
    )


def _draw_court(
    image: NDArray[np.float32],
    *,
    bounds: tuple[float, float, float, float],
    spacing: float,
    center: tuple[float, float],
    orientation: float,
    scale: float,
) -> None:
    segments = (
        ((-HALF_DOUBLES_WIDTH, -HALF_LENGTH), (-HALF_DOUBLES_WIDTH, HALF_LENGTH)),
        ((HALF_DOUBLES_WIDTH, -HALF_LENGTH), (HALF_DOUBLES_WIDTH, HALF_LENGTH)),
        ((-HALF_SINGLES_WIDTH, -HALF_LENGTH), (-HALF_SINGLES_WIDTH, HALF_LENGTH)),
        ((HALF_SINGLES_WIDTH, -HALF_LENGTH), (HALF_SINGLES_WIDTH, HALF_LENGTH)),
        ((-HALF_DOUBLES_WIDTH, -HALF_LENGTH), (HALF_DOUBLES_WIDTH, -HALF_LENGTH)),
        ((-HALF_DOUBLES_WIDTH, HALF_LENGTH), (HALF_DOUBLES_WIDTH, HALF_LENGTH)),
        (
            (-HALF_SINGLES_WIDTH, -SERVICE_LINE_DISTANCE),
            (HALF_SINGLES_WIDTH, -SERVICE_LINE_DISTANCE),
        ),
        (
            (-HALF_SINGLES_WIDTH, SERVICE_LINE_DISTANCE),
            (HALF_SINGLES_WIDTH, SERVICE_LINE_DISTANCE),
        ),
        ((0.0, -SERVICE_LINE_DISTANCE), (0.0, SERVICE_LINE_DISTANCE)),
    )
    cosine = math.cos(orientation)
    sine = math.sin(orientation)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    for start, end in segments:
        uv = np.asarray(
            (start, end), dtype=np.float64
        ) @ rotation.T * scale + np.asarray(center)
        pixels = np.rint(
            np.column_stack(
                (
                    (uv[:, 0] - bounds[0]) / spacing,
                    (uv[:, 1] - bounds[2]) / spacing,
                )
            )
        ).astype(np.int32)
        cv2.line(image, pixels[0], pixels[1], 5.0, 2, cv2.LINE_AA)


def _evidence(
    centers: tuple[tuple[float, float], ...],
) -> tuple[NDArray[np.float32], tuple[float, float, float, float], float]:
    bounds = (-1.2, 1.2, -1.1, 1.2)
    spacing = 0.008
    width = int(np.ceil((bounds[1] - bounds[0]) / spacing)) + 1
    height = int(np.ceil((bounds[3] - bounds[2]) / spacing)) + 1
    evidence: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
    for center in centers:
        _draw_court(
            evidence,
            bounds=bounds,
            spacing=spacing,
            center=center,
            orientation=-1.50,
            scale=0.070,
        )
    return evidence, bounds, spacing


def test_settings_reject_invalid_proposal_mixture() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        CourtMultiStartFitSettings(
            proposal_fraction=0.7,
            uniform_fraction=0.2,
            orthogonal_fraction=0.2,
        )


def test_clustering_treats_180_degree_solutions_as_one_court() -> None:
    runs = (
        CourtFitRun(
            run_id="run-0",
            seed=0,
            evidence_subset_id="full",
            initialisation_kind="mixed",
            parameters=(0.1, -0.2, 0.02, 0.07),
            component_scores={"template_score": 1.0},
            optimizer_evaluations=10,
        ),
        CourtFitRun(
            run_id="run-1",
            seed=1,
            evidence_subset_id="blocks",
            initialisation_kind="mixed",
            parameters=(0.101, -0.201, -math.pi + 0.02, 0.0701),
            component_scores={"template_score": 0.9},
            optimizer_evaluations=10,
        ),
        CourtFitRun(
            run_id="run-2",
            seed=2,
            evidence_subset_id="full",
            initialisation_kind="mixed",
            parameters=(0.8, 0.8, math.pi / 2.0, 0.06),
            component_scores={"template_score": 0.4},
            optimizer_evaluations=10,
        ),
    )

    clusters = cluster_fit_runs(runs, CourtMultiStartFitSettings())

    assert len(clusters) == 2
    assert clusters[0].support_rate == pytest.approx(2.0 / 3.0)
    assert set(clusters[0].member_run_ids) == {"run-0", "run-1"}


def test_scoring_and_soft_suppression_favour_the_complete_orientation() -> None:
    parameters = (-0.08, 0.67, -1.50, 0.070)
    evidence, bounds, spacing = _evidence(((parameters[0], parameters[1]),))
    settings = CourtMultiStartFitSettings(
        num_global_runs=2,
        bootstrap_runs=1,
        max_instances=1,
    )

    correct = score_court_hypothesis(
        evidence,
        parameters=parameters,
        bounds=bounds,
        grid_spacing=spacing,
        settings=settings,
    )
    orthogonal = score_court_hypothesis(
        evidence,
        parameters=(
            parameters[0],
            parameters[1],
            parameters[2] + math.pi / 2.0,
            parameters[3],
        ),
        bounds=bounds,
        grid_spacing=spacing,
        settings=settings,
    )
    residual = suppress_explained_evidence(
        evidence,
        parameters=parameters,
        bounds=bounds,
        grid_spacing=spacing,
        settings=replace(settings, residual_suppression_strength=0.8),
    )

    assert correct["template_score"] > orthogonal["template_score"]
    assert correct["internal_line_coverage"] > orthogonal["internal_line_coverage"]
    assert 0.0 < float(np.sum(residual)) < float(np.sum(evidence))
    assert bool(np.any(residual[evidence > 0.0] > 0.0))


def test_selection_rejects_close_90_degree_competitors() -> None:
    settings = CourtMultiStartFitSettings(
        num_global_runs=2,
        bootstrap_runs=1,
        max_instances=1,
    )
    candidates = [
        candidate_from_parameters(
            (0.0, 0.0, orientation, 0.07),
            candidate_id=f"candidate-{index}",
            template_score=1.0,
            optimizer_evaluations=10,
            plane=_plane(),
        )
        for index, orientation in enumerate((0.0, math.pi / 2.0))
    ]
    clusters = tuple(
        CourtFitCluster(
            cluster_id=f"cluster-{index}",
            representative_parameters=(0.0, 0.0, orientation, 0.07),
            member_run_ids=(f"run-{index}",),
            support_rate=0.5,
            bootstrap_survival_rate=1.0,
            component_scores={
                "template_score": 1.0,
                "line_coverage": 0.5,
                "internal_line_coverage": 0.45,
                "explained_evidence": 0.2,
                "background_contrast": 0.5,
                "background_contrast_ratio": 0.5,
            },
            confidence=confidence,
            residual_gain=0.2,
            candidate=candidates[index],
        )
        for index, (orientation, confidence) in enumerate(
            ((0.0, 0.80), (math.pi / 2.0, 0.75))
        )
    )

    selected, evaluated = select_reliable_cluster(
        clusters,
        existing=(),
        settings=settings,
    )

    assert selected is None
    assert all(
        "ambiguous_orientation" in cluster.rejection_reasons for cluster in evaluated
    )


@pytest.mark.slow
def test_unknown_count_fit_recovers_two_synthetic_courts() -> None:
    expected_centers = ((-0.08, 0.67), (-0.02, -0.34))
    evidence, bounds, spacing = _evidence(expected_centers)
    settings = CourtMultiStartFitSettings(
        seed=19,
        num_global_runs=6,
        bootstrap_runs=2,
        min_cluster_support_rate=0.25,
        min_bootstrap_survival_rate=0.5,
        residual_suppression_strength=0.9,
        min_residual_gain=0.01,
        max_instances=3,
        optimizer_max_iterations=45,
        optimizer_population_size=8,
        optimizer_tolerance=1.0e-6,
        local_optimizer_max_iterations=35,
        local_optimizer_population_size=8,
        min_confidence=0.45,
        min_line_coverage=0.30,
        min_internal_line_coverage=0.30,
    )

    result = fit_unknown_number_of_courts(
        evidence,
        bounds=bounds,
        grid_spacing=spacing,
        plane=_plane(),
        settings=settings,
    )

    assert len(result.accepted_candidates) == 2
    fitted_centers = np.asarray(
        [candidate.center_uv for candidate in result.accepted_candidates]
    )
    for expected_center in expected_centers:
        distances = np.linalg.norm(fitted_centers - expected_center, axis=1)
        assert float(np.min(distances)) < 0.04
    assert result.stop_status in {
        "stopped_no_reliable_cluster",
        "stopped_low_residual_gain",
    }
    assert len(result.runs_by_iteration) == 3
