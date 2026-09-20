"""Known-geometry LINE refinement, hard KP exclusion and failure contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import cv2
import numpy as np
import pytest

from src.tasks.court_detection.geometry.hybrid_homography import (
    HybridHomographyConfig,
    estimate_hybrid_homography,
    project_candidate,
    select_keypoints,
    selected_kp_residuals,
)
from src.tasks.court_detection.geometry.line_evidence import (
    LineEvidence,
    clip_segments,
    line_residuals,
)

EDGES = np.array(
    [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
)
CONFIG = HybridHomographyConfig(refine_candidates=3, max_nfev=100)


def scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    template = np.array(
        [
            [0, 0],
            [10, 0],
            [0, 24],
            [10, 24],
            [1.25, 0],
            [1.25, 24],
            [8.75, 0],
            [8.75, 24],
            [1.25, 6.4],
            [8.75, 6.4],
            [1.25, 17.6],
            [8.75, 17.6],
            [5, 6.4],
            [5, 17.6],
        ],
        dtype=float,
    )
    matrix = np.array([[20, 2, 30], [1, 9, 25], [0.005, 0.0005, 1]])
    truth = cv2.perspectiveTransform(template[None], matrix)[0]
    probability: np.ndarray = np.zeros((320, 320), dtype=np.float32)
    for a, b in EDGES:
        cv2.line(
            probability,
            tuple(np.rint(truth[a]).astype(int)),
            tuple(np.rint(truth[b]).astype(int)),
            0.95,
            3,
        )
    return template, truth, probability


def test_line_corrects_biased_kp_and_excludes_confident_outliers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.tasks.court_detection.geometry.hybrid_homography as module

    template, truth, probability = scene()
    observed = truth + [1.7, 0.9]
    bad = [0, 3, 13]
    observed[bad] += [60, -50]
    scores = np.full(14, 0.8)
    scores[bad] = 0.99
    selections = []
    original = module.selected_kp_residuals

    def capture(*args: Any, **kwargs: Any) -> Any:
        selections.append(args[3].copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "selected_kp_residuals", capture)
    result = estimate_hybrid_homography(
        template,
        observed,
        scores,
        probability,
        edges=EDGES,
        image_size_hw=(320, 320),
        config=CONFIG,
    )
    assert result.status == "ok"
    assert 4 <= result.selected.sum() <= 8
    assert not result.selected[bad].any()
    assert all(4 <= mask.sum() <= 8 for mask in selections)
    assert result.best is not None
    assert tuple(np.flatnonzero(result.selected)) == result.best.selection_history[-1]
    before = np.mean(np.linalg.norm(result.kp_only.projected - truth, axis=1))
    after = np.mean(np.linalg.norm(result.projected - truth, axis=1))
    assert after < 0.8
    assert after < 0.5 * before
    assert max(
        np.linalg.norm(
            result.projected[result.selected] - observed[result.selected], axis=1
        )
    ) <= CONFIG.threshold_diagonal_ratio * np.hypot(320, 320)


def test_rejected_kp_have_exactly_zero_effect_on_objective() -> None:
    _, truth, _ = scene()
    observed = truth + 0.3
    selected: np.ndarray = np.zeros(14, dtype=bool)
    selected[[0, 1, 2, 3, 8, 9]] = True
    scores = np.full(14, 0.8)
    expected = selected_kp_residuals(truth, observed, scores, selected, 3)
    observed[~selected] = np.nan
    scores[~selected] = 1e100
    actual = selected_kp_residuals(truth, observed, scores, selected, 3)
    np.testing.assert_array_equal(expected, actual)
    assert len(actual) == 2 * selected.sum()


def test_nonconvergent_joint_fit_has_no_kp_only_fallback() -> None:
    template, truth, probability = scene()
    result = estimate_hybrid_homography(
        template,
        truth + [1.7, 0.9],
        np.full(14, 0.8),
        probability,
        edges=EDGES,
        image_size_hw=(320, 320),
        config=replace(CONFIG, max_nfev=1),
    )
    assert result.status == "joint_optimization_failed"
    assert result.matrix is None
    assert result.kp_only.matrix is not None
    assert result.line_selection is not None
    assert not result.selected.any()


def test_a_projection_pole_inside_court_is_rejected() -> None:
    template, _, _ = scene()
    matrix = np.array([[20, 2, 30], [1, 9, 25], [1, 0, -5.1]])
    assert project_candidate(matrix, template) is None


def test_gate_excludes_geometric_and_line_outliers_and_never_uses_all_points() -> None:
    template, truth, _ = scene()
    scores = np.linspace(0.99, 0.5, 14)
    observed = truth.copy()
    observed[0] += 50  # highest score does not override geometry
    distances = np.zeros(14)
    distances[1] = 30  # a plausible KP unsupported by LINE is also excluded
    selected = select_keypoints(template, observed, scores, truth, distances, 3, CONFIG)
    assert selected.sum() == 8
    assert not selected[:2].any()
    all_good = select_keypoints(template, truth, scores, truth, np.zeros(14), 3, CONFIG)
    assert all_good.sum() == 8
    small_input = select_keypoints(
        template[:8], truth[:8], scores[:8], truth[:8], np.zeros(8), 3, CONFIG
    )
    assert small_input.sum() == 7  # Never use the complete input, even below KP14.


@pytest.mark.parametrize("cap", [3, 9, 14, 100])
def test_config_cannot_override_the_eight_point_limit(cap: int) -> None:
    with pytest.raises(ValueError, match="cap between 4 and 8"):
        replace(CONFIG, max_kp=cap)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_absent_or_diffuse_line_is_explicit_failure(value: float) -> None:
    template, truth, probability = scene()
    result = estimate_hybrid_homography(
        template,
        truth,
        np.ones(14),
        np.full_like(probability, value),
        edges=EDGES,
        image_size_hw=(320, 320),
        config=CONFIG,
    )
    assert result.status == "insufficient_or_diffuse_line_evidence"
    assert result.matrix is None
    assert not result.selected.any()
    assert np.isnan(result.projected).all()
    assert result.kp_only.matrix is not None  # deliberately not used as fallback


def test_fewer_than_four_eligible_kp_does_not_invent_correspondences() -> None:
    template, truth, probability = scene()
    scores = np.zeros(14)
    scores[:3] = 1
    result = estimate_hybrid_homography(
        template,
        truth,
        scores,
        probability,
        edges=EDGES,
        image_size_hw=(320, 320),
        config=CONFIG,
    )
    assert result.status == "no_jointly_supported_candidate"
    assert result.matrix is None


def test_bidirectional_line_objective_penalizes_collapsed_court() -> None:
    _, truth, probability = scene()
    evidence = LineEvidence.from_probability(probability, (320, 320))
    real, real_diag = line_residuals(truth, EDGES, evidence, 3)
    collapsed = truth.mean(axis=0) + 0.15 * (truth - truth.mean(axis=0))
    wrong, wrong_diag = line_residuals(collapsed, EDGES, evidence, 3)
    assert wrong @ wrong > real @ real + 5
    assert real_diag["reverse_support"] > 0.95
    assert wrong_diag["reverse_support"] < 0.2


def test_cropped_segments_are_clipped_without_assuming_missing_lines() -> None:
    segments = np.array(
        [
            [[-10, 5], [30, 5]],
            [[5, -10], [5, 30]],
            [[-20, 3], [-10, 3]],
            [[-20, -2], [30, -2]],
        ],
        dtype=float,
    )
    clipped, visible = clip_segments(segments, 20, 10)
    assert visible.tolist() == [True, True, False, False]
    np.testing.assert_allclose(clipped[:2], [[[0, 5], [19, 5]], [[5, 0], [5, 9]]])


@pytest.mark.parametrize("value", [np.nan, -0.1, 1.1])
def test_invalid_line_probabilities_raise(value: float) -> None:
    template, truth, probability = scene()
    probability[0, 0] = value
    with pytest.raises(ValueError, match="probabilities"):
        estimate_hybrid_homography(
            template,
            truth,
            np.ones(14),
            probability,
            edges=EDGES,
            image_size_hw=(320, 320),
            config=CONFIG,
        )
