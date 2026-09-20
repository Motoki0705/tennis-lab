"""Confidence sampling, degenerate Top-4, outliers and failure contracts."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from src.tasks.court_detection.geometry.confidence_homography import (
    estimate_confidence_homography,
)


def points() -> tuple[np.ndarray, np.ndarray]:
    src = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [0.5, 1],
            [2.5, 1],
        ],
        dtype=float,
    )
    matrix = np.array([[80, 15, 30], [5, 60, 20], [0.04, 0.02, 1.0]])
    dst = cv2.perspectiveTransform(src[None], matrix)[0]
    return src, dst


def test_collinear_top_four_expands_pool_and_refits_noisy_inliers() -> None:
    src, truth = points()
    observed = truth + np.random.default_rng(8).normal(0, 0.15, truth.shape)
    observed[-2:] += [70, -50]
    score = np.linspace(0.99, 0.6, len(src))
    result = estimate_confidence_homography(
        src, observed, score, reprojection_threshold_px=1
    )
    assert result.status == "ok"
    assert result.ranked_indices[:4].tolist() == [0, 1, 2, 3]
    assert result.fit_inliers.sum() == 8
    assert not result.inliers[-2:].any()
    assert result.inliers[:8].all()
    assert np.max(np.linalg.norm(result.projected - truth, axis=1)) < 0.5
    # Explicit all-inlier least-squares fit, rather than a four-point template.
    expected, _ = cv2.findHomography(src[:8], observed[:8], method=0)
    np.testing.assert_allclose(result.matrix, expected, atol=1e-10)
    again = estimate_confidence_homography(
        src, observed, score, reprojection_threshold_px=1
    )
    np.testing.assert_array_equal(result.matrix, again.matrix)


def test_scores_reorder_actual_solver_inputs_and_masks_return_semantic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.tasks.court_detection.geometry.confidence_homography as module

    src, dst = points()
    score = np.array([0.6, 0.8, 0.8, 0.9, 0.7, 0.95, 0.5, 0.4, 0.3, 0.2])
    calls: list[tuple[Any, ...]] = []
    original = module.cv2.findHomography

    def record(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.cv2, "findHomography", record)
    result = estimate_confidence_homography(
        src, dst, score, reprojection_threshold_px=1
    )
    order = [5, 3, 1, 2, 4, 0, 6, 7, 8, 9]
    assert result.ranked_indices.tolist() == order
    np.testing.assert_array_equal(calls[0][0], src[order])
    assert calls[0][2].sampler == cv2.SAMPLING_PROSAC
    assert result.inliers.all()
    np.testing.assert_allclose(result.projected, dst, atol=1e-4)


@pytest.mark.parametrize(
    "kind", ["collinear", "duplicate", "three_collinear", "too_few"]
)
def test_degenerate_or_insufficient_input_has_no_fallback(kind: str) -> None:
    src, dst = points()
    if kind == "collinear":
        src, dst = src[:4], dst[:4]
    elif kind == "duplicate":
        src, dst = np.repeat(src[:1], 4, axis=0), np.repeat(dst[:1], 4, axis=0)
    elif kind == "three_collinear":
        src, dst = src[[0, 1, 2, 4]], dst[[0, 1, 2, 4]]
    else:
        src, dst = src[:3], dst[:3]
    result = estimate_confidence_homography(
        src, dst, np.ones(len(src)), reprojection_threshold_px=1
    )
    assert result.matrix is None
    assert result.status != "ok"
    assert not result.inliers.any()
    assert np.isnan(result.projected).all()


def test_missing_and_low_score_points_do_not_enter_solver() -> None:
    src, dst = points()
    dst[0] = np.nan
    scores = np.ones(len(src))
    scores[1] = 0.01
    result = estimate_confidence_homography(
        src, dst, scores, reprojection_threshold_px=1
    )
    assert result.status == "ok"
    assert result.ranked_indices.tolist() == list(range(2, 10))
    assert not result.inliers[:2].any()
    assert np.isnan(result.residuals_px[:2]).all()


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_invalid_score_rejected(bad: float) -> None:
    src, dst = points()
    scores = np.ones(len(src))
    scores[0] = bad
    with pytest.raises(ValueError, match="scores"):
        estimate_confidence_homography(src, dst, scores, reprojection_threshold_px=1)


def test_threshold_scales_with_image_units() -> None:
    src, dst = points()
    dst[-1] += [100, 50]
    score = np.linspace(0.99, 0.6, len(src))
    one = estimate_confidence_homography(src, dst, score, reprojection_threshold_px=2)
    two = estimate_confidence_homography(
        src, dst * 2, score, reprojection_threshold_px=4
    )
    assert one.status == two.status == "ok"
    np.testing.assert_array_equal(one.inliers, two.inliers)
    np.testing.assert_allclose(one.projected * 2, two.projected, atol=1e-3)


def test_highest_confidence_wrong_point_can_be_rejected() -> None:
    src, truth = points()
    dst = truth.copy()
    dst[0] += [70, -50]
    result = estimate_confidence_homography(
        src,
        dst,
        np.linspace(0.99, 0.6, len(src)),
        reprojection_threshold_px=1,
    )
    assert result.status == "ok"
    assert not result.inliers[0]
    assert result.inliers[1:].all()
    np.testing.assert_allclose(result.projected, truth, atol=1e-4)


@pytest.mark.parametrize("crosses_pole", [False, True])
def test_exact_four_points_and_projection_pole(crosses_pole: bool) -> None:
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    matrix = np.array([[100, 0, 10], [0, 100, 20], [0, 0, 1.0]])
    if crosses_pole:
        matrix[2] = [1, 0, -0.5]
    dst = cv2.perspectiveTransform(src[None], matrix)[0]
    result = estimate_confidence_homography(
        src,
        dst,
        np.ones(4),
        reprojection_threshold_px=1,
    )
    if crosses_pole:
        assert result.matrix is None
        assert not result.inliers.any()
    else:
        assert result.status == "ok"
        assert result.inliers.sum() == 4
        np.testing.assert_allclose(result.projected, dst, atol=1e-5)
