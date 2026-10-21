"""Confidence sampling, degenerate Top-4, outliers and failure contracts."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from src.tasks.court_detection.geometry.confidence_homography import (
    PROSAC_OPTIONS,
    ProsacOrientation,
    estimate_confidence_homography,
)
from src.tasks.court_detection.geometry.homography import court_template_xy


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
    assert len(calls) == 3
    np.testing.assert_array_equal(calls[0][0], src[order])
    np.testing.assert_array_equal(calls[1][0], src[order] * [-1, 1])
    for call in calls[:2]:
        np.testing.assert_array_equal(call[1], dst[order])
        for name, value in PROSAC_OPTIONS.items():
            assert getattr(call[2], name) == value
        assert call[2].threshold == 1
    np.testing.assert_array_equal(calls[2][0], src)
    np.testing.assert_array_equal(calls[2][1], dst)
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
    assert result.prosac_orientation is None
    assert result.status == (
        "insufficient_points" if kind == "too_few" else "degenerate_correspondences"
    )
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
@pytest.mark.parametrize("x_sign", [-1, 1])
def test_exact_four_points_and_projection_pole(crosses_pole: bool, x_sign: int) -> None:
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    matrix = np.array([[100 * x_sign, 0, 10], [0, 100, 20], [0, 0, 1.0]])
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
        assert result.prosac_orientation is None
        assert not result.inliers.any()
    else:
        assert result.status == "ok"
        assert result.prosac_orientation is (
            ProsacOrientation.IDENTITY if x_sign == 1 else ProsacOrientation.X_REFLECTED
        )
        assert result.inliers.sum() == 4
        np.testing.assert_allclose(result.projected, dst, atol=1e-5)


@pytest.mark.parametrize("y_sign", [-1, 1], ids=["camera_view", "preserving"])
@pytest.mark.parametrize("with_outliers", [False, True])
def test_court_template_supports_both_orientations_and_equivalent_source_frames(
    y_sign: int, with_outliers: bool
) -> None:
    template = court_template_xy().astype(np.float64)
    matrix = np.array([[30, 3, 640], [2, y_sign * 20, 400], [0.005, 0.01, 1]])
    truth = cv2.perspectiveTransform(template[None], matrix)[0]
    observed = truth.copy()
    scores = np.linspace(0.99, 0.6, len(template))
    scores[1:] = scores[1:][::-1]
    if with_outliers:
        observed += np.random.default_rng(4).normal(0, 0.1, observed.shape)
        # Includes the highest-confidence point and noncontiguous semantic IDs.
        observed[[0, 7, 10]] += [90, -70]
    result = estimate_confidence_homography(
        template, observed, scores, reprojection_threshold_px=1
    )
    reflection = np.diag([-1.0, 1.0, 1.0])
    reflected_template = template * [-1, 1]
    reflected = estimate_confidence_homography(
        reflected_template, observed, scores, reprojection_threshold_px=1
    )
    assert result.status == reflected.status == "ok"
    assert result.prosac_orientation is (
        ProsacOrientation.IDENTITY if y_sign == 1 else ProsacOrientation.X_REFLECTED
    )
    assert reflected.prosac_orientation is (
        ProsacOrientation.X_REFLECTED if y_sign == 1 else ProsacOrientation.IDENTITY
    )
    assert result.matrix is not None and reflected.matrix is not None
    # H_reflected @ F == H, including the projective denominator, so changing
    # source handedness cannot change pixel errors, masks or downstream points.
    np.testing.assert_allclose(reflected.matrix @ reflection, result.matrix, atol=1e-7)
    np.testing.assert_allclose(reflected.projected, result.projected, atol=1e-6)
    np.testing.assert_allclose(reflected.residuals_px, result.residuals_px, atol=1e-6)
    np.testing.assert_array_equal(reflected.ranked_indices, result.ranked_indices)
    np.testing.assert_array_equal(reflected.fit_inliers, result.fit_inliers)
    np.testing.assert_array_equal(reflected.inliers, result.inliers)
    assert np.max(np.linalg.norm(result.projected - truth, axis=1)) < 0.5
    expected_mask: np.ndarray = np.ones(len(template), dtype=bool)
    if with_outliers:
        expected_mask[[0, 7, 10]] = False
    np.testing.assert_array_equal(result.fit_inliers, expected_mask)
    expected, _ = cv2.findHomography(
        template[expected_mask], observed[expected_mask], method=0
    )
    np.testing.assert_allclose(result.matrix, expected, atol=1e-10)


def test_both_solver_orientations_failing_preserves_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src, dst = points()
    calls = []

    def fail(*args: Any, **kwargs: Any) -> tuple[None, None]:
        calls.append(args)
        return None, None

    monkeypatch.setattr(cv2, "findHomography", fail)
    result = estimate_confidence_homography(
        src, dst, np.ones(len(src)), reprojection_threshold_px=1
    )
    assert len(calls) == 2
    assert result.status == "prosac_failed"
    assert result.prosac_orientation is None
    assert result.matrix is None
    assert not result.fit_inliers.any()
    assert not result.inliers.any()
    assert np.isnan(result.projected).all()
    assert np.isnan(result.residuals_px).all()


@pytest.mark.parametrize(
    "preferred",
    [ProsacOrientation.IDENTITY, ProsacOrientation.X_REFLECTED, None],
    ids=["identity_wins", "x_reflected_wins", "tie_keeps_identity"],
)
def test_orientation_candidates_compete_on_pixel_cost_before_original_frame_refit(
    monkeypatch: pytest.MonkeyPatch,
    preferred: ProsacOrientation | None,
) -> None:
    src, dst = points()
    correct, _ = cv2.findHomography(src, dst, method=0)
    assert correct is not None
    identity_candidate = correct.copy()
    reflected_candidate = correct @ np.diag([-1.0, 1.0, 1.0])
    if preferred is ProsacOrientation.X_REFLECTED:
        identity_candidate[0, 2] += 40
    elif preferred is ProsacOrientation.IDENTITY:
        reflected_candidate[0, 2] += 40
    mask: np.ndarray = np.ones((len(src), 1), dtype=np.uint8)
    mask[-1] = 0
    expected_orientation = preferred or ProsacOrientation.IDENTITY
    expected_mask = (
        mask.astype(bool)
        if preferred is ProsacOrientation.X_REFLECTED
        else np.ones_like(mask, dtype=bool)
    )
    original = cv2.findHomography
    expected, _ = original(src[expected_mask[:, 0]], dst[expected_mask[:, 0]], method=0)
    calls = 0

    def solve(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return identity_candidate, np.ones_like(mask)
        if calls == 2:
            return reflected_candidate, mask
        np.testing.assert_array_equal(args[0], src[expected_mask[:, 0]])
        np.testing.assert_array_equal(args[1], dst[expected_mask[:, 0]])
        assert kwargs == {"method": 0}
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "findHomography", solve)
    result = estimate_confidence_homography(
        src, dst, np.ones(len(src)), reprojection_threshold_px=1
    )
    assert calls == 3
    assert result.status == "ok"
    assert result.prosac_orientation is expected_orientation
    np.testing.assert_array_equal(result.matrix, expected)
    np.testing.assert_array_equal(result.fit_inliers, expected_mask[:, 0])
    assert result.inliers.all()
    np.testing.assert_allclose(result.projected, dst, atol=1e-4)


@pytest.mark.parametrize("failed_orientation", list(ProsacOrientation))
def test_one_solver_orientation_failing_records_the_successful_orientation(
    monkeypatch: pytest.MonkeyPatch,
    failed_orientation: ProsacOrientation,
) -> None:
    src, dst = points()
    original = cv2.findHomography
    expected, _ = original(src, dst, method=0)
    assert expected is not None
    calls = 0

    def solve(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if kwargs == {"method": 0}:
            np.testing.assert_array_equal(args[0], src)
            np.testing.assert_array_equal(args[1], dst)
            return original(*args, **kwargs)
        orientation = (
            ProsacOrientation.IDENTITY if calls == 1 else ProsacOrientation.X_REFLECTED
        )
        if orientation is failed_orientation:
            return None, None
        frame = np.eye(3) if calls == 1 else np.diag([-1.0, 1.0, 1.0])
        return expected @ frame, np.ones((len(src), 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "findHomography", solve)
    result = estimate_confidence_homography(
        src, dst, np.ones(len(src)), reprojection_threshold_px=1
    )
    assert calls == 3
    assert result.status == "ok"
    assert result.prosac_orientation is (
        ProsacOrientation.X_REFLECTED
        if failed_orientation is ProsacOrientation.IDENTITY
        else ProsacOrientation.IDENTITY
    )
    np.testing.assert_array_equal(result.matrix, expected)
    assert result.fit_inliers.all() and result.inliers.all()


def test_failure_after_orientation_selection_clears_the_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src, dst = points()
    correct, _ = cv2.findHomography(src, dst, method=0)
    assert correct is not None
    calls = 0

    def solve(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if kwargs == {"method": 0}:
            return None, None
        frame = np.eye(3) if calls == 1 else np.diag([-1.0, 1.0, 1.0])
        return correct @ frame, np.ones((len(src), 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "findHomography", solve)
    result = estimate_confidence_homography(
        src, dst, np.ones(len(src)), reprojection_threshold_px=1
    )
    assert calls == 3
    assert result.status == "refit_failed"
    assert result.matrix is None
    assert result.prosac_orientation is None
    assert not result.inliers.any()
