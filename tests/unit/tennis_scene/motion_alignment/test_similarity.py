"""Tests for the gravity-fixed similarity core used by GVHMR -> PLCS alignment."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.motion_alignment.similarity import (
    FIXED_SCALE,
    HEADING_CIRCULAR_MEAN,
    POSITION_SVD_UMEYAMA,
    SimilarityConfig,
    SimilarityTransform,
    _make_loss,
    fit_similarity,
    wrap_to_pi,
)

Float64Array: TypeAlias = NDArray[np.float64]


def _rz(yaw: float) -> Float64Array:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _track(n_frames: int, seed: int) -> Float64Array:
    """A deterministic, three-dimensional track with vertical and horizontal spread."""

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)
    return np.stack(
        [
            2.0 * np.cos(2.0 * math.pi * t) + 0.3 * t,
            1.5 * np.sin(3.0 * math.pi * t) - 0.2 * t,
            0.7 * np.sin(5.0 * math.pi * t),
        ],
        axis=1,
    ) + rng.normal(0.0, 0.01, (n_frames, 3))


def _apply(
    source: Float64Array,
    scale: float,
    yaw: float,
    translation: Float64Array,
) -> Float64Array:
    return scale * (source @ _rz(yaw).T) + translation


def _heading_pair(
    n_frames: int,
    yaw: float,
    seed: int,
) -> tuple[Float64Array, Float64Array]:
    rng = np.random.default_rng(seed)
    source_heading = rng.uniform(-math.pi, math.pi, n_frames)
    return source_heading, wrap_to_pi(source_heading + yaw)


def test_recovers_exact_transform_and_scales_pairwise_distances() -> None:
    n_frames = 24
    scale = 1.3
    yaw = 2.7
    translation = np.array([5.0, -2.0, 0.75])
    source = _track(n_frames, seed=1)
    target = _apply(source, scale, yaw, translation)
    source_heading, target_heading = _heading_pair(n_frames, yaw, seed=11)

    # The scale prior regularises towards 1; disable it to check exact recovery.
    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(scale_prior=0.0),
    )

    assert result.success
    assert result.transform.scale == pytest.approx(scale, abs=1e-6)
    assert wrap_to_pi(result.transform.yaw - yaw) == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(result.transform.translation, translation, atol=1e-7)
    np.testing.assert_allclose(result.transform.apply(source), target, atol=1e-7)
    assert result.diagnostics.initializer == HEADING_CIRCULAR_MEAN
    assert result.diagnostics.jacobian_rank == 5
    assert result.diagnostics.n_free_parameters == 5

    transformed = result.transform.apply(source)
    source_distances = np.linalg.norm(source[:, None] - source[None, :], axis=-1)
    target_distances = np.linalg.norm(
        transformed[:, None] - transformed[None, :], axis=-1
    )
    np.testing.assert_allclose(target_distances, scale * source_distances, atol=1e-9)


def test_tilted_source_orientation_is_preserved_without_estimating_tilt() -> None:
    n_frames = 20
    scale = 0.8
    yaw = -1.9
    translation = np.array([-1.0, 4.0, 1.5])
    # A track whose local plane is tilted relative to the ground: the fit must
    # carry that 3D shape through unchanged instead of flattening it.
    t = np.linspace(0.0, 1.0, n_frames)
    source = np.stack(
        [
            1.4 * np.cos(3.0 * t),
            1.1 * np.sin(3.0 * t),
            0.9 * np.cos(3.0 * t) + 0.6 * t,
        ],
        axis=1,
    )
    target = _apply(source, scale, yaw, translation)
    source_heading = 0.4 + 0.2 * t
    target_heading = wrap_to_pi(source_heading + yaw)

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(scale_prior=0.0),
    )

    rotation = result.transform.rotation()
    np.testing.assert_allclose(rotation[:2, 2], 0.0, atol=0.0)
    np.testing.assert_allclose(rotation[2, :2], 0.0, atol=0.0)
    assert rotation[2, 2] == 1.0
    np.testing.assert_allclose(rotation, _rz(result.transform.yaw), atol=1e-12)
    # Vertical offsets scale exactly; the fit introduces no tilt component.
    np.testing.assert_allclose(
        result.transform.apply(source)[:, 2],
        scale * source[:, 2] + translation[2],
        atol=1e-9,
    )
    np.testing.assert_allclose(result.transform.apply(source), target, atol=1e-7)


def test_fixed_scale_keeps_scale_and_fits_yaw_and_translation() -> None:
    n_frames = 18
    yaw = 0.9
    translation = np.array([2.0, 2.5, -0.5])
    source = _track(n_frames, seed=3)
    target = _apply(source, 1.0, yaw, translation)
    source_heading, target_heading = _heading_pair(n_frames, yaw, seed=13)

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(fixed_scale=1.0),
    )

    assert result.transform.scale == 1.0
    assert result.diagnostics.free_scale is False
    assert result.diagnostics.fixed_scale == 1.0
    assert result.diagnostics.scale_initializer == FIXED_SCALE
    assert result.diagnostics.n_free_parameters == 4
    assert wrap_to_pi(result.transform.yaw - yaw) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(result.transform.translation, translation, atol=1e-8)


def test_wraps_headings_across_plus_minus_pi() -> None:
    n_frames = 12
    yaw = 0.3
    source = _track(n_frames, seed=5)
    target = _apply(source, 1.1, yaw, np.zeros(3))
    # Source headings sit just below +pi, so the transformed headings cross the
    # branch cut to just above -pi.
    source_heading: Float64Array = np.full(n_frames, 3.10, dtype=np.float64)
    target_heading = wrap_to_pi(source_heading + yaw)
    # Unwrapped the target would be 3.4 rad; wrapping puts it just above -pi.
    assert float(target_heading[0]) == pytest.approx(3.4 - 2.0 * math.pi)

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(scale_prior=0.0),
    )

    assert wrap_to_pi(result.transform.yaw - yaw) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(
        result.transform.apply_heading(source_heading), target_heading, atol=1e-9
    )


def test_heading_residual_is_continuous_at_the_branch_cut() -> None:
    # A heading difference of exactly pi wraps to -pi rather than flipping sign.
    assert wrap_to_pi(math.pi) == -math.pi
    assert wrap_to_pi(-math.pi) == -math.pi
    assert wrap_to_pi(math.pi - 1e-12) == pytest.approx(math.pi - 1e-12)


def test_downweighted_outliers_do_not_drag_the_fit() -> None:
    n_inliers = 30
    scale = 1.2
    yaw = 0.6
    translation = np.array([1.0, -1.0, 0.2])
    source = _track(n_inliers, seed=7)
    target = _apply(source, scale, yaw, translation)
    source_heading, target_heading = _heading_pair(n_inliers, yaw, seed=17)

    # Five frames are corrupted by a large position offset and a heading spike.
    outlier_index: NDArray[np.int64] = np.arange(0, n_inliers, 6, dtype=np.int64)
    target = target.copy()
    target[outlier_index] += np.array([6.0, 6.0, 0.0])
    target_heading = target_heading.copy()
    target_heading[outlier_index] = wrap_to_pi(target_heading[outlier_index] + 1.2)
    position_weights: Float64Array = np.ones(n_inliers, dtype=np.float64)
    position_weights[outlier_index] = 0.01
    heading_weights: Float64Array = np.ones(n_inliers, dtype=np.float64)
    heading_weights[outlier_index] = 0.01

    weighted = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        position_weights=position_weights,
        heading_weights=heading_weights,
    )
    unweighted = fit_similarity(source, target, source_heading, target_heading)

    weighted_scale_error = abs(weighted.transform.scale - scale)
    unweighted_scale_error = abs(unweighted.transform.scale - scale)
    assert weighted_scale_error < unweighted_scale_error
    assert abs(weighted.transform.scale - scale) < 0.01
    assert abs(wrap_to_pi(weighted.transform.yaw - yaw)) < 0.01
    np.testing.assert_allclose(weighted.transform.translation, translation, atol=0.02)


def test_huber_weights_scale_the_loss_and_its_derivatives_once() -> None:
    weights = np.array([2.0, 0.5], dtype=np.float64)
    loss = _make_loss(weights, huber_delta=1.0, has_prior_row=False)
    z = np.array([0.25, 4.0], dtype=np.float64)

    evaluated = loss(z)

    # Inlier: rho(z) = z, rho'(z) = 1, rho''(z) = 0.
    # Outlier: rho(z) = 2 * sqrt(z) - 1, rho'(z) = z ** -0.5, rho''(z) = -0.5 z ** -1.5.
    assert evaluated.shape == (3, 2)
    assert evaluated[0, 0] == pytest.approx(2.0 * 0.25)
    assert evaluated[0, 1] == pytest.approx(0.5 * (2.0 * 2.0 - 1.0))
    assert evaluated[1, 0] == pytest.approx(2.0)
    assert evaluated[1, 1] == pytest.approx(0.5 * 0.5)
    assert evaluated[2, 0] == pytest.approx(0.0)
    assert evaluated[2, 1] == pytest.approx(0.5 * -0.5 * 4.0**-1.5)


def test_prior_row_always_uses_squared_loss() -> None:
    weights = np.array([1.0, 1.0], dtype=np.float64)
    loss = _make_loss(weights, huber_delta=0.5, has_prior_row=True)

    evaluated = loss(np.array([9.0, 9.0], dtype=np.float64))

    # The observation row is far outside the Huber threshold, the prior row is
    # squared loss even though its z is identically large.
    assert evaluated[0, 0] == pytest.approx(2.0 * 0.5 * 3.0 - 0.25)
    assert evaluated[0, 1] == pytest.approx(9.0)
    assert evaluated[1, 0] == pytest.approx(0.5 / 3.0)
    assert evaluated[1, 1] == pytest.approx(1.0)
    assert evaluated[2, 1] == pytest.approx(0.0)


def test_reported_cost_matches_weighted_huber_formula() -> None:
    n_frames = 20
    scale = 1.4
    yaw = -0.5
    source = _track(n_frames, seed=9)
    target = _apply(source, scale, yaw, np.array([0.5, 0.5, 0.5]))
    target[:2] += np.array([3.0, -3.0, 0.0])
    source_heading, target_heading = _heading_pair(n_frames, yaw, seed=19)
    position_weights: Float64Array = np.full(n_frames, 0.5, dtype=np.float64)
    heading_weights: Float64Array = np.full(n_frames, 2.0, dtype=np.float64)
    delta = 1.0

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        position_weights=position_weights,
        heading_weights=heading_weights,
        config=SimilarityConfig(huber_delta=delta),
    )

    z = result.residuals**2
    rho = np.where(z <= delta * delta, z, 2.0 * delta * np.sqrt(z) - delta * delta)
    # The prior row uses squared loss regardless of the Huber threshold.
    rho = np.concatenate((rho[:-1], z[-1:]))
    expected_cost = 0.5 * float(np.sum(result.row_weights * rho))

    assert result.row_weights.shape == result.residuals.shape
    np.testing.assert_allclose(result.row_weights[: 3 * n_frames], 0.5)
    np.testing.assert_allclose(result.row_weights[3 * n_frames : -1], 2.0)
    assert result.row_weights[-1] == 1.0
    assert result.cost == pytest.approx(expected_cost, rel=1e-9)


def test_heading_weight_enters_the_loss_once_not_quadratically() -> None:
    n_frames = 16
    yaw = 0.4
    source = _track(n_frames, seed=21)
    target = _apply(source, 1.0, yaw, np.zeros(3))
    source_heading, target_heading = _heading_pair(n_frames, yaw, seed=23)
    heading_weight = 4.0

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(heading_weight=heading_weight),
    )

    np.testing.assert_allclose(result.row_weights[3 * n_frames : -1], heading_weight)
    # Residuals carry no heading weight, so a quadratic factor would show up as a
    # mismatch between the solver cost and the loss formula.
    heading_residuals = result.residuals[3 * n_frames : -1]
    heading_loss = float(np.sum(heading_weight * 0.5 * heading_residuals**2))
    position_loss = float(np.sum(0.5 * result.residuals[: 3 * n_frames] ** 2))
    prior_loss = float(0.5 * result.residuals[-1] ** 2)
    assert result.cost == pytest.approx(
        position_loss + heading_loss + prior_loss, rel=1e-9
    )


def test_zero_weight_frames_may_hold_non_finite_values() -> None:
    n_frames = 14
    scale = 1.15
    yaw = 1.1
    translation = np.array([3.0, -1.0, 0.4])
    source = _track(n_frames, seed=25)
    target = _apply(source, scale, yaw, translation)
    source_heading, target_heading = _heading_pair(n_frames, yaw, seed=27)
    position_weights: Float64Array = np.ones(n_frames, dtype=np.float64)
    heading_weights: Float64Array = np.ones(n_frames, dtype=np.float64)
    missing = np.array([2, 9], dtype=np.int64)
    source = source.copy()
    target = target.copy()
    source_heading = source_heading.copy()
    target_heading = target_heading.copy()
    source[missing] = np.nan
    target[missing] = np.inf
    source_heading[missing] = np.nan
    target_heading[missing] = np.nan
    position_weights[missing] = 0.0
    heading_weights[missing] = 0.0

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        position_weights=position_weights,
        heading_weights=heading_weights,
        config=SimilarityConfig(scale_prior=0.0),
    )

    assert result.diagnostics.n_position_observations == n_frames - missing.size
    assert result.diagnostics.n_heading_observations == n_frames - missing.size
    assert not result.position_mask[missing].any()
    assert not result.heading_mask[missing].any()
    assert result.heading_mask.sum() == n_frames - missing.size
    assert result.transform.scale == pytest.approx(scale, abs=1e-6)
    assert wrap_to_pi(result.transform.yaw - yaw) == pytest.approx(0.0, abs=1e-6)


def test_non_finite_active_values_are_rejected() -> None:
    source = _track(6, seed=29)
    target = _apply(source, 1.1, 0.2, np.zeros(3))
    source_bad = source.copy()
    source_bad[1, 0] = np.nan
    target_bad = target.copy()
    target_bad[2, 1] = np.inf

    with pytest.raises(ValueError, match="source_position"):
        fit_similarity(source_bad, target)
    with pytest.raises(ValueError, match="target_position"):
        fit_similarity(source, target_bad)


def test_all_zero_position_weights_are_rejected() -> None:
    source = _track(6, seed=31)
    target = _apply(source, 1.1, 0.2, np.zeros(3))

    with pytest.raises(ValueError, match="positive weight"):
        fit_similarity(
            source,
            target,
            position_weights=np.zeros(6, dtype=np.float64),
        )


def test_negative_or_non_finite_weights_are_rejected() -> None:
    source = _track(6, seed=33)
    target = _apply(source, 1.1, 0.2, np.zeros(3))
    negative: Float64Array = np.ones(6, dtype=np.float64)
    negative[3] = -0.5
    non_finite: Float64Array = np.ones(6, dtype=np.float64)
    non_finite[0] = np.nan

    with pytest.raises(ValueError, match="non-negative"):
        fit_similarity(source, target, position_weights=negative)
    with pytest.raises(ValueError, match="NaN or infinity"):
        fit_similarity(source, target, position_weights=non_finite)


def test_stationary_track_without_reliable_heading_is_rejected() -> None:
    source: Float64Array = np.zeros((8, 3), dtype=np.float64)
    target = np.tile(np.array([1.0, 2.0, 0.0]), (8, 1))
    # Alternating headings cancel out, so the circular resultant (and thus the
    # yaw initializer) is unusable.
    source_heading = np.array([0.0, math.pi] * 4, dtype=np.float64)
    target_heading = np.array([0.5, math.pi] * 4, dtype=np.float64)

    # No horizontal spread and no reliable heading: the yaw has no usable
    # initializer.
    with pytest.raises(ValueError, match="cannot initialise"):
        fit_similarity(source, target)
    with pytest.raises(ValueError, match="cannot initialise"):
        fit_similarity(source, target, source_heading, target_heading)


def test_unreliable_heading_resultant_falls_back_to_position_svd() -> None:
    n_frames = 24
    scale = 1.2
    yaw = 0.7
    source = _track(n_frames, seed=35)
    target = _apply(source, scale, yaw, np.array([0.2, 0.3, 0.4]))
    rng = np.random.default_rng(37)
    # Headings that cancel out: the circular resultant drops below the threshold,
    # so the initializer must fall back to the weighted XY Umeyama fit.
    source_heading = rng.uniform(-math.pi, math.pi, n_frames)
    target_heading = wrap_to_pi(rng.uniform(-math.pi, math.pi, n_frames))

    result = fit_similarity(source, target, source_heading, target_heading)

    assert result.diagnostics.heading_resultant_length is not None
    assert (
        result.diagnostics.heading_resultant_length
        < SimilarityConfig().heading_resultant_threshold
    )
    assert result.diagnostics.initializer == POSITION_SVD_UMEYAMA
    assert result.diagnostics.position_xy_yaw == pytest.approx(yaw, abs=1e-6)


def test_stationary_track_free_scale_is_rejected_but_fixed_scale_passes() -> None:
    n_frames = 10
    yaw = 0.45
    source = np.tile(np.array([0.0, 0.0, 1.0]), (n_frames, 1))
    target = np.tile(np.array([4.0, -3.0, 2.0]), (n_frames, 1))
    source_heading: Float64Array = np.full(n_frames, -1.0, dtype=np.float64)
    target_heading = wrap_to_pi(source_heading + yaw)

    # Only vertical separations exist, so a free scale cannot be observed.
    with pytest.raises(ValueError, match="cannot initialise a free scale"):
        fit_similarity(source, target, source_heading, target_heading)

    # With the scale fixed the yaw is observable from the reliable headings and
    # the translation from the mean offset, so the fit is well posed.
    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(fixed_scale=1.0),
    )
    assert result.transform.scale == 1.0
    assert wrap_to_pi(result.transform.yaw - yaw) == pytest.approx(0.0, abs=1e-7)
    np.testing.assert_allclose(
        result.transform.translation, np.array([4.0, -3.0, 1.0]), atol=1e-7
    )


def test_scale_bound_hit_is_reported() -> None:
    n_frames = 16
    yaw = 0.2
    source = _track(n_frames, seed=39)
    target = _apply(source, 5.0, yaw, np.zeros(3))
    source_heading, target_heading = _heading_pair(n_frames, yaw, seed=41)

    result = fit_similarity(
        source,
        target,
        source_heading,
        target_heading,
        config=SimilarityConfig(max_scale=2.0),
    )

    assert result.transform.scale == pytest.approx(2.0, abs=1e-9)
    assert result.diagnostics.scale_at_upper_bound
    assert result.diagnostics.initial_scale_clamped


def test_config_validation_rejects_inconsistent_bounds() -> None:
    with pytest.raises(ValueError, match="min_scale must not exceed max_scale"):
        SimilarityConfig(min_scale=3.0, max_scale=2.0)
    with pytest.raises(ValueError, match="fixed_scale"):
        SimilarityConfig(fixed_scale=5.0, max_scale=2.0)
    with pytest.raises(ValueError, match="sigma_position"):
        SimilarityConfig(sigma_position=0.0)


def test_transform_round_trip_and_validation() -> None:
    transform = SimilarityTransform(
        scale=1.5, yaw=math.pi, translation=np.array([1.0, 2.0, 3.0])
    )
    assert transform.yaw == -math.pi

    points = _track(7, seed=43)
    round_tripped = transform.invert().apply(transform.apply(points))
    np.testing.assert_allclose(round_tripped, points, atol=1e-12)

    with pytest.raises(ValueError, match="shape"):
        transform.apply(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="positive"):
        SimilarityTransform(scale=0.0, yaw=0.0, translation=np.zeros(3))


def test_mismatched_shapes_are_rejected() -> None:
    source = _track(6, seed=45)
    target = _track(7, seed=47)

    with pytest.raises(ValueError, match="same shape"):
        fit_similarity(source, target)
    with pytest.raises(ValueError, match="shape"):
        fit_similarity(
            source,
            source,
            np.zeros(6, dtype=np.float64),
            np.zeros(5, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="both be given or both be omitted"):
        fit_similarity(source, source, np.zeros(6, dtype=np.float64))
    with pytest.raises(ValueError, match="at least one frame"):
        fit_similarity(
            np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        )
