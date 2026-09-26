"""Fixed candidate identity and shared real/synthetic calibration contract."""

from __future__ import annotations

import inspect
import random

import numpy as np
import pytest

import src.tasks.plcs.data.augmentation.residual as cameras_module
from src.tasks.plcs.data.augmentation.residual import (
    fit_court_rig,
    fixed_six_camera_rig,
)
from src.tennis_scene.pipeline.utilts.court_reference import fit_camera
from src.utils.geometry.planar_camera import (
    PlanarCameraFailure,
    PlanarCameraFit,
    fit_planar_camera,
)
from src.utils.geometry.triangulation import project_multiview
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    FENCE_HEIGHT,
    STANDARD_COURT_CONFIG,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    court_keypoints_3d,
)


def _observations() -> tuple[np.ndarray, np.ndarray]:
    rig = fixed_six_camera_rig((1920, 1080))
    world = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(float)
    pixels, depth = project_multiview(world, rig.matrices)
    pixels = pixels.transpose(1, 0, 2)
    scores = (
        (depth.T > 0)
        & (pixels >= 0).all(-1)
        & (pixels <= rig.image_size[:, None]).all(-1)
    ).astype(float)
    return pixels, scores


def test_fixed_six_candidates_preserve_order_and_random_state() -> None:
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    rig = fixed_six_camera_rig((1920, 1080))
    repeated = fixed_six_camera_rig((1920, 1080))

    assert random.getstate() == random_state
    for actual, expected in zip(np.random.get_state(), numpy_state, strict=True):
        np.testing.assert_equal(actual, expected)
    np.testing.assert_array_equal(rig.matrices, repeated.matrices)
    np.testing.assert_allclose(
        rig.centers,
        [
            [X_MIN, Y_MAX, FENCE_HEIGHT],
            [X_MAX, Y_MAX, FENCE_HEIGHT],
            [X_MAX, Y_MIN, FENCE_HEIGHT],
            [X_MIN, Y_MIN, FENCE_HEIGHT],
            [0, Y_MAX, 5],
            [0, Y_MIN, 5],
        ],
        atol=3e-6,
        rtol=0,
    )
    np.testing.assert_allclose(rig.K[:, 0, 0], 1920 / (2 * np.tan(np.deg2rad(30))))
    direction = -rig.centers / np.linalg.norm(rig.centers, axis=1, keepdims=True)
    np.testing.assert_allclose(rig.R[:, 2], direction, atol=1e-7)


def test_visible_court_subset_calibrates_all_six_candidates() -> None:
    pixels, scores = _observations()
    report = fit_court_rig(pixels, scores, (1920, 1080))
    recovered = report.subset(report.valid_indices)

    np.testing.assert_array_equal(scores.sum(1), [12, 12, 12, 12, 10, 10])
    np.testing.assert_array_equal(report.valid_indices, np.arange(6))
    assert report.failures == (None,) * 6
    truth = fixed_six_camera_rig((1920, 1080))
    np.testing.assert_allclose(recovered.K, truth.K, atol=2e-3, rtol=0)
    np.testing.assert_allclose(recovered.centers, truth.centers, atol=3e-5, rtol=0)


@pytest.mark.parametrize("noise_px", [0.0])
@pytest.mark.parametrize("camera_index", [0, 2, 4, 5])
def test_real_wrapper_matches_physical_fit_for_both_sides_and_noise(
    camera_index: int, noise_px: float
) -> None:
    pixels, _ = _observations()
    pixels += np.random.default_rng(123).normal(0, noise_px, pixels.shape)
    report = fit_court_rig(pixels, np.ones((6, 14)), (1920, 1080))
    half_turn = camera_index in (0, 4)
    local = pixels[camera_index]
    if half_turn:
        local = local[list(CAMERA_VIEW_HALF_TURN_INDEX[:14])]
    real_fit = fit_camera(local / [1920, 1080], (1920, 1080), half_turn)
    physical_fit = report.fits[camera_index]

    assert physical_fit is not None
    np.testing.assert_allclose(real_fit["K"], physical_fit.K, atol=2e-3, rtol=0)
    np.testing.assert_allclose(real_fit["R"], physical_fit.R, atol=2e-6, rtol=0)
    np.testing.assert_allclose(real_fit["t"], physical_fit.t, atol=3e-5, rtol=0)
    assert abs(real_fit["rmse_px"] - physical_fit.rmse_px) < 1e-4


def test_real_wrapper_keeps_explicit_side_validation() -> None:
    pixels, _ = _observations()
    # Physical-order +Y observations are deliberately not relabelled to the
    # wrapper's camera-view order. The generic core accepts either side, but
    # the existing provenance wrapper must still reject this declaration.
    with pytest.raises(ValueError, match="contradicts explicit court side"):
        fit_camera(pixels[0] / [1920, 1080], (1920, 1080), False)


def test_per_camera_failures_and_selected_order_are_explicit() -> None:
    pixels, scores = _observations()
    scores[1] = 0
    pixels[1] = np.nan
    pixels[4, :, 0] = 500
    report = fit_court_rig(pixels, scores, np.tile([1920, 1080], (6, 1)))

    np.testing.assert_array_equal(report.valid_indices, [0, 2, 3, 5])
    assert report.fits[1] is None
    assert report.fits[4] is None
    assert report.failures[1] is not None
    assert report.failures[4] is not None
    assert report.failures[1].reason == PlanarCameraFailure.INSUFFICIENT_POINTS
    assert report.failures[4].reason == PlanarCameraFailure.DEGENERATE_POINTS
    selected = report.subset(np.array([5, 0]))
    truth = fixed_six_camera_rig((1920, 1080))
    np.testing.assert_allclose(selected.centers, truth.centers[[5, 0]], atol=3e-5)
    with pytest.raises(ValueError, match="no successful court fit"):
        report.subset(np.array([0, 1]))


def test_fitting_occurs_once_per_camera_without_ground_truth_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixels, scores = _observations()
    calls: list[np.ndarray] = []

    def recorded_fit(
        world_points_m: np.ndarray,
        observed: np.ndarray,
        confidence: np.ndarray,
        image_size: tuple[int, int],
        *,
        min_score: float,
        min_points: int,
    ) -> PlanarCameraFit:
        calls.append(observed.copy())
        return fit_planar_camera(
            world_points_m,
            observed,
            confidence,
            image_size,
            min_score=min_score,
            min_points=min_points,
        )

    monkeypatch.setattr(cameras_module, "fit_planar_camera", recorded_fit)
    report = fit_court_rig(pixels, scores, (1920, 1080))
    report.subset(np.array([0, 1]))
    report.subset(np.array([4, 5, 2]))

    np.testing.assert_array_equal(np.stack(calls), pixels)
    assert set(inspect.signature(fit_court_rig).parameters) == {
        "court_px",
        "court_scores",
        "image_size",
        "min_score",
        "min_points",
    }
