"""Observed plane calibration and explicit unusable-geometry failures."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from src.utils.geometry.planar_camera import (
    PlanarCameraFailure,
    PlanarCameraFitError,
    fit_planar_camera,
)
from src.utils.projection.camera_projector import make_look_at_camera
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def _plane_observations(
    center: tuple[float, float, float] = (5.0, -22.0, 6.0),
    hfov_deg: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(float)
    camera = make_look_at_camera(
        center,
        look_at=(0.0, 0.0, 0.0),
        image_size=(1920, 1080),
        hfov_deg=hfov_deg,
    )
    intrinsic = np.array(
        [[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]], dtype=float
    )
    # Rodrigues removes float32 orthogonality error from the fixture camera.
    rotation_vector = cv2.Rodrigues(camera.R.numpy().astype(float))[0]
    rotation = cv2.Rodrigues(rotation_vector)[0]
    translation = -rotation @ np.asarray(center)
    pixels = cv2.projectPoints(world, rotation_vector, translation, intrinsic, None)[0][
        :, 0
    ]
    return world, pixels, intrinsic, rotation, translation


@pytest.mark.parametrize("side", [-1, 1])
def test_clean_observations_recover_camera_on_either_court_side(side: int) -> None:
    world, pixels, intrinsic, rotation, translation = _plane_observations(
        (5.0, side * 22.0, 6.0)
    )
    fit = fit_planar_camera(world, pixels, np.ones(14), (1920, 1080))

    np.testing.assert_allclose(fit.K, intrinsic, atol=2e-3, rtol=0)
    np.testing.assert_allclose(fit.R, rotation, atol=2e-6, rtol=0)
    np.testing.assert_allclose(fit.t, translation, atol=3e-5, rtol=0)
    assert fit.rmse_px < 1e-4
    assert fit.diagnostics.min_depth_m > 0
    assert fit.diagnostics.camera_height_above_plane_m > 0


def test_missing_points_and_low_score_outliers_do_not_bias_fit() -> None:
    world, pixels, intrinsic, rotation, translation = _plane_observations()
    scores = np.ones(14)
    scores[[8, 10]] = 0
    pixels[[8, 10]] = np.nan
    scores[12] = 0.1
    pixels[12] = [1500, 50]

    fit = fit_planar_camera(world, pixels, scores, (1920, 1080))

    assert not fit.used_mask[[8, 10, 12]].any()
    assert fit.diagnostics.used_count == int(fit.used_mask.sum())
    np.testing.assert_allclose(fit.K, intrinsic, atol=2e-3, rtol=0)
    np.testing.assert_allclose(fit.R, rotation, atol=2e-6, rtol=0)
    np.testing.assert_allclose(fit.t, translation, atol=3e-5, rtol=0)


def test_out_of_image_outliers_cannot_supply_calibration_support() -> None:
    world, pixels, _, _, _ = _plane_observations()
    inside = np.flatnonzero((pixels >= 0).all(-1) & (pixels <= [1920, 1080]).all(-1))
    pixels[inside[5:]] = [2000, 1200]
    with pytest.raises(PlanarCameraFitError) as error:
        fit_planar_camera(world, pixels, np.ones(14), (1920, 1080))
    assert error.value.reason == PlanarCameraFailure.INSUFFICIENT_POINTS


@pytest.mark.parametrize("space", ["world", "image"])
def test_collinear_coverage_is_rejected(space: str) -> None:
    world, pixels, _, _, _ = _plane_observations()
    if space == "world":
        world[:, 1] = 0
    else:
        pixels[:, 0] = np.linspace(100, 1800, 14)
        pixels[:, 1] = 500
    with pytest.raises(PlanarCameraFitError) as error:
        fit_planar_camera(world, pixels, np.ones(14), (1920, 1080))
    assert error.value.reason == PlanarCameraFailure.DEGENERATE_POINTS


@pytest.mark.parametrize("invalid", ["nan", "score", "plane", "size"])
def test_invalid_calibration_input_has_typed_failure(invalid: str) -> None:
    world, pixels, _, _, _ = _plane_observations()
    scores = np.ones(14)
    size = (1920, 1080)
    if invalid == "nan":
        pixels[0, 0] = np.nan
    elif invalid == "score":
        scores[0] = -1
    elif invalid == "plane":
        world[0, 2] = 1
    else:
        size = (0, 1080)
    with pytest.raises(PlanarCameraFitError) as error:
        fit_planar_camera(world, pixels, scores, size)
    assert error.value.reason == PlanarCameraFailure.INVALID_INPUT


def test_camera_below_known_plane_is_rejected() -> None:
    world, pixels, _, _, _ = _plane_observations((5.0, -22.0, -6.0))
    with pytest.raises(PlanarCameraFitError) as error:
        fit_planar_camera(world, pixels, np.ones(14), (1920, 1080))
    assert error.value.reason == PlanarCameraFailure.CAMERA_BELOW_PLANE


def test_focal_optimization_cannot_silently_saturate_bounds() -> None:
    world, pixels, _, _, _ = _plane_observations(hfov_deg=150.0)
    with pytest.raises(PlanarCameraFitError) as error:
        fit_planar_camera(world, pixels, np.ones(14), (1920, 1080))
    assert error.value.reason == PlanarCameraFailure.INVALID_FOCAL_LENGTH


def test_negative_depth_planar_solution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, pixels, _, _, _ = _plane_observations()
    original_solve = cv2.solvePnP

    def reversed_depth(
        *args: Any, **kwargs: Any
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        ok, rotation_vector, translation = original_solve(*args, **kwargs)
        # A plane admits this homogeneous sign ambiguity: the reprojection is
        # unchanged although all plane points are now behind the camera.
        rotation = cv2.Rodrigues(rotation_vector)[0] @ np.diag([-1.0, -1.0, 1.0])
        return bool(ok), cv2.Rodrigues(rotation)[0], -translation

    monkeypatch.setattr(cv2, "solvePnP", reversed_depth)
    with pytest.raises(PlanarCameraFitError) as error:
        fit_planar_camera(world, pixels, np.ones(14), (1920, 1080))
    assert error.value.reason == PlanarCameraFailure.INVALID_DEPTH
