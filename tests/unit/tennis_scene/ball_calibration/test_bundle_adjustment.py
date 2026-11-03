"""Known camera and signed clock recovery, masks, and explicit refusal."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.tennis_scene.ball_calibration.bundle_adjustment import (
    BundleAdjustmentConfig,
    court_world_points,
    refine_court_ball,
    resample_observations,
)
from tests.support.court_ball_calibration import (
    calibration_fixture,
    destroyed_holdout,
)


def fit(fixture, **kwargs):
    obs, cameras, courts, turns, _, _, fps = fixture
    return refine_court_ball(
        obs, cameras, courts, turns, fps=fps, reprojection_px=30., min_frames=1,
        config=BundleAdjustmentConfig(prior_weight=0, max_nfev=300, **kwargs),
    )


@pytest.fixture(scope="module")
def recovered():
    fixture = calibration_fixture()
    return fixture, fit(fixture)


def test_recovers_camera_perturbations_and_signed_fractional_clocks(recovered):
    fixture, result = recovered
    _, _, _, _, truth, offsets, fps = fixture
    np.testing.assert_allclose(result.calibration.time_offsets_seconds * fps, offsets, atol=.025)
    assert result.calibration.time_offsets_seconds[0] == 0
    for actual, target in zip(result.calibration.cameras, truth, strict=True):
        assert actual.intrinsic[0, 0] == pytest.approx(target.intrinsic[0, 0], rel=.002)
        np.testing.assert_allclose(actual.center, target.center, atol=.025)
        angle = Rotation.from_matrix(actual.rotation @ target.rotation.T).magnitude()
        assert angle < .001
        np.testing.assert_array_equal(actual.intrinsic[:2, 2], [960, 540])
    assert result.calibration.diagnostics["court_fit_count"] == [12, 12, 12]
    assert result.calibration.diagnostics["court_holdout_count"] == [2, 2, 2]


def test_holdout_never_enters_fit_and_camera_permutation_preserves_identity(recovered):
    fixture, original = recovered
    obs, cameras, courts, turns, truth, offsets, fps = fixture
    # Reverse all view axes; the clock anchor remains a_near, not array index 0.
    order = [2, 1, 0]
    other_obs = replace(obs, uv_px=obs.uv_px[:, order], confidence=obs.confidence[:, order],
                        visibility=obs.visibility[:, order], raw_indices=obs.raw_indices[:, order])
    damaged = destroyed_holdout(courts)
    other = fit((other_obs, tuple(cameras[i] for i in order), tuple(damaged[i] for i in order),
                 tuple(turns[i] for i in order), truth, offsets, fps))
    np.testing.assert_allclose(other.calibration.time_offsets_seconds[::-1], original.calibration.time_offsets_seconds, atol=2e-5)
    for actual, previous in zip(other.calibration.cameras[::-1], original.calibration.cameras, strict=True):
        np.testing.assert_allclose(actual.matrix, previous.matrix, rtol=0, atol=.02)


def test_clock_resampling_uses_t_minus_delta_and_never_bridges_missing_data():
    obs, *_ = calibration_fixture(frames=5)
    uv = np.array([[[[10., 20.]], [[18., 26.]], [[30., 38.]], [[50., 46.]], [[90., 74.]]]], np.float32)
    uv = np.repeat(uv[None], 3, axis=1).reshape(1, 3, 5, 1, 2)
    mask: NDArray[np.bool_] = np.ones((1, 3, 5, 1), bool)
    mask[0, 1, 2] = False
    obs = replace(obs, uv_px=uv, visibility=mask)
    corrected = resample_observations(obs, np.array([0., .25, -.5]))
    np.testing.assert_array_equal(corrected.uv_px[0, 0], obs.uv_px[0, 0])
    np.testing.assert_allclose(corrected.uv_px[0, 1, 1, 0], [16, 24.5])
    np.testing.assert_allclose(corrected.uv_px[0, 2, 1, 0], [24, 32])
    np.testing.assert_array_equal(corrected.visibility[0, 1, :, 0], [False, True, False, False, True])
    np.testing.assert_array_equal(corrected.visibility[0, 2, :, 0], [True, True, True, True, False])
    assert (corrected.uv_px[~corrected.visibility] == 0).all()


def test_half_turn_and_prosac_provenance_do_not_double_reflect_semantics():
    a, b = court_world_points(False), court_world_points(True)
    np.testing.assert_allclose(a[0], [-5.485, 11.885, 0], atol=1e-6)
    np.testing.assert_allclose(b[11], [-4.115, 6.4, 0], atol=1e-6)


@pytest.mark.parametrize("changes", [
    {"court_sigma_px": float("nan")}, {"ball_sigma_px": 0},
    {"prior_weight": -1}, {"max_offset_frames": 1},
    {"max_ball_frames": True}, {"max_nfev": 0},
])
def test_invalid_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        BundleAdjustmentConfig(**changes)


@pytest.mark.parametrize("failure", ["camera_axis", "court_support", "ball_support", "static_ball", "solver_budget"])
def test_unidentifiable_or_failed_fit_has_no_baseline_substitution(failure):
    fixture = list(calibration_fixture(frames=12))
    if failure == "camera_axis":
        fixture[2] = fixture[2][::-1]
    elif failure == "court_support":
        mask: NDArray[np.bool_] = np.zeros((1, 14), bool)
        mask[0, [0, 1, 4, 6]] = True  # all on the same far baseline
        fixture[2] = (replace(fixture[2][0], selected_for_homography=mask), *fixture[2][1:])
    elif failure == "ball_support":
        fixture[0] = replace(fixture[0], visibility=np.zeros_like(fixture[0].visibility))
    elif failure == "static_ball":
        fixture[0] = replace(fixture[0], uv_px=np.repeat(fixture[0].uv_px[:, :, 5:6], 12, axis=2))
    if failure == "solver_budget":
        obs, cameras, courts, turns, _, _, fps = fixture
        with pytest.raises(RuntimeError, match="failed"):
            refine_court_ball(obs, cameras, courts, turns, fps=fps, reprojection_px=30,
                              min_frames=1, config=BundleAdjustmentConfig(max_nfev=1))
    else:
        with pytest.raises(ValueError, match="axes|noncollinear|correspondences"):
            fit(fixture)
