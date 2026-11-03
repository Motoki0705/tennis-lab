from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.geometry.temporal_smoothing import (
    DEFAULT_TEMPORAL_SMOOTHING,
    TemporalSmoothingConfig,
    smooth_2d_trajectory,
)


def smooth(xy, observed=None, *, fps=60., size=(1920, 1080), config=DEFAULT_TEMPORAL_SMOOTHING):
    observed = np.ones(len(xy), bool) if observed is None else observed
    return smooth_2d_trajectory(xy, np.ones(len(xy)), observed, fps=fps, image_size_wh=size, config=config)


@pytest.mark.parametrize("fps", [30., 60., 120.])
def test_fast_motion_and_bounce_survive_with_transient_outlier_removed(fps):
    t = np.arange(int(2 * fps) + 1) / fps
    truth = np.column_stack((200 + 900 * t, 600 - 250 * np.abs(t - 1)))
    points = truth.copy()
    bad = round(.5 * fps)
    points[bad] += [400, -400]
    result = smooth(points, fps=fps)
    np.testing.assert_array_equal(np.flatnonzero(result.rejected), [bad])
    assert result.accepted[round(fps)]
    assert np.linalg.norm(result.coordinates[round(fps)] - truth[round(fps)]) < 8
    assert np.max(np.linalg.norm(result.coordinates[result.accepted] - truth[result.accepted], axis=1)) < 8
    repeated = smooth(points, fps=fps)
    np.testing.assert_array_equal(result.coordinates, repeated.coordinates)
    np.testing.assert_array_equal(result.rejected, repeated.rejected)


def test_noise_reduced_without_phase_shift_and_resolution_scaled():
    rng = np.random.default_rng(13)
    t = np.arange(151) / 60
    truth = np.column_stack((100 + 300 * t, 200 + 20 * t ** 2))
    points = truth + rng.normal(0, 3, truth.shape)
    result = smooth(points)
    assert result.accepted.all()
    assert np.mean((result.coordinates - truth) ** 2) < np.mean((points - truth) ** 2) / 2
    small = smooth(points / 2, size=(960, 540))
    np.testing.assert_allclose(small.coordinates, result.coordinates / 2, atol=1e-5)
    np.testing.assert_array_equal(small.rejected, result.rejected)


def test_short_fills_are_not_observations_long_gaps_and_edges_stay_missing():
    t = np.arange(140)
    truth = np.column_stack((50 + t * 2., 100 + .01 * t ** 2))
    observed: NDArray[np.bool_] = np.ones(140, bool)
    observed[:3] = observed[30:34] = observed[60:90] = observed[-3:] = False
    points = truth.copy()
    points[~observed] = np.nan
    result = smooth(points, observed)
    np.testing.assert_array_equal(result.accepted, observed)
    assert result.interpolated[30:34].all()
    assert not result.interpolated[60:90].any()
    assert not result.interpolated[:3].any()
    assert not result.interpolated[-3:].any()
    assert np.all(result.coordinates[60:90] == 0)
    np.testing.assert_allclose(result.coordinates[30:34], truth[30:34], atol=1e-4)


def test_disabled_sparse_empty_and_invalid_inputs():
    xy = np.array([[0., 0.], [100., 50.], [30., 20.]])
    observed = np.array([True, False, True])
    result = smooth(xy, observed, config=replace(DEFAULT_TEMPORAL_SMOOTHING, enabled=False))
    np.testing.assert_array_equal(result.accepted, observed)
    assert not result.smoothed.any()
    assert not smooth(xy, np.zeros(3, bool)).accepted.any()
    sparse = smooth(xy[:1])
    assert sparse.accepted[0] and not sparse.smoothed[0]
    with pytest.raises(ValueError, match="FPS"):
        smooth(xy, fps=0)
    with pytest.raises(ValueError, match="finite"):
        smooth(np.full((3, 2), np.nan))
    with pytest.raises(ValueError, match="trials"):
        TemporalSmoothingConfig(trials=0)
