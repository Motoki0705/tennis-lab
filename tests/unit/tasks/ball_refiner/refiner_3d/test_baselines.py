import numpy as np
import pytest

from src.tasks.ball_refiner.refiner_3d.baseline_comparison import assert_same_metrics
from src.tasks.ball_refiner.refiner_3d.baseline_rts import smooth_mixture_mean
from src.tasks.ball_refiner.refiner_3d.baselines import condition_points


def test_mixture_mean_keeps_all_components_and_top_uses_weight_only() -> None:
    arrays = {'gmm3d_means_m': np.array([[[0., 0, 0], [10., 20, 30], [100., 0, 0]]]),
              'gmm3d_weights': np.array([[.4, .5, .1]])}
    result = condition_points(arrays)
    np.testing.assert_allclose(result['mixture_mean'], [[15, 10, 15]])
    np.testing.assert_array_equal(result['top_component'], [[10, 20, 30]])
    np.testing.assert_array_equal(arrays['gmm3d_weights'], [[.4, .5, .1]])
    arrays['gmm3d_weights'][0] = [.5, .5, 0]
    # Ties use saved index order, without looking at truth/covariance.
    np.testing.assert_array_equal(condition_points(arrays)['top_component'], [[0, 0, 0]])
    arrays['gmm3d_weights'][0] = [.4, .4, .1]
    with pytest.raises(ValueError, match='normalized'):
        condition_points(arrays)


def test_rts_smooths_ballistic_noise_without_ground_truth_or_frame_rejection() -> None:
    times = np.arange(60) / 60
    truth = np.column_stack((times, times * 2, 8 + times * 3 - .5 * 9.81 * times**2))
    points = truth + np.random.default_rng(42).normal(0, .07, truth.shape)
    smooth, diagnostic = smooth_mixture_mean(points, fps=60)
    assert np.mean((smooth - truth)**2) < np.mean((points - truth)**2)
    assert np.quantile(np.linalg.norm(np.diff(smooth, n=2, axis=0) * 60**2, axis=-1), .95) < 30
    assert diagnostic['event_frames'] == []
    outside = points + [100, 0, 100]
    result, violations = smooth_mixture_mean(outside, fps=60)
    assert len(result) == len(points)
    assert violations['geometric_bounds_violations'] == len(points)
    with pytest.raises(ValueError):
        smooth_mixture_mean(np.full((60, 3), np.nan), fps=60)


def test_historical_metric_and_support_changes_fail_explicitly() -> None:
    assert_same_metrics({'n': 10, 'value': .2, 'gap': None}, {'n': 10, 'value': .2, 'gap': None})
    with pytest.raises(ValueError):
        assert_same_metrics({'n': 9}, {'n': 10})
    with pytest.raises(ValueError):
        assert_same_metrics({'value': .3}, {'value': .2})
