from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_refiner.refiner_3d.diffusion.data import window_starts
from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import (
    TrajectoryMetrics,
    metric_values,
    summarize_metrics,
)


def trajectory() -> dict[str, Any]:
    times = np.arange(20) * 1001 / 60000
    truth = np.column_stack((times, .2 * times**2, 4 - .5 * 9.81 * times**2))
    gap: NDArray[np.bool_] = np.zeros((3, 20), dtype=bool)
    gap[:, 5:10] = True
    event: NDArray[np.bool_] = np.zeros(20, dtype=bool)
    event[15:] = True
    return {'timestamps_seconds': times, 'positions_3d_m': truth, 'occlusion_mask': gap,
            'out_of_frame_mask': np.zeros((3, 20), dtype=bool), 'event_region_mask': event,
            'free_flight_mask': ~event, 'camera_true_R': np.tile(np.eye(3), (3, 1, 1)),
            'camera_true_t': np.zeros((3, 3)), 'camera_true_K': np.tile(np.diag([100, 100, 1]), (3, 1, 1))}


def test_absolute_rmse_derivative_units_and_gap_event_support() -> None:
    arrays = trajectory()
    prediction = arrays['positions_3d_m'] + [3, 4, 0]
    values = metric_values(prediction[None], arrays)
    result = summarize_metrics({k: [v] for k, v in values.items()})
    assert result['rmse_m_overall'] == {'count': 20, 'value': 5.}
    assert result['rmse_m_gap'] == {'count': 5, 'value': 5.}
    assert result['rmse_m_event_pm5'] == {'count': 5, 'value': 5.}
    assert result['acceleration_free_flight']['count'] == 13
    assert result['acceleration_free_flight']['p95'] == pytest.approx(np.hypot(.4, 9.81))
    assert result['jerk_free_flight']['count'] == 12
    assert result['jerk_free_flight']['p95'] < 1e-7
    assert result['reprojection_px_all']['p50'] == pytest.approx(np.median(500 / prediction[:, 2]))


def test_no_gap_is_na_and_behind_camera_is_never_a_perfect_reprojection() -> None:
    arrays = trajectory()
    arrays['occlusion_mask'][:] = False
    prediction = arrays['positions_3d_m'].copy()
    prediction[:, 2] *= -1
    values = metric_values(prediction[None], arrays)
    result = summarize_metrics({k: [v] for k, v in values.items()})
    assert result['rmse_m_gap'] == {'count': 0, 'value': None}
    assert result['reprojection_px_all']['count'] == 0
    assert result['reprojection_px_all']['p95'] is None
    assert result['behind_all']['invalid_count'] == 60
    assert not result['reprojection_all_defined']


def test_metrics_pool_frames_and_samples_without_rally_mean_bias() -> None:
    result = summarize_metrics({'error2_overall': [np.ones(9), np.array([100.])],
                                'behind_all': [np.zeros(10)]})
    assert result['rmse_m_overall']['value'] == pytest.approx(np.sqrt(10.9))
    arrays = trajectory()
    predictions = np.stack([arrays['positions_3d_m'] + [1, 0, 0], arrays['positions_3d_m'] - [1, 0, 0]])
    assert np.mean(metric_values(predictions, arrays)['error2_overall']) == pytest.approx(1.)
    assert np.mean(metric_values(predictions.mean(0)[None], arrays)['error2_overall']) < 1e-20


def test_nonuniform_timestamps_are_rejected() -> None:
    arrays = trajectory()
    arrays['timestamps_seconds'][4] += .001
    with pytest.raises(ValueError, match='uniformly'):
        metric_values(arrays['positions_3d_m'][None], arrays)


def test_strata_partition_original_derivatives_and_do_not_join_disjoint_frames() -> None:
    arrays = trajectory()
    # Alternate 0/1/2/3 cameras; separated frames must not be joined to take a
    # difference. Each derivative stencil is assigned by its central frame.
    visible: NDArray[np.int64] = np.arange(20) % 4
    arrays['occlusion_mask'] = np.arange(3)[:, None] >= visible[None]
    metric = TrajectoryMetrics()
    metric.add(arrays['positions_3d_m'][None], arrays)
    summary = metric.summarize()
    for key in ('rmse_m_overall', 'acceleration_all', 'jerk_all', 'acceleration_free_flight', 'reprojection_px_all', 'behind_all'):
        assert sum(s[key]['count'] for s in summary['by_visible_cameras'].values()) == summary['metrics'][key]['count']
    for stratum in summary['by_visible_cameras'].values():
        assert stratum['rmse_m_overall']['count'] == 5
        assert stratum['acceleration_all']['p95'] == pytest.approx(np.hypot(.4, 9.81))
        assert stratum['jerk_all']['p95'] < 1e-7


def test_camera_strata_use_visibility_including_out_of_frame_not_presence() -> None:
    arrays = trajectory()
    arrays['occlusion_mask'][:] = False
    arrays['out_of_frame_mask'][1:] = True
    metric = TrajectoryMetrics()
    metric.add(arrays['positions_3d_m'][None], arrays)
    summary = metric.summarize()['by_visible_cameras']
    assert summary['1']['rmse_m_overall']['count'] == 20
    assert summary['3']['rmse_m_overall'] == {'count': 0, 'value': None}


@pytest.mark.parametrize('length', [4, 127, 128, 129, 130, 255, 512])
def test_windows_cover_every_frame_including_short_tail(length: int) -> None:
    starts = window_starts(length, 128, 128)
    coverage: NDArray[np.bool_] = np.zeros(length, dtype=bool)
    assert len(set(starts)) == len(starts)
    for start in starts:
        assert length - start >= 3
        coverage[start:start + 128] = True
    assert coverage.all()
