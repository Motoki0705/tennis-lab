"""Density normalization, reference-label separation and absent/unknown denominators."""

import math

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner.data.targets import ClipTargets, TargetReason
from src.tasks.ball_refiner.evaluation.paired_metrics import (
    gmm_rows,
    heatmap_rows,
    strata,
    summarize,
)
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


def targets(reasons, uv=None):
    reasons = np.asarray(reasons, np.uint8)
    n = len(reasons)
    present = reasons == TargetReason.OBSERVED
    known = present | (reasons == TargetReason.OUT_OF_FRAME)
    return ClipTargets(np.arange(n, dtype=np.int32), np.arange(n), np.arange(n, dtype=np.float32),
                       np.full((n, 2), .5, np.float32) if uv is None else np.asarray(uv, np.float32),
                       present, present, known, reasons, np.zeros(n, bool), np.zeros(n, np.uint8))


@pytest.mark.parametrize('value', [0., .3, 1.])
def test_uniform_and_zero_density_are_normalized_with_conservative_tie_regions(value):
    t = targets([0, 0, 0], [[0, 0], [1, 1], [.5, .5]])
    rows = heatmap_rows(np.full((3, 3, 5), value, np.float32), t, (101, 51), (.99, .98),
                        levels=(.5, .9, .95), uniform_weight=.01)
    np.testing.assert_allclose(rows['nll_uv'], 0, atol=1e-14)
    np.testing.assert_allclose(rows['nll_px'], math.log(5000))
    assert rows['coverage'].all()
    np.testing.assert_allclose(rows['area_px2'], 5000)


def test_heatmap_mass_and_coverage_match_two_by_two_analytic_distribution():
    t = targets([0, 0], [[.1, .1], [.9, .9]])
    maps: NDArray[np.float32] = np.zeros((2, 2, 2), np.float32)
    maps[:, 0, 0] = 1
    rows = heatmap_rows(maps, t, (101, 51), (1., 1.), levels=(.5, .9, .95), uniform_weight=.1)
    np.testing.assert_allclose(rows['nll_uv'], -np.log([3.7, .1]))
    np.testing.assert_array_equal(rows['coverage'], [[1, 1, 1], [0, 0, 1]])
    np.testing.assert_allclose(rows['area_px2'], [[1250, 1250, 5000]] * 2)


def test_native_endpoint_cells_have_half_width_at_image_boundaries():
    t = targets([0], [[.5, .5]])
    maps: NDArray[np.float32] = np.zeros((1, 3, 3), np.float32)
    maps[0, 1, 1] = 1
    rows = heatmap_rows(maps, t, (101, 101), (1., 1.), levels=(.9,), uniform_weight=.01)
    assert rows['nll_uv'][0] == pytest.approx(-math.log(3.97))
    assert rows['area_px2'][0, 0] == pytest.approx(2500)


def test_estimated_and_absent_labels_are_not_observed_or_unknown_presence():
    t = targets([0, 1, 3, 6, 5, 7])
    gmm = BallGMM2D(torch.full((1, 6, 1, 2), .5), torch.eye(2).expand(1, 6, 1, 2, 2) * .1,
                    torch.zeros(1, 6, 1), torch.zeros(1, 6))
    rows = gmm_rows(gmm, t, (101, 51), levels=(.5, .9), samples=128, seed=1729, chunk_size=2,
                    clip_id='test', condition='observed', device=torch.device('cpu'))
    masks = strata(t, np.ones(6, bool), 'observed')
    reports = {key: summarize([{name: values[mask] for name, values in rows.items()}], (.5, .9)) for key, mask in masks.items()}
    assert reports['observed']['error_px_frames'] == 1
    assert reports['occlusion_estimated_reference']['error_px_frames'] == 1
    assert reports['interpolated_reference']['error_px_frames'] == 1
    assert reports['absent']['mean_presence_nll'] == pytest.approx(math.log(2))
    assert reports['absent']['mean_nll_px'] is None
    assert reports['absent']['coverage_0.9'] is None
    assert reports['no_instance_unknown']['mean_presence_nll'] is None
    assert reports['unresolved']['mean_nll_px'] is None
    assert reports['observed']['mean_nll_uv'] == pytest.approx(math.log(2 * math.pi * .01))


def test_artificial_gap_scores_only_the_same_fixed_frames():
    t = targets([0, 0, 0, 6])
    mask = np.array([False, True, False, True])
    result = strata(t, mask, 'evidence_gap')
    np.testing.assert_array_equal(result['observed'], [False, True, False, False])
    np.testing.assert_array_equal(result['occlusion_estimated_reference'], [False, False, False, True])


def test_heatmap_rejects_nonfinite_input_and_out_of_frame_spatial_reference():
    t = targets([0], [[1.1, .5]])
    with pytest.raises(ValueError, match='inside'):
        heatmap_rows(np.ones((1, 3, 3), np.float32), t, (101, 101), (1., 1.), levels=(.9,), uniform_weight=.01)
    with pytest.raises(ValueError, match='finite'):
        heatmap_rows(np.full((1, 3, 3), np.nan, np.float32), t, (101, 101), (1., 1.), levels=(.9,), uniform_weight=.01)
