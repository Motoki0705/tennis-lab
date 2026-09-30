"""Pixel anisotropy, top-weight selection and strict observed error buckets."""

import numpy as np
import pytest

from src.tasks.ball_refiner.evaluation.precision import detector_groups, precision_rows


def test_pixel_sigma_and_mahalanobis_select_the_same_component() -> None:
    means = np.array([[[.1, .1], [.5, .5]]])
    scales = np.array([[[[.5, 0], [0, .5]], [[.02, 0], [.01, .02]]]])
    target = np.array([[.52, .51]])
    rows = precision_rows(means, scales, np.array([[-3., 2.]]), target, target,
                          np.array([[[.5, .5], [.9, .9]]]), np.array([[True, True]]), (101, 201))
    assert rows['error_px'][0] == pytest.approx(np.sqrt(8))
    assert rows['z_radius'][0] == pytest.approx(1)
    assert rows['z_y'][0] == pytest.approx(0, abs=1e-12)
    expected = np.sqrt(np.linalg.eigvalsh(np.array([[4., 4.], [4., 20.]]))[-1])
    assert rows['sigma_major_px'][0] == pytest.approx(expected)
    assert rows['nearest_candidate_px'][0] == 0


def test_missing_candidates_are_not_zero_coordinate_observations() -> None:
    rows = precision_rows(np.zeros((1, 1, 2)), np.eye(2).reshape(1, 1, 2, 2),
                          np.zeros((1, 1)), np.zeros((1, 2)), np.zeros((1, 2)),
                          np.zeros((1, 2, 2)), np.zeros((1, 2), bool), (101, 101))
    assert np.isnan(rows['nearest_candidate_px'][0])


def test_detector_buckets_partition_boundary_values() -> None:
    groups = detector_groups(np.array([0., 8., 8.1, 20., 20.1]))
    np.testing.assert_equal(groups['within8'], [True, True, False, False, False])
    np.testing.assert_equal(groups['8to20'], [False, False, True, True, False])
    np.testing.assert_equal(groups['wrong'], [False, False, False, False, True])
    with pytest.raises(ValueError, match='finite'):
        detector_groups(np.array([np.nan]))
