"""Censored peaks are not independent measured intersections."""

from functools import partial

import numpy as np

from src.tasks.court_detection.geometry.observations import keypoint_support


def test_censoring_uses_native_cells_and_preserves_raw_coordinates() -> None:
    raw = np.array([[0., 50.], [10., 50.], [100., 70.], [102., 70.], [200., 190.], [np.nan, 4.]])
    original = raw.copy()
    support = keypoint_support(raw, np.ones(6), image_size_hw=(201, 301), native_size_hw=(101, 151),
        min_score=.05, boundary_margin_cells=1, duplicate_distance_cells=1)
    np.testing.assert_array_equal(support.boundary, [True, False, False, False, False, False])
    np.testing.assert_array_equal(support.duplicate, [False, False, True, True, False, False])
    np.testing.assert_array_equal(support.eligible, [False, True, False, False, True, False])
    np.testing.assert_array_equal(raw, original)
    assert support.diagnostics()["duplicate_peak_indices"] == [2, 3]


def test_source_resize_preserves_peak_eligibility_and_low_scores_do_not_collide() -> None:
    raw = np.array([[5., 10.], [5.1, 10.1], [1., 20.], [15., 10.]])
    scores = np.array([.9, .01, .8, .8])
    support = partial(keypoint_support, native_size_hw=(51, 101), min_score=.05, boundary_margin_cells=1., duplicate_distance_cells=1.)
    first = support(raw, scores, image_size_hw=(101, 201))
    second = support(raw * 3, scores, image_size_hw=(301, 601))
    np.testing.assert_array_equal(first.eligible, [True, False, False, True])
    np.testing.assert_array_equal(first.eligible, second.eligible)
