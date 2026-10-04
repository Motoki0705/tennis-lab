"""A gap must not become an observed trajectory segment."""

import numpy as np
import pytest

from src.tennis_scene.review.trajectory import mask_intervals, masked_trajectory


@pytest.mark.parametrize("mask,intervals", [
    ([], []), ([False, False], []), ([True, True], [(0, 2)]),
    ([True, False, True], [(0, 1), (2, 3)]),
    ([False, True, True, False, True, False], [(1, 3), (4, 5)]),
])
def test_source_frame_intervals(mask: list[bool], intervals: list[tuple[int, int]]) -> None:
    assert mask_intervals(np.asarray(mask, bool)) == intervals


def test_nan_separator_preserves_valid_origin_and_isolated_sample() -> None:
    points = np.array([[0., 0.], [1., 1.], [99., 99.], [5., 5.], [99., 99.]])
    original = points.copy()
    mask = np.array([True, True, False, True, False])
    plotted = masked_trajectory(points, mask)
    assert plotted.shape == points.shape
    np.testing.assert_array_equal(plotted[mask], points[mask])
    assert np.isnan(plotted[~mask]).all()
    assert np.array_equal(points, original)
    # Matplotlib can connect exactly the first observed pair, never 1 -> 3.
    finite = np.isfinite(plotted).all(-1)
    assert np.flatnonzero(finite[:-1] & finite[1:]).tolist() == [0]


def test_scalar_height_keeps_missing_endpoints() -> None:
    plotted = masked_trajectory(np.array([99., 0., 2., 99.]), np.array([False, True, True, False]))
    assert np.isnan(plotted[[0, 3]]).all()
    assert plotted[1:3].tolist() == [0., 2.]


def test_coordinate_values_cannot_substitute_for_a_validity_mask() -> None:
    with pytest.raises(ValueError, match="boolean"):
        masked_trajectory(np.zeros((3, 2)), np.array([0., 1., 0.]))
    with pytest.raises(ValueError, match="boolean"):
        mask_intervals(np.ones((2, 3), bool))
