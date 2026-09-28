"""Unequal cluster sizes and shared paired resampling, without frame-IID intervals."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_refiner.evaluation.bootstrap import grouped_intervals


def test_whole_group_resampling_keeps_frame_weighted_estimand_and_paired_difference():
    groups = ["clip-a"] * 9 + ["clip-b"] * 3
    values = np.array([1.] * 9 + [9.] * 3)
    report = grouped_intervals({"mean": (values, "mean"), "p95": (values, "p95"),
                                "paired_delta": (np.full(12, -2.), "mean")}, groups,
                               repetitions=1000, confidence=.95, seed=5)
    assert report["mean"] == {"value": 3., "ci_low": 1., "ci_high": 9.}
    assert report["paired_delta"] == {"value": -2., "ci_low": -2., "ci_high": -2.}
    assert report["p95"]["value"] == 9.


def test_frame_order_does_not_change_intervals():
    groups = ["b", "a", "b", "c", "a", "c"]
    values: NDArray[np.float64] = np.arange(6, dtype=np.float64)
    first = grouped_intervals({"mean": (values, "mean"), "median": (values, "median")}, groups, repetitions=60, confidence=.9, seed=14)
    second = grouped_intervals({"mean": (values[::-1], "mean"), "median": (values[::-1], "median")}, groups[::-1], repetitions=60, confidence=.9, seed=14)
    assert first == second


@pytest.mark.parametrize("values,groups", [(np.array([1.]), ["a"]), (np.array([1., np.nan]), ["a", "b"]),
                                          (np.array([1.]), ["a", "b"]), (np.array([1., 2.]), ["", "b"])])
def test_invalid_or_single_group_evidence_cannot_be_bootstrapped(values, groups):
    with pytest.raises(ValueError):
        grouped_intervals({"mean": (values, "mean")}, groups, repetitions=10, confidence=.95, seed=1)
