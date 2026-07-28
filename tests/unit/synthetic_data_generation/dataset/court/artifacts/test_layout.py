"""Tests for strict multi-court instance geometry."""

from __future__ import annotations

import numpy as np

from src.synthetic_data_generation.dataset.court.artifacts.layout import (
    MultiCourtLayout,
)


def test_two_courts_keep_distinct_instances_and_metric_centres(
    two_court_layout: MultiCourtLayout,
) -> None:
    assert [court.court_instance_id for court in two_court_layout.courts] == [
        "court_0",
        "court_1",
    ]
    np.testing.assert_allclose(
        two_court_layout.centers_in_reference(),
        ((0.0, 0.0, 0.0), (15.0, 0.0, 0.0)),
        atol=1.0e-12,
    )
