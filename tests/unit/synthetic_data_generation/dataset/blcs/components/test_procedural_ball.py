"""Tests for deterministic metric procedural-ball Gaussian geometry."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.blcs.components.procedural_ball import (
    build_procedural_ball_geometry,
)


def test_procedural_ball_is_deterministic_centred_and_metric() -> None:
    first = build_procedural_ball_geometry()
    second = build_procedural_ball_geometry()

    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.log_scales, second.log_scales)
    np.testing.assert_allclose(first.means.mean(axis=0), 0.0, atol=1.0e-9)
    np.testing.assert_allclose(
        first.three_sigma_radii_m(),
        0.067 * 0.5,
        atol=1.0e-7,
    )
    assert first.gaussian_count == 512
    assert not first.means.flags.writeable
    assert cast(float, first.metric_summary()["mean_offset_m"]) < 1.0e-9


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("nominal_diameter_m", 0.049),
        ("gaussian_count", 31),
        ("gaussian_count", 33),
        ("gaussian_sigma_fraction", 0.0),
        ("gaussian_sigma_fraction", 1.0 / 3.0),
        ("opacity", 1.0),
    ],
)
def test_procedural_ball_rejects_invalid_geometry(
    argument: str,
    value: float | int,
) -> None:
    with pytest.raises(ValueError):
        cast(Any, build_procedural_ball_geometry)(**{argument: value})
