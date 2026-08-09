"""Tests for explicit normalized-NHT to metric-scene depth conversion."""

from __future__ import annotations

import numpy as np
import pytest

from src.synthetic_data_generation.rendering.nht.depth import nht_depth_to_metric


def test_nht_depth_to_metric_preserves_zero_and_converts_scale() -> None:
    depth = np.asarray([[[0.0], [0.25], [1.0]]], dtype=np.float32)

    converted = nht_depth_to_metric(
        depth,
        nht_scene_units_per_metre=0.25,
    )

    assert converted.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        converted,
        np.asarray([[[0.0], [1.0], [4.0]]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("depth", "error"),
    [
        (np.ones((1, 1, 1), dtype=np.float64), TypeError),
        (np.asarray([[[-1.0]]], dtype=np.float32), ValueError),
        (np.asarray([[[np.nan]]], dtype=np.float32), ValueError),
        (np.asarray([[[np.inf]]], dtype=np.float32), ValueError),
    ],
)
def test_nht_depth_to_metric_rejects_invalid_public_arrays(
    depth: np.ndarray,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        nht_depth_to_metric(depth, nht_scene_units_per_metre=0.25)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_nht_depth_to_metric_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        nht_depth_to_metric(
            np.ones((1, 1, 1), dtype=np.float32),
            nht_scene_units_per_metre=scale,
        )


def test_nht_depth_to_metric_rejects_boolean_scale() -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        nht_depth_to_metric(
            np.ones((1, 1, 1), dtype=np.float32),
            nht_scene_units_per_metre=True,
        )


def test_nht_depth_to_metric_rejects_unrepresentable_scale_and_overflow() -> None:
    depth = np.asarray([[[np.finfo(np.float32).max]]], dtype=np.float32)
    with pytest.raises(ValueError, match="representable as float32"):
        nht_depth_to_metric(depth, nht_scene_units_per_metre=1.0e-300)
    with pytest.raises(ValueError, match="overflow float32"):
        nht_depth_to_metric(depth, nht_scene_units_per_metre=0.5)
