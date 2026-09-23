"""Unnormalized pose scores must not leak into scene visibility."""

from __future__ import annotations

import numpy as np
import pytest

from src.tennis_scene.pipeline.observations import pose_visibility_from_heatmap_peaks


def test_pose_visibility_bounds_raw_peaks_without_modifying_observations() -> None:
    raw = np.array([0.0, 0.4, 1.02, -0.1], dtype=np.float32)
    before = raw.copy()
    visibility, audit = pose_visibility_from_heatmap_peaks(raw)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(visibility, np.array([0, 0.4, 1, 0], np.float32))
    assert audit["conversion"] == "clip_0_1"
    assert audit["saturated_above_one_count"] == 1
    assert audit["saturated_below_zero_count"] == 1
    assert audit["raw_max"] == pytest.approx(1.02)


@pytest.mark.parametrize("raw", [[], [np.nan], [np.inf], [-np.inf]])
def test_pose_visibility_rejects_invalid_model_observations(raw: list[float]) -> None:
    with pytest.raises(ValueError, match="nonempty and finite"):
        pose_visibility_from_heatmap_peaks(np.array(raw, dtype=np.float32))
