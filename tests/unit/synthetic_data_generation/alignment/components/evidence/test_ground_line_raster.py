"""Tests for per-view ground-line raster aggregation."""

import numpy as np
import pytest

from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
    GroundLineAccumulator,
)
from src.synthetic_data_generation.alignment.components.ground.projection import (
    ProjectedLinePixels,
)


def test_accumulator_limits_duplicate_pixels_to_one_view_contribution() -> None:
    projection = ProjectedLinePixels(
        points_scene=np.asarray([[0.0, 0.0, 0.0], [0.01, 0.01, 0.0]]),
        points_uv=np.asarray([[0.0, 0.0], [0.01, 0.01]]),
        probabilities=np.asarray([0.6, 0.9], dtype=np.float32),
        camera_ranges=np.asarray([1.0, 1.0]),
        proximity_weights=np.asarray([0.5, 0.5]),
        input_count=2,
        invalid_parallel_count=0,
        invalid_behind_count=0,
        invalid_range_count=0,
        invalid_bounds_count=0,
    )
    accumulator = GroundLineAccumulator(
        bounds=(-1.0, 1.0, -1.0, 1.0),
        grid_spacing=0.1,
    )

    assert accumulator.add_view(projection) == 1
    arrays = accumulator.arrays()
    assert arrays["view_count"].max() == 1
    assert arrays["evidence_sum"].max() == pytest.approx(0.45)
    assert arrays["mean_probability"].max() == pytest.approx(0.9)
