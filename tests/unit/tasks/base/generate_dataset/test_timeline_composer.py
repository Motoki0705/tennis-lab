from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tasks.base.generate_dataset.timeline_composer import (
    TimelineComposer,
    TimelineConfig,
)


def _config(**overrides: object) -> TimelineConfig:
    values = {
        "num_frames": 64,
        "min_tracks": 4,
        "max_tracks": 4,
        "max_concurrent": 2,
        "min_reuse_gap_frames": 4,
        "start_index_range": (-16, 56),
        "min_active_frames": 8,
        "overlap_probability": 0.5,
        "min_gap_frames": 4,
        "max_gap_frames": 12,
    }
    values.update(overrides)
    return TimelineConfig(**values)  # type: ignore[arg-type]


def test_composer_builds_fixed_timeline_and_never_exceeds_concurrency() -> None:
    composer = TimelineComposer(_config(), rng=random.Random(7))
    result = composer.compose(
        [f"source_{index}" for index in range(4)],
        [24, 20, 28, 18],
    )

    assert result.present.shape == (64, 4)
    assert result.present.sum(axis=1).max() <= 2
    occupancy: NDArray[np.int64] = np.zeros(64, dtype=np.int64)
    for placement in result.placements:
        occupancy[
            placement.birth_frame : min(64, placement.death_frame + 4)
        ] += 1
    assert occupancy.max() <= 2
    assert all(placement.num_active_frames >= 8 for placement in result.placements)
    assert [placement.track_id for placement in result.placements] == list(range(4))
    for placement in result.placements:
        assert placement.source_end - placement.source_start == placement.num_active_frames


def test_composition_places_numpy_and_tensor_sources_with_zero_inactive_values() -> None:
    composer = TimelineComposer(
        _config(min_tracks=2, max_tracks=2, max_concurrent=1),
        rng=random.Random(11),
    )
    sources_np: list[NDArray[np.float32]] = [
        np.full((20, 3), index + 1, dtype=np.float32) for index in range(2)
    ]
    result = composer.compose(["a", "b"], [20, 20])
    composed_np = result.compose_numpy(sources_np)
    composed_tensor = result.compose_tensor(
        [torch.from_numpy(source) for source in sources_np]
    )

    assert composed_np.shape == (64, 2, 3)
    np.testing.assert_array_equal(composed_np, composed_tensor.numpy())
    for placement in result.placements:
        active = result.present[:, placement.track_id]
        assert np.all(composed_np[active, placement.track_id] == placement.track_id + 1)
        assert np.all(composed_np[~active, placement.track_id] == 0)


def test_composer_rejects_sources_shorter_than_minimum_active_interval() -> None:
    composer = TimelineComposer(
        _config(min_tracks=1, max_tracks=1, max_concurrent=1)
    )
    with pytest.raises(ValueError, match="min_active_frames"):
        composer.compose(["short"], [7])


def test_composer_subclips_many_long_sources_to_keep_timeline_feasible() -> None:
    composer = TimelineComposer(
        TimelineConfig(
            num_frames=1024,
            min_tracks=10,
            max_tracks=10,
            max_concurrent=4,
            min_reuse_gap_frames=4,
            start_index_range=(-128, 992),
            min_active_frames=32,
            overlap_probability=0.3,
            min_gap_frames=8,
            max_gap_frames=256,
        ),
        rng=random.Random(34),
    )

    result = composer.compose(
        [f"long_{index}" for index in range(10)],
        [742] * 10,
    )

    assert result.present.sum(1).max() <= 4
    assert all(placement.num_active_frames >= 32 for placement in result.placements)
    assert any(placement.source_start > 0 for placement in result.placements)
