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


def config(**overrides):
    return TimelineConfig(
        **dict(
            min_tracks=4,
            max_tracks=10,
            max_concurrent=4,
            min_reuse_gap_frames=4,
            min_scene_frames=1,
            planning_iterations=80,
        )
        | overrides
    )


def test_complete_sources_and_variable_length_with_first_birth_at_zero():
    lengths = [742] * 10
    plan = TimelineComposer(config(), rng=random.Random(34)).compose(
        [str(i) for i in range(10)], lengths
    )
    assert len(plan.present) > 1024
    assert plan.present[0].any()
    assert len(plan.present) == max(p.death_frame for p in plan.placements)
    assert plan.present.sum(1).max() <= 4
    reserved: NDArray[np.int64] = np.zeros(len(plan.present) + 4, dtype=int)
    sources: list[NDArray[np.float32]] = [
        np.arange(n * 3, dtype=np.float32).reshape(n, 3) + i
        for i, n in enumerate(lengths)
    ]
    actual = plan.compose_numpy(sources)
    torch.testing.assert_close(
        plan.compose_tensor([torch.from_numpy(s) for s in sources]),
        torch.from_numpy(actual),
    )
    for p, source in zip(plan.placements, sources, strict=True):
        assert p.source_start == 0 and p.source_end == len(source)
        assert p.death_frame - p.birth_frame == len(source)
        np.testing.assert_array_equal(
            actual[p.birth_frame : p.death_frame, p.track_id], source
        )
        assert not actual[~plan.present[:, p.track_id], p.track_id].any()
        reserved[p.birth_frame : p.death_frame + 4] += 1
    assert reserved.max() <= 4


def test_short_sources_are_not_arbitrarily_dropped_or_extended():
    plan = TimelineComposer(
        config(min_tracks=1, max_tracks=1, max_concurrent=1)
    ).compose(["short"], [7])
    assert plan.present.shape == (7, 1)
    assert plan.placements[0].source_end == 7
    with pytest.raises(ValueError, match="positive"):
        TimelineComposer(config(min_tracks=1)).compose(["x"] * 4, [7, 3, 1, 0])


def test_birth_optimization_is_repeatable_and_balances_aggregate_occupancy():
    counts = np.zeros(5)
    for seed in range(8):
        lengths = [400 + 100 * ((seed + i) % 5) for i in range(6)]
        a = TimelineComposer(config(), rng=random.Random(seed)).compose(
            [str(i) for i in range(6)], lengths
        )
        b = TimelineComposer(config(), rng=random.Random(seed)).compose(
            [str(i) for i in range(6)], lengths
        )
        assert a.placements == b.placements
        counts += np.bincount(a.present.sum(1), minlength=5)
    np.testing.assert_allclose(counts[1:] / counts[1:].sum(), 0.25, atol=0.04, rtol=0)


def test_legacy_truncation_knobs_are_rejected():
    with pytest.raises(ValueError, match="exactly"):
        TimelineConfig.from_mapping({"min_active_frames": 32})


def test_dataset_ledger_balances_unequal_lengths_and_rates():
    composer = TimelineComposer(config(planning_iterations=120), rng=random.Random(5))
    sources = [[683, 281, 296, 1039], [400] * 8, [981, 242, 265, 379], [600] * 8]
    for index, lengths in enumerate(sources):
        composer.rng.seed(index)
        composer.compose(
            [str(i) for i in range(len(lengths))],
            lengths,
            fps=30.0 if index % 2 else 120.0,
            balance_dataset=True,
        )
    seconds = composer.occupancy_seconds[1:]
    np.testing.assert_allclose(seconds / seconds.sum(), 0.25, atol=0.025, rtol=0)
