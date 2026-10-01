"""Exact epoch quotas, coverage, and deterministic source mixing."""

from collections import Counter

import pytest

from src.tasks.ball_detection.data.components.source_mix_sampler import (
    SourceMixSampler,
    allocate_counts,
)


def test_epoch_quotas_and_reproducibility() -> None:
    sampler = SourceMixSampler(
        source_indices={"a": range(10), "b": range(10, 12)},
        weights={"a": 1, "b": 2},
        samples_per_epoch=18,
        seed=4,
    )
    first = list(sampler)
    assert Counter("a" if index < 10 else "b" for index in first) == {"a": 6, "b": 12}
    assert len(set(index for index in first if index < 10)) == 6
    assert Counter(index for index in first if index >= 10) == {10: 6, 11: 6}
    assert first == sampler.epoch_indices(0)
    assert first != list(sampler)
    sampler.set_epoch(0)
    assert first == list(sampler)


def test_rounding_and_zero_quota_are_defined() -> None:
    assert allocate_counts(8, {"a": 1, "b": 1, "c": 1}) == {"a": 3, "b": 3, "c": 2}
    sampler = SourceMixSampler(
        source_indices={"a": [0], "b": [1]},
        weights={"a": 100, "b": 1},
        samples_per_epoch=1,
        seed=3,
    )
    assert list(sampler) == [0]


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_weights_fail(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        allocate_counts(10, {"a": weight})


def test_missing_source_has_no_fallback() -> None:
    with pytest.raises(ValueError, match="without samples"):
        SourceMixSampler(
            source_indices={"a": []}, weights={"a": 1}, samples_per_epoch=8, seed=1
        )
    with pytest.raises(ValueError, match="exactly"):
        SourceMixSampler(
            source_indices={"a": [0]}, weights={"b": 1}, samples_per_epoch=8, seed=1
        )
