"""Unit tests for deterministic exact-ratio source mixing."""

from __future__ import annotations

import pytest

from src.tasks.ball_detection.data.components.source_mix_sampler import (
    ExactSourceMixBatchSampler,
)


def _sampler(
    *,
    synthetic_per_batch: int = 1,
    synthetic_batch_period: int = 1,
) -> ExactSourceMixBatchSampler:
    return ExactSourceMixBatchSampler(
        real_size=20,
        synthetic_size=10,
        batch_size=6,
        synthetic_per_batch=synthetic_per_batch,
        synthetic_batch_period=synthetic_batch_period,
        steps_per_epoch=4,
        seed=731,
    )


def test_each_batch_has_exact_source_ratio() -> None:
    batches = list(_sampler())

    assert len(batches) == 4
    assert all(len(batch) == 6 for batch in batches)
    assert all(sum(index >= 20 for index in batch) == 1 for batch in batches)


def test_control_batches_are_real_only() -> None:
    batches = list(_sampler(synthetic_per_batch=0))

    assert all(all(index < 20 for index in batch) for batch in batches)


def test_periodic_mix_rotates_phase_between_epochs() -> None:
    sampler = _sampler(synthetic_batch_period=2)

    first_counts = [
        sum(index >= 20 for index in batch) for batch in list(sampler)
    ]
    second_counts = [
        sum(index >= 20 for index in batch) for batch in list(sampler)
    ]

    assert first_counts == [1, 0, 1, 0]
    assert second_counts == [0, 1, 0, 1]


def test_seed_is_reproducible_and_epochs_change() -> None:
    first_sampler = _sampler()
    second_sampler = _sampler()

    first_epoch = list(first_sampler)
    assert first_epoch == list(second_sampler)
    assert first_epoch != list(first_sampler)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"real_size": 0}, "real_size"),
        ({"synthetic_size": 0}, "synthetic_size"),
        ({"synthetic_per_batch": 6}, "synthetic_per_batch"),
        ({"synthetic_batch_period": 0}, "synthetic_batch_period"),
        ({"steps_per_epoch": 0}, "steps_per_epoch"),
    ],
)
def test_invalid_plan_fails_explicitly(
    kwargs: dict[str, int],
    match: str,
) -> None:
    config = {
        "real_size": 20,
        "synthetic_size": 10,
        "batch_size": 6,
        "synthetic_per_batch": 1,
        "synthetic_batch_period": 1,
        "steps_per_epoch": 4,
        "seed": 731,
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ExactSourceMixBatchSampler(**config)
