"""Deterministic exact-ratio batch sampling for two supervised sources."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class ExactSourceMixBatchSampler(Sampler[list[int]]):
    """Draw fixed-size batches with a deterministic synthetic schedule.

    Real samples occupy ``[0, real_size)`` in the concatenated dataset and
    synthetic samples occupy the following range. Each epoch uses deterministic
    shuffled cycles, so sources are covered before an index is repeated.
    ``synthetic_batch_period=1`` preserves an exact synthetic count in every
    batch. Larger periods place that count in one of every N batches and rotate
    the scheduled phase across epochs.
    """

    def __init__(
        self,
        *,
        real_size: int,
        synthetic_size: int,
        batch_size: int,
        synthetic_per_batch: int,
        synthetic_batch_period: int = 1,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        if real_size <= 0:
            raise ValueError("real_size must be positive.")
        if synthetic_size < 0:
            raise ValueError("synthetic_size must be non-negative.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0 <= synthetic_per_batch < batch_size:
            raise ValueError(
                "synthetic_per_batch must be in [0, batch_size)."
            )
        if synthetic_per_batch > 0 and synthetic_size == 0:
            raise ValueError(
                "synthetic_size must be positive when synthetic samples are enabled."
            )
        if synthetic_batch_period <= 0:
            raise ValueError("synthetic_batch_period must be positive.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")

        self.real_size = real_size
        self.synthetic_size = synthetic_size
        self.batch_size = batch_size
        self.synthetic_per_batch = synthetic_per_batch
        self.synthetic_batch_period = synthetic_batch_period
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self._epoch = 0

    @staticmethod
    def _shuffled_cycles(
        *,
        size: int,
        count: int,
        generator: torch.Generator,
        offset: int = 0,
    ) -> list[int]:
        indices: list[int] = []
        while len(indices) < count:
            indices.extend(
                (torch.randperm(size, generator=generator) + offset).tolist()
            )
        return indices[:count]

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self._epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        self._epoch += 1

        synthetic_counts = [
            (
                self.synthetic_per_batch
                if (step + epoch) % self.synthetic_batch_period == 0
                else 0
            )
            for step in range(self.steps_per_epoch)
        ]
        real_counts = [
            self.batch_size - synthetic_count
            for synthetic_count in synthetic_counts
        ]
        real_indices = self._shuffled_cycles(
            size=self.real_size,
            count=sum(real_counts),
            generator=generator,
        )
        synthetic_indices = self._shuffled_cycles(
            size=max(self.synthetic_size, 1),
            count=sum(synthetic_counts),
            generator=generator,
            offset=self.real_size,
        )

        real_start = 0
        synthetic_start = 0
        for real_count, synthetic_count in zip(
            real_counts, synthetic_counts, strict=True
        ):
            batch = real_indices[real_start : real_start + real_count]
            batch.extend(
                synthetic_indices[
                    synthetic_start : synthetic_start + synthetic_count
                ]
            )
            real_start += real_count
            synthetic_start += synthetic_count
            order = torch.randperm(self.batch_size, generator=generator).tolist()
            yield [batch[index] for index in order]

    def __len__(self) -> int:
        """Return the fixed optimizer-step count."""
        return self.steps_per_epoch


__all__ = ["ExactSourceMixBatchSampler"]
