"""Curriculum sampling utilities for WASB datasets.

This module provides a step-aware sampler that can switch sampling strategy
mid-training, e.g. to balance visibility classes after a warmup phase.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import Sampler


class VisibilityCurriculumSampler(Sampler[int]):
    """Sampler that switches to visibility-balanced sampling after a step.

    Notes:
        - The switch is applied when the sampler is iterated. In Lightning this
          typically happens at epoch boundaries (new dataloader iterator).
        - Balancing is implemented via weighted sampling with replacement.
    """

    def __init__(
        self,
        *,
        visibilities: Sequence[int],
        switch_step: int,
        balance_values: Sequence[int] = (1, 2),
        target_ratio: Sequence[float] = (0.5, 0.5),
        replacement: bool = True,
        seed: int = 0,
        num_samples: int | None = None,
    ) -> None:
        if len(balance_values) != len(target_ratio):
            raise ValueError("balance_values and target_ratio must have same length")
        if switch_step < 0:
            raise ValueError("switch_step must be >= 0")
        if any(r < 0 for r in target_ratio):
            raise ValueError("target_ratio must be non-negative")
        ratio_sum = float(sum(target_ratio))
        if ratio_sum <= 0:
            raise ValueError("target_ratio must sum to > 0")

        self._visibilities = list(map(int, visibilities))
        self._switch_step = int(switch_step)
        self._balance_values = list(map(int, balance_values))
        self._target_ratio = [float(r) / ratio_sum for r in target_ratio]
        self._replacement = bool(replacement)
        self._seed = int(seed)

        self._num_samples = (
            int(num_samples) if num_samples is not None else len(self._visibilities)
        )
        if self._num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        self._epoch = 0
        self._step = 0
        self._balanced_weights = self._compute_balanced_weights()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def using_balanced_sampling(self) -> bool:
        return self._step >= self._switch_step and self._balanced_weights is not None

    def _compute_balanced_weights(self) -> torch.Tensor | None:
        counts = Counter(self._visibilities)
        missing = [v for v in self._balance_values if counts.get(v, 0) <= 0]
        if missing:
            return None

        per_class_weight: dict[int, float] = {}
        for v, r in zip(self._balance_values, self._target_ratio):
            per_class_weight[v] = r / float(counts[v])

        weights = torch.tensor(
            [per_class_weight.get(v, 0.0) for v in self._visibilities],
            dtype=torch.double,
        )
        if float(weights.sum().item()) <= 0.0:
            return None
        return weights

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self._seed + self._epoch)

        n = len(self._visibilities)
        if not self.using_balanced_sampling():
            # Warmup phase: behave like a standard shuffle.
            if self._num_samples <= n:
                indices = torch.randperm(n, generator=generator)[: self._num_samples]
                yield from indices.tolist()
            else:
                remaining = self._num_samples
                while remaining > 0:
                    perm = torch.randperm(n, generator=generator)
                    take = min(remaining, n)
                    yield from perm[:take].tolist()
                    remaining -= take
            return

        assert self._balanced_weights is not None
        sampled = torch.multinomial(
            self._balanced_weights,
            num_samples=self._num_samples,
            replacement=self._replacement,
            generator=generator,
        )
        yield from sampled.tolist()

    def __len__(self) -> int:
        return self._num_samples


class CurriculumStepCallback(pl.Callback):
    """Updates a step-aware sampler with Lightning's step/epoch counters."""

    def __init__(self, sampler: VisibilityCurriculumSampler) -> None:
        super().__init__()
        self._sampler = sampler

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        self._sampler.set_epoch(trainer.current_epoch)
        self._sampler.set_step(trainer.global_step)
