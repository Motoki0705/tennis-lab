"""Variable-T sampling for the staged multi-frame training schedule (issue #579).

The multi-frame phases draw a clip length ``T`` per optimizer-step *group*:
``P(T=1) = 0.5`` and the remaining ``0.5`` is spread over ``T in [2, T_max]`` with
a linearly decreasing weight (largest at ``T=2``). ``T`` is uniform inside a
batch. To bound VRAM, each ``T`` uses a precomputed physical batch size
``B(T)`` (roughly ``B(T)*T`` constant). The effective batch size is held constant
at ``EBS`` by emitting ``accumulate(T) = round(EBS / B(T))`` consecutive
micro-batches of size ``B(T)`` per group; the lightning module accumulates
gradients over the group and then steps once.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator, Mapping, Sequence, Sized
from typing import cast

import numpy as np
from torch.utils.data import Dataset, Sampler

from src.tasks.ball_detection.data.types import BallDetectionSample


def linear_decreasing_t_probs(t_max: int, t1_prob: float = 0.5) -> dict[int, float]:
    """Return ``P(T)`` for ``T in [1, t_max]``.

    ``T=1`` gets ``t1_prob``; the remaining mass is spread over ``[2, t_max]``
    with weights linearly decreasing from ``T=2`` (weight ``t_max-1``) to
    ``T=t_max`` (weight ``1``), so the tail sums to ``1 - t1_prob``.
    """
    if t_max < 1:
        raise ValueError("t_max must be >= 1.")
    if not 0.0 < t1_prob <= 1.0:
        raise ValueError("t1_prob must be in (0, 1].")
    probs = {1: float(t1_prob)}
    if t_max == 1:
        return {1: 1.0}
    weights = {t: float(t_max - t + 1) for t in range(2, t_max + 1)}
    weight_sum = sum(weights.values())
    tail_mass = 1.0 - t1_prob
    for t, weight in weights.items():
        probs[t] = tail_mass * weight / weight_sum
    return probs


def accumulation_for(effective_batch: int, physical_batch: int) -> int:
    """Number of micro-batches whose union approximates ``effective_batch``."""
    return max(1, round(effective_batch / max(physical_batch, 1)))


class VariableTBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yield ``(window_index, T)`` micro-batches grouped per optimizer step.

    Each yielded item is a list of ``(index, T)`` tuples of length ``B(T)`` with a
    single shared ``T``. Consecutive yields belonging to one optimizer-step group
    share ``T`` and number ``accumulate(T)``; the module steps after each group.
    Incomplete trailing groups are dropped so accumulation stays aligned.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        t_probs: Mapping[int, float],
        batch_size_by_t: Mapping[int, int],
        effective_batch: int,
        seed: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.num_samples = int(num_samples)
        self.t_values = np.array(sorted(t_probs), dtype=np.int64)
        self.t_weights = np.array(
            [t_probs[int(t)] for t in self.t_values], dtype=np.float64
        )
        self.t_weights = self.t_weights / self.t_weights.sum()
        missing = [int(t) for t in self.t_values if int(t) not in batch_size_by_t]
        if missing:
            raise ValueError(f"batch_size_by_t missing entries for T={missing}.")
        self.batch_size_by_t = {int(t): int(b) for t, b in batch_size_by_t.items()}
        self.effective_batch = int(effective_batch)
        self.seed = int(seed)
        self._epoch = 0
        self._planned_epoch: int | None = None
        self._planned_batches: list[list[tuple[int, int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose deterministic batch plan is used next."""
        self._epoch = int(epoch)

    def _build_plan(self, epoch: int) -> list[list[tuple[int, int]]]:
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(self.num_samples)
        batches: list[list[tuple[int, int]]] = []

        cursor = 0
        while cursor < self.num_samples:
            t = int(rng.choice(self.t_values, p=self.t_weights))
            physical = self.batch_size_by_t[t]
            accumulate = accumulation_for(self.effective_batch, physical)
            group_size = physical * accumulate
            if cursor + group_size > self.num_samples:
                break
            for micro in range(accumulate):
                lo = cursor + micro * physical
                chunk = order[lo : lo + physical]
                batches.append([(int(i), t) for i in chunk])
            cursor += group_size

        return batches

    def _plan_for_epoch(self, epoch: int) -> list[list[tuple[int, int]]]:
        if self._planned_epoch != epoch or self._planned_batches is None:
            self._planned_batches = self._build_plan(epoch)
            self._planned_epoch = epoch
        return self._planned_batches

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        # Each epoch re-iterates the DataLoader, so a per-epoch plan varies the
        # permutation and T draws without requiring explicit epoch wiring.
        epoch = self._epoch
        plan = self._plan_for_epoch(epoch)
        try:
            for batch in plan:
                yield list(batch)
        finally:
            if self._epoch == epoch:
                self._epoch = epoch + 1

    def __len__(self) -> int:
        return len(self._plan_for_epoch(self._epoch))


class ConcatVariableTDataset(Dataset[BallDetectionSample]):
    """Concatenate window datasets while forwarding ``(index, T)`` tuples.

    ``torch.utils.data.ConcatDataset`` assumes integer indices, so it cannot
    route the ``(window_index, T)`` tuples emitted by
    :class:`VariableTBatchSampler`. This variant maps the global index to a
    sub-dataset and forwards ``(local_index, T)`` (or a bare int) to it.
    """

    def __init__(self, datasets: Sequence[Dataset[BallDetectionSample]]) -> None:
        if not datasets:
            raise ValueError("ConcatVariableTDataset needs at least one dataset.")
        self.datasets = list(datasets)
        self.cumulative: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(cast(Sized, dataset))
            self.cumulative.append(total)
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int | tuple[int, int]) -> BallDetectionSample:
        if isinstance(index, tuple):
            global_index, num_frames = int(index[0]), int(index[1])
        else:
            global_index, num_frames = int(index), None
        dataset_index = bisect.bisect_right(self.cumulative, global_index)
        offset = self.cumulative[dataset_index - 1] if dataset_index > 0 else 0
        local_index = global_index - offset
        dataset = self.datasets[dataset_index]
        if num_frames is None:
            return dataset[local_index]
        return dataset[(local_index, num_frames)]


class FixedTDataset(Dataset[BallDetectionSample]):
    """Expose a variable-T window dataset at a single fixed ``T`` (val/test)."""

    def __init__(self, base: Dataset[BallDetectionSample], num_frames: int) -> None:
        self.base = base
        self.num_frames = int(num_frames)

    def __len__(self) -> int:
        return len(cast(Sized, self.base))

    def __getitem__(self, index: int) -> BallDetectionSample:
        return self.base[(int(index), self.num_frames)]


__all__ = [
    "ConcatVariableTDataset",
    "FixedTDataset",
    "VariableTBatchSampler",
    "accumulation_for",
    "linear_decreasing_t_probs",
]
