"""Deterministic per-epoch source mixing for the training loader."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler


def allocate_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split ``total`` draws over sources proportionally (largest remainder)."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not weights or any(
        not np.isfinite(weight) or weight <= 0 for weight in weights.values()
    ):
        raise ValueError("Every source weight must be positive")
    weight_sum = float(sum(weights.values()))
    if not np.isfinite(weight_sum):
        raise ValueError("Sum of source weights must be finite")
    exact = {name: total * weight / weight_sum for name, weight in weights.items()}
    counts = {name: int(np.floor(value)) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    by_fraction = sorted(exact, key=lambda name: (counts[name] - exact[name], name))
    for name in by_fraction[:remainder]:
        counts[name] += 1
    return counts


class SourceMixSampler(Sampler[int]):
    """Draw ``samples_per_epoch`` dataset indices with fixed source proportions.

    ``source_indices`` maps a source name to its dataset indices. Each epoch
    draws exactly ``allocate_counts(samples_per_epoch, weights)`` indices per
    source, walking shuffled full passes of that source so every index is used
    before any is repeated, then shuffles the union. Epoch ``e`` is a pure
    function of ``seed + e``.
    """

    def __init__(
        self,
        *,
        source_indices: Mapping[str, Sequence[int]],
        weights: Mapping[str, float],
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        if set(source_indices) != set(weights):
            raise ValueError(
                f"Source weights {sorted(weights)} must name exactly the sources "
                f"with samples {sorted(source_indices)}"
            )
        empty = sorted(
            name for name, indices in source_indices.items() if len(indices) == 0
        )
        if empty:
            raise ValueError(f"Weighted source(s) without samples: {empty}")
        self.names = tuple(sorted(source_indices))
        self.source_indices = {
            name: np.asarray(source_indices[name], dtype=np.int64)
            for name in self.names
        }
        self.counts = allocate_counts(
            samples_per_epoch, {name: float(weights[name]) for name in self.names}
        )
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def epoch_indices(self, epoch: int) -> list[int]:
        rng = np.random.default_rng(self.seed + epoch)
        drawn: list[np.ndarray] = []
        for name in self.names:
            indices = self.source_indices[name]
            count = self.counts[name]
            if count == 0:
                continue
            passes = -(-count // indices.size)
            order = np.concatenate([rng.permutation(indices) for _ in range(passes)])
            drawn.append(order[:count])
        merged = np.concatenate(drawn)
        return [int(value) for value in rng.permutation(merged)]

    def __iter__(self) -> Iterator[int]:
        epoch = self._epoch
        self._epoch += 1
        yield from self.epoch_indices(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch


__all__ = ["SourceMixSampler", "allocate_counts"]
