"""Deterministic equal-FPS mixing; each epoch has an explicit window budget."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from src.tasks.ball_detection.data.windows.dataset import FrozenBallWindowDataset
from src.tasks.ball_detection.data.windows.hybrid import WINDOW_SOURCES


class FPSMixSampler(Sampler[int]):
    def __init__(self, dataset: FrozenBallWindowDataset, *, windows_per_epoch: int, seed: int) -> None:
        if windows_per_epoch < 1:
            raise ValueError("Epoch window budget must be positive")
        self.steps = dataset.frame_steps
        self.groups = {step: [i for i, (_, window) in enumerate(dataset.windows) if window.frame_step == step]
                       for step in self.steps}
        if any(not indices for indices in self.groups.values()):
            raise ValueError("Every configured FPS must have training windows; do not silently drop an FPS")
        self.count, self.seed, self.epoch = windows_per_epoch, seed, 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[int]:
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        assignments = (torch.arange(self.count) + self.epoch) % len(self.steps)
        assignments = assignments[torch.randperm(self.count, generator=rng)].tolist()
        buckets: dict[int, list[int]] = {}
        for position, step in enumerate(self.steps):
            quota = assignments.count(position)
            available = self.groups[step]
            selected: list[int] = []
            while len(selected) < quota:
                selected.extend(available[i] for i in torch.randperm(len(available), generator=rng).tolist())
            buckets[position] = selected[:quota]
        for position in assignments:
            yield buckets[position].pop()


class HybridWindowSampler(Sampler[int]):
    """Equal FPS budgets, then explicit window-source fractions with largest remainders.

    A positive fraction must get at least one window per FPS per epoch. Empty
    enabled buckets and undersized budgets fail instead of renormalizing ratios.
    """

    def __init__(self, dataset: FrozenBallWindowDataset, *, windows_per_epoch: int, seed: int) -> None:
        if dataset.absence_policy is None or windows_per_epoch < 1:
            raise ValueError("Hybrid sampler requires an explicit absence policy and positive budget")
        self.count, self.seed, self.epoch = windows_per_epoch, seed, 0
        self.steps = dataset.frame_steps
        self.weights = dataset.absence_policy.window_weights
        self.groups = {(step, source): [i for i, (_, window) in enumerate(dataset.windows)
                                       if window.frame_step == step and dataset.window_sources[i] == source]
                       for step in self.steps for source in WINDOW_SOURCES if self.weights[source] > 0}
        if any(not values for values in self.groups.values()):
            missing = [key for key, values in self.groups.items() if not values]
            raise ValueError(f"No training windows for enabled FPS/source buckets: {missing}")
        self.quotas(0)

    def quotas(self, epoch: int) -> dict[tuple[int, str], int]:
        counts = {step: sum((i + epoch) % len(self.steps) == j for i in range(self.count))
                  for j, step in enumerate(self.steps)}
        result = {}
        for step, count in counts.items():
            raw = {s: count * self.weights[s] for s in WINDOW_SOURCES if self.weights[s] > 0}
            quotas = {s: int(v) for s, v in raw.items()}
            order = sorted(raw, key=lambda s: (-(raw[s] - quotas[s]), s))
            for source in order[:count - sum(quotas.values())]:
                quotas[source] += 1
            if any(v == 0 for v in quotas.values()):
                raise ValueError("Window budget too small for all explicitly enabled FPS/source fractions")
            result.update({(step, source): quota for source, quota in quotas.items()})
        return result

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[int]:
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        selected = []
        for key, count in self.quotas(self.epoch).items():
            pool = self.groups[key]
            bucket: list[int] = []
            while len(bucket) < count:
                bucket.extend(pool[i] for i in torch.randperm(len(pool), generator=rng).tolist())
            selected.extend(bucket[:count])
        for i in torch.randperm(len(selected), generator=rng).tolist():
            yield selected[i]
