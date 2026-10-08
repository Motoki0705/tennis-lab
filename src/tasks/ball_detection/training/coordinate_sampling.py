"""Deterministic equal-FPS mixing; each epoch has an explicit window budget."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

from src.tasks.ball_detection.data.coordinate_dataset import CoordinateWindowDataset


class FPSMixSampler(Sampler[int]):
    def __init__(self, dataset: CoordinateWindowDataset, *, windows_per_epoch: int, seed: int) -> None:
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
