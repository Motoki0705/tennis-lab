"""Scene-aware batch sampler for PLCS datasets.

This sampler ensures that each batch contains samples from a single scene,
which improves cache locality when loading scene files.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import torch
from torch.utils.data import BatchSampler, Dataset, Subset


class SceneBatchSampler(BatchSampler):
    """Batch sampler that groups samples by scene index.

    Args:
        dataset: Dataset or Subset with an "index" attribute whose first
            element is scene_idx.
        batch_size: Number of samples per batch.
        drop_last: Whether to drop the last incomplete batch per scene.
        shuffle: Whether to shuffle scenes and samples within each scene.
        generator: Optional torch Generator for deterministic shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

        self._scene_to_indices = self._build_scene_index_map(dataset)

    def __iter__(self) -> Iterable[list[int]]:
        scene_ids = list(self._scene_to_indices.keys())
        if self.shuffle:
            scene_ids = self._shuffle_list(scene_ids)

        for scene_id in scene_ids:
            indices = list(self._scene_to_indices[scene_id])
            if self.shuffle:
                indices = self._shuffle_list(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self._scene_to_indices.values():
            num = len(indices)
            if self.drop_last:
                total += num // self.batch_size
            else:
                total += (num + self.batch_size - 1) // self.batch_size
        return total

    def _build_scene_index_map(self, dataset: Dataset) -> dict[int, list[int]]:
        scene_to_indices: dict[int, list[int]] = defaultdict(list)

        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = list(dataset.indices)
            for subset_pos, base_idx in enumerate(subset_indices):
                scene_idx = self._extract_scene_idx(base_dataset, base_idx)
                scene_to_indices[scene_idx].append(subset_pos)
        else:
            for idx in range(len(dataset)):
                scene_idx = self._extract_scene_idx(dataset, idx)
                scene_to_indices[scene_idx].append(idx)

        return scene_to_indices

    @staticmethod
    def _extract_scene_idx(dataset: Dataset, idx: int) -> int:
        if hasattr(dataset, "index"):
            scene_idx = dataset.index[idx][0]
            return int(scene_idx)
        raise AttributeError("Dataset must expose an 'index' attribute")

    def _shuffle_list(self, values: list[int]) -> list[int]:
        if self.generator is None:
            values = list(values)
            random.shuffle(values)
            return values

        perm = torch.randperm(len(values), generator=self.generator).tolist()
        return [values[i] for i in perm]


if __name__ == "__main__":
    class _DummyDataset(Dataset[torch.Tensor]):
        def __init__(self) -> None:
            self.index = [
                (0, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
            ]

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, idx: int) -> torch.Tensor:
            return torch.tensor(self.index[idx], dtype=torch.int64)

    dummy = _DummyDataset()
    sampler = SceneBatchSampler(dummy, batch_size=2, drop_last=False, shuffle=False)
    batches = list(sampler)
    for batch in batches:
        scene_ids = {dummy.index[i][0] for i in batch}
        assert len(scene_ids) == 1
    print("SceneBatchSampler smoke test passed.")
