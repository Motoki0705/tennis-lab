"""Scene-aware batch samplers shared across tasks."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import torch
from torch.utils.data import BatchSampler, Dataset, Subset


def _build_scene_index_map(dataset: Dataset) -> dict[int, list[int]]:
    """Build a map from scene ID to sample indices.

    For :class:`Subset`, maps from base-dataset indices.  For datasets
    exposing ``scene_id_for_sample(idx)``, that protocol is used.
    Otherwise each sample index is treated as its own scene (identity
    mapping), which is correct for :class:`NPZSceneDatasetBase` where
    ``__len__ == len(scenes)``.

    Args:
        dataset: Dataset or Subset.

    Returns:
        Mapping from scene ID to list of sample indices.
    """

    scene_to_indices: dict[int, list[int]] = defaultdict(list)

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        subset_indices = list(dataset.indices)
        if hasattr(base_dataset, "scene_id_for_sample"):
            for subset_pos, base_idx in enumerate(subset_indices):
                scene_idx = int(base_dataset.scene_id_for_sample(int(base_idx)))
                scene_to_indices[scene_idx].append(subset_pos)
        else:
            # NPZSceneDatasetBase: scene_id == sample index (identity)
            for subset_pos, base_idx in enumerate(subset_indices):
                scene_to_indices[int(base_idx)].append(subset_pos)
        return scene_to_indices

    if hasattr(dataset, "scene_id_for_sample"):
        for idx in range(len(dataset)):
            scene_idx = int(dataset.scene_id_for_sample(idx))
            scene_to_indices[scene_idx].append(idx)
        return scene_to_indices

    # Default: identity mapping (each sample is its own "scene")
    return {i: [i] for i in range(len(dataset))}


def _shuffle_list(values: list[int], generator: torch.Generator | None) -> list[int]:
    """Shuffle a list with optional torch generator.

    Args:
        values: List of values to shuffle.
        generator: Optional torch generator for deterministic shuffling.

    Returns:
        Shuffled list.
    """

    values = list(values)
    if generator is None:
        random.shuffle(values)
        return values

    perm = torch.randperm(len(values), generator=generator).tolist()
    return [values[i] for i in perm]


def resolve_scene_sampler_mode(data_cfg: dict) -> str:
    """Resolve sampler mode from config values.

    Args:
        data_cfg: Data configuration dictionary.

    Returns:
        Sampler mode string.
    """

    mode = str(data_cfg.get("scene_sampler_mode", "")).strip().lower()
    if mode:
        return mode
    if bool(data_cfg.get("chunked_scene_sampler", False)):
        return "chunked"
    if bool(data_cfg.get("scene_batch_sampler", False)):
        scenes_per_batch = int(data_cfg.get("scenes_per_batch", 1))
        return "mixed" if scenes_per_batch > 1 else "scene"
    return "none"


def build_scene_sampler(
    dataset: Dataset,
    batch_size: int,
    mode: str,
    scenes_per_batch: int,
    chunk_max_scenes: int,
    drop_last: bool,
    shuffle: bool,
) -> BatchSampler | None:
    """Build a scene-aware sampler for the dataloader.

    Args:
        dataset: Dataset or Subset.
        batch_size: Samples per batch.
        mode: Sampler mode ("scene", "mixed", "chunked", or "none").
        scenes_per_batch: Number of scenes per batch for mixed mode.
        chunk_max_scenes: Maximum scenes per chunk for chunked mode.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle samples.

    Returns:
        BatchSampler instance or None.
    """

    if mode == "chunked":
        return ChunkedSceneBatchSampler(
            dataset,
            batch_size=batch_size,
            chunk_max_scenes=chunk_max_scenes,
            drop_last=drop_last,
            shuffle=shuffle,
        )
    if mode == "mixed" and scenes_per_batch > 1:
        return MixedSceneBatchSampler(
            dataset,
            batch_size=batch_size,
            scenes_per_batch=scenes_per_batch,
            drop_last=drop_last,
            shuffle=shuffle,
        )
    if mode == "scene":
        return SceneBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
        )
    return None


class SceneBatchSampler(BatchSampler):
    """Batch sampler that groups samples by scene ID.

    Args:
        dataset: Dataset or Subset.
        batch_size: Number of samples per batch.
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle scenes and samples within a scene.
        generator: Optional torch generator for deterministic shuffling.
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

        self._scene_to_indices = _build_scene_index_map(dataset)

    def __iter__(self) -> Iterable[list[int]]:
        scene_ids = list(self._scene_to_indices.keys())
        if self.shuffle:
            scene_ids = _shuffle_list(scene_ids, self.generator)

        for scene_id in scene_ids:
            indices = list(self._scene_to_indices[scene_id])
            if self.shuffle:
                indices = _shuffle_list(indices, self.generator)

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


class MixedSceneBatchSampler(BatchSampler):
    """Batch sampler that mixes a fixed number of scenes per batch.

    Args:
        dataset: Dataset or Subset.
        batch_size: Number of samples per batch.
        scenes_per_batch: Number of distinct scenes per batch.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle scenes and samples within scenes.
        generator: Optional torch generator for deterministic shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        scenes_per_batch: int,
        drop_last: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if scenes_per_batch <= 0:
            raise ValueError("scenes_per_batch must be positive")
        if batch_size % scenes_per_batch != 0:
            raise ValueError("batch_size must be divisible by scenes_per_batch")

        self.dataset = dataset
        self.batch_size = batch_size
        self.scenes_per_batch = scenes_per_batch
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator
        self.chunk_size = batch_size // scenes_per_batch

        self._scene_to_indices = _build_scene_index_map(dataset)

    def __iter__(self) -> Iterable[list[int]]:
        scene_ids = list(self._scene_to_indices.keys())
        if self.shuffle:
            scene_ids = _shuffle_list(scene_ids, self.generator)

        scene_to_chunks: dict[int, list[list[int]]] = {}
        for scene_id in scene_ids:
            indices = list(self._scene_to_indices[scene_id])
            if self.shuffle:
                indices = _shuffle_list(indices, self.generator)

            chunks = [
                indices[start : start + self.chunk_size]
                for start in range(0, len(indices), self.chunk_size)
            ]
            if self.drop_last:
                chunks = [chunk for chunk in chunks if len(chunk) == self.chunk_size]
            scene_to_chunks[scene_id] = chunks

        available = [scene_id for scene_id, chunks in scene_to_chunks.items() if chunks]
        while len(available) >= self.scenes_per_batch:
            if self.shuffle:
                selected = _shuffle_list(available, self.generator)[: self.scenes_per_batch]
            else:
                selected = available[: self.scenes_per_batch]

            batch: list[int] = []
            for scene_id in selected:
                batch.extend(scene_to_chunks[scene_id].pop(0))
                if not scene_to_chunks[scene_id]:
                    available.remove(scene_id)

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last:
                continue

    def __len__(self) -> int:
        total_chunks = 0
        for indices in self._scene_to_indices.values():
            num = len(indices)
            if self.drop_last:
                total_chunks += num // self.chunk_size
            else:
                total_chunks += (num + self.chunk_size - 1) // self.chunk_size
        return total_chunks // self.scenes_per_batch


class ChunkedSceneBatchSampler(BatchSampler):
    """Batch sampler that limits sampling to scene chunks.

    Args:
        dataset: Dataset or Subset.
        batch_size: Number of samples per batch.
        chunk_max_scenes: Maximum number of scenes in a chunk.
        drop_last: Whether to drop incomplete batches.
        shuffle: Whether to shuffle scenes and samples within a chunk.
        generator: Optional torch generator for deterministic shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        chunk_max_scenes: int,
        drop_last: bool = True,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunk_max_scenes <= 0:
            raise ValueError("chunk_max_scenes must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_max_scenes = int(chunk_max_scenes)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

        self._scene_to_indices = _build_scene_index_map(dataset)

    def __iter__(self) -> Iterable[list[int]]:
        scene_ids = list(self._scene_to_indices.keys())
        if self.shuffle:
            scene_ids = _shuffle_list(scene_ids, self.generator)

        for start in range(0, len(scene_ids), self.chunk_max_scenes):
            chunk_scene_ids = scene_ids[start : start + self.chunk_max_scenes]
            chunk_indices: list[int] = []
            for scene_id in chunk_scene_ids:
                indices = list(self._scene_to_indices[scene_id])
                if self.shuffle:
                    indices = _shuffle_list(indices, self.generator)
                chunk_indices.extend(indices)

            if self.shuffle:
                chunk_indices = _shuffle_list(chunk_indices, self.generator)

            for offset in range(0, len(chunk_indices), self.batch_size):
                batch = chunk_indices[offset : offset + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        scene_ids = list(self._scene_to_indices.keys())
        total = 0
        for start in range(0, len(scene_ids), self.chunk_max_scenes):
            chunk_scene_ids = scene_ids[start : start + self.chunk_max_scenes]
            chunk_count = sum(
                len(self._scene_to_indices[scene_id]) for scene_id in chunk_scene_ids
            )
            if self.drop_last:
                total += chunk_count // self.batch_size
            else:
                total += (chunk_count + self.batch_size - 1) // self.batch_size
        return total


if __name__ == "__main__":
    class _DummyDataset(Dataset[int]):
        def __init__(self) -> None:
            self.index = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, idx: int) -> int:
            return idx

    dummy = _DummyDataset()
    sampler = SceneBatchSampler(dummy, batch_size=2, drop_last=False, shuffle=False)
    assert list(sampler) == [[0, 1], [2], [3, 4], [5]]

    chunked = ChunkedSceneBatchSampler(dummy, batch_size=2, chunk_max_scenes=2, drop_last=False, shuffle=False)
    batches = list(chunked)
    assert batches
    print("common.data.scene_batch_sampler smoke ok")
