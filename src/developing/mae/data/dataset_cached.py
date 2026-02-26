"""Datasets for cached-batch MAE training.

Train-time performance is improved by caching fully preprocessed batches to disk.
The dataset reads cached batch tensors with minimal per-item overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.developing.mae.data.cache.manifest import EpochCacheManifest


def _distributed_rank_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


@dataclass(frozen=True)
class CachedBatchSource:
    manifest_path: Optional[Path] = None
    manifest_pointer_path: Optional[Path] = None

    def resolve_manifest_path(self) -> Path:
        if self.manifest_path is not None:
            return self.manifest_path
        if self.manifest_pointer_path is None:
            raise ValueError("Either manifest_path or manifest_pointer_path must be set.")
        text = self.manifest_pointer_path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Empty CURRENT pointer: {self.manifest_pointer_path}")
        p = Path(text)
        if not p.is_absolute():
            p = self.manifest_pointer_path.parent / p
        return p


class CachedBatchIterableDataset(IterableDataset[dict]):
    """IterableDataset yielding already-batched tensors from a cache manifest."""

    def __init__(
        self,
        source: CachedBatchSource,
        *,
        map_location: str = "cpu",
    ) -> None:
        super().__init__()
        self.source = source
        self.map_location = map_location

    def __iter__(self) -> Iterator[dict]:
        manifest_path = self.source.resolve_manifest_path()
        manifest = EpochCacheManifest.load(manifest_path)
        base_dir = manifest_path.parent
        entries = list(manifest.batches)

        rank, world_size = _distributed_rank_world_size()
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker.id)
            num_workers = int(worker.num_workers)

        partitions = world_size * num_workers
        partition_id = rank * num_workers + worker_id

        for idx, entry in enumerate(entries):
            if (idx % partitions) != partition_id:
                continue
            batch_path = base_dir / entry.path
            batch = torch.load(batch_path, map_location=self.map_location)
            if isinstance(batch, dict):
                yield batch
            else:
                yield {"image": batch}

    def __len__(self) -> int:
        manifest_path = self.source.resolve_manifest_path()
        manifest = EpochCacheManifest.load(manifest_path)
        return len(manifest.batches)


if __name__ == "__main__":  # pragma: no cover
    tmp = Path("outputs/tmp_cache/mae_dummy")
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"ready: {tmp}")
