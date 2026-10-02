"""Split provenance for evaluation datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

from torch.utils.data import Dataset

from src.tasks.ball_detection.data.store import INDEX_FILE, METADATA_FILE
from src.tasks.ball_detection.data.store_dataset import BallStoreDataset


def build_split_provenance(
    *,
    data_config: Any,
    split: str,
    dataset: Dataset[Any],
) -> dict[str, Any]:
    """Record schema and fixed split identity without reading train data."""
    source = str(data_config.source)
    if isinstance(dataset, BallStoreDataset):
        store = dataset.store
        return {
            "source": source,
            "schema": str(store.metadata["schema_version"]),
            "data_dir": str(store.directory),
            "split": split,
            "store_sources": sorted({dataset.source_of(index) for index in range(len(dataset))}),
            "metadata_sha256": sha256_file(store.directory / METADATA_FILE),
            "index_sha256": sha256_file(store.directory / INDEX_FILE),
            "sample_count": len(cast(Sized, dataset)),
        }
    raise TypeError(f"No split provenance for dataset type {type(dataset).__name__}")


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for provenance and resume checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["build_split_provenance", "sha256_file"]
