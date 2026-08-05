"""Sequential source labels and split provenance for evaluation datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

from torch.utils.data import Dataset

from src.tasks.ball_detection.data.web_datamodule import WebBallDetectionDataset
from src.utils.configuration import PathResolver, PathRole
from src.utils.io import load_json


class SequentialSourceResolver:
    """Recover source names for val/test loaders that use sequential sampling."""

    def __init__(self, dataset: Dataset[Any], *, default_source: str) -> None:
        self.dataset = dataset
        self.default_source = default_source
        self.cursor = 0

    def next(self, batch_size: int) -> list[str]:
        """Return source names for the next sequential batch."""
        start = self.cursor
        stop = start + batch_size
        if stop > len(cast(Sized, self.dataset)):
            raise RuntimeError(
                "Evaluation dataloader yielded more samples than its dataset."
            )
        self.cursor = stop
        if not isinstance(self.dataset, WebBallDetectionDataset):
            return [self.default_source] * batch_size

        sources: list[str] = []
        for window in self.dataset.windows[start:stop]:
            indices = tuple(int(frame_name) for frame_name in window.frame_names)
            window_sources = {
                self.dataset.store.source_name(index) for index in indices
            }
            if len(window_sources) != 1:
                raise RuntimeError(
                    "A web evaluation window spans multiple sources: "
                    f"{sorted(window_sources)}."
                )
            sources.append(next(iter(window_sources)))
        return sources


def build_split_provenance(
    *,
    data_config: Any,
    split: str,
    dataset: Dataset[Any],
    resolver: PathResolver,
) -> dict[str, Any]:
    """Record schema and fixed split identity without reading train data."""
    source = str(data_config.source)
    data_dir = resolver.resolve(PathRole.DATA, str(data_config.data_dir))
    if isinstance(dataset, WebBallDetectionDataset):
        manifest_path = data_dir / "manifest.json"
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise TypeError(f"Web manifest must be a mapping: {manifest_path}")
        return {
            "source": source,
            "schema": str(manifest.get("schema", dataset.store.schema_version)),
            "data_dir": str(data_dir),
            "split": split,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "sample_count": len(dataset),
        }

    split_key = f"{split}_file"
    split_role = PathRole(str(data_config.split.root_role))
    split_path = resolver.resolve(split_role, str(data_config.split[split_key]))
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Configured {split} split file not found: {split_path}"
        )
    return {
        "source": source,
        "schema": "tracknet_label_csv_v1",
        "data_dir": str(data_dir),
        "split": split,
        "split_file": str(split_path),
        "split_sha256": sha256_file(split_path),
        "sample_count": len(cast(Sized, dataset)),
    }


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for provenance and resume checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "SequentialSourceResolver",
    "build_split_provenance",
    "sha256_file",
]
