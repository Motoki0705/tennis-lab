"""Manifest reading/writing for the DINOv3 SSL image dataset."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.tasks.dino_ssl.generate_dataset.collectors import CollectedImage

MANIFEST_NAME = "meta.json"


@dataclass(frozen=True)
class ManifestRecord:
    """One image entry in the manifest."""

    path: str
    source_type: str
    provenance: str


@dataclass
class DatasetManifest:
    """A collected SSL dataset described relative to its root directory."""

    root: Path
    records: list[ManifestRecord]

    @property
    def num_images(self) -> int:
        return len(self.records)

    def image_paths(self) -> list[Path]:
        return [self.root / record.path for record in self.records]


def write_manifest(
    *, root: Path, images: list[CollectedImage], extra: dict | None = None
) -> DatasetManifest:
    """Write ``meta.json`` describing ``images`` and return the manifest."""
    root = Path(root)
    records = [
        ManifestRecord(
            path=str(item.path.relative_to(root)),
            source_type=item.source_type,
            provenance=item.provenance,
        )
        for item in images
    ]
    source_counts = Counter(record.source_type for record in records)
    payload = {
        "num_images": len(records),
        "source_counts": dict(source_counts),
        "images": [record.__dict__ for record in records],
    }
    if extra:
        payload.update(extra)

    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return DatasetManifest(root=root, records=records)


def read_manifest(root: Path) -> DatasetManifest:
    """Load a previously written manifest from ``root/meta.json``."""
    root = Path(root)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DINOv3 SSL manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        ManifestRecord(
            path=str(entry["path"]),
            source_type=str(entry.get("source_type", "unknown")),
            provenance=str(entry.get("provenance", "")),
        )
        for entry in payload.get("images", [])
    ]
    return DatasetManifest(root=root, records=records)


__all__ = [
    "MANIFEST_NAME",
    "ManifestRecord",
    "DatasetManifest",
    "write_manifest",
    "read_manifest",
]
