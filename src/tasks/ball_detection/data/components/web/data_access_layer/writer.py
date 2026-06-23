"""Write normalized web ball-detection frames to the unified store."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.tasks.ball_detection.data.components.web.data_access_layer.web_store import (
    INDEX_FILE,
    INSTANCE_FIELDS,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    MANIFEST_FILE,
    SAMPLE_FIELDS,
    SCHEMA_VERSION,
    SPLIT_CODES,
    STORE_FILE,
    STORE_SHARD,
    STRINGS_FILE,
    shard_name,
)
from src.utils.io import ensure_dir, save_json


@dataclass
class WebFrameRecord:
    """One normalized, explicitly labeled frame ready for persistence."""

    instances: list[tuple[float, float, int]]
    orig_w: int
    orig_h: int
    temporal: int
    source: str
    sequence: str
    frame_index: int
    split: str
    jpeg: bytes | None = None
    file_path: Path | None = None


class ShardWriter:
    """Append JPEG bytes to rotating shard files."""

    def __init__(self, shards_dir: Path, shard_size_bytes: int) -> None:
        self.shards_dir = shards_dir
        self.shard_size_bytes = shard_size_bytes
        ensure_dir(self.shards_dir)
        self._shard_id = 0
        self._offset = 0
        self._handle = (self.shards_dir / shard_name(0)).open("wb")
        self.shard_count = 1
        self.total_bytes = 0

    def write(self, data: bytes) -> tuple[int, int, int]:
        """Write bytes and return ``(shard_id, offset, length)``."""
        if self._offset > 0 and self._offset + len(data) > self.shard_size_bytes:
            self._handle.close()
            self._shard_id += 1
            self._offset = 0
            self._handle = (self.shards_dir / shard_name(self._shard_id)).open("wb")
            self.shard_count = self._shard_id + 1
        offset = self._offset
        self._handle.write(data)
        self._offset += len(data)
        self.total_bytes += len(data)
        return self._shard_id, offset, len(data)

    def close(self) -> None:
        """Close the active shard."""
        self._handle.close()


class _Interner:
    """Map strings to stable integer ids."""

    def __init__(self) -> None:
        self.values: list[str] = []
        self._lookup: dict[str, int] = {}

    def intern(self, value: str) -> int:
        existing = self._lookup.get(value)
        if existing is not None:
            return existing
        index = len(self.values)
        self.values.append(value)
        self._lookup[value] = index
        return index


class IndexBuilder:
    """Accumulate per-sample and per-instance store columns."""

    def __init__(self) -> None:
        self.samples: dict[str, list[int]] = {name: [] for name in SAMPLE_FIELDS}
        self.instances: dict[str, list[float]] = {name: [] for name in INSTANCE_FIELDS}
        self.sources = _Interner()
        self.sequences = _Interner()
        self.paths = _Interner()
        self.split_counts = {
            name: {"total": 0, "positive": 0, "negative": 0} for name in SPLIT_CODES
        }
        self.source_counts: dict[str, dict[str, int]] = {}
        self.source_sequences: dict[str, set[str]] = {}
        self.sequence_splits: dict[str, str] = {}
        self.sequence_sources: dict[str, str] = {}
        self.temporal_count = 0
        self.positive_count = 0
        self.negative_count = 0

    def add(self, record: WebFrameRecord, writer: ShardWriter) -> None:
        """Persist one normalized frame into the index and shard writer."""
        existing_split = self.sequence_splits.setdefault(
            record.sequence,
            record.split,
        )
        if existing_split != record.split:
            raise ValueError(
                f"Sequence {record.sequence!r} spans splits "
                f"{existing_split!r} and {record.split!r}."
            )
        existing_source = self.sequence_sources.setdefault(
            record.sequence,
            record.source,
        )
        if existing_source != record.source:
            raise ValueError(
                f"Sequence {record.sequence!r} spans sources "
                f"{existing_source!r} and {record.source!r}."
            )

        if record.jpeg is not None:
            shard_id, offset, length = writer.write(record.jpeg)
            store, path_id = STORE_SHARD, -1
        elif record.file_path is not None:
            shard_id, offset, length = -1, 0, 0
            store = STORE_FILE
            relative_path = os.path.relpath(
                record.file_path.resolve(),
                writer.shards_dir.parent,
            )
            path_id = self.paths.intern(relative_path)
        else:
            raise ValueError("WebFrameRecord needs jpeg bytes or a file_path.")

        instance_start = len(self.instances["inst_x"])
        for x, y, visibility in record.instances:
            self.instances["inst_x"].append(float(x))
            self.instances["inst_y"].append(float(y))
            self.instances["inst_vis"].append(int(visibility))

        sample = self.samples
        sample["store"].append(store)
        sample["shard"].append(shard_id)
        sample["offset"].append(offset)
        sample["length"].append(length)
        sample["path_id"].append(path_id)
        sample["orig_w"].append(record.orig_w)
        sample["orig_h"].append(record.orig_h)
        sample["temporal"].append(record.temporal)
        sample["split"].append(SPLIT_CODES[record.split])
        sample["source_id"].append(self.sources.intern(record.source))
        sample["sequence_id"].append(self.sequences.intern(record.sequence))
        sample["frame_index"].append(record.frame_index)
        label_state = LABEL_POSITIVE if record.instances else LABEL_NEGATIVE
        sample["label_state"].append(label_state)
        sample["inst_start"].append(instance_start)
        sample["inst_count"].append(len(record.instances))

        label_name = "positive" if label_state == LABEL_POSITIVE else "negative"
        self.split_counts[record.split]["total"] += 1
        self.split_counts[record.split][label_name] += 1
        source_counts = self.source_counts.setdefault(
            record.source,
            {"total": 0, "positive": 0, "negative": 0, "sequences": 0},
        )
        source_counts["total"] += 1
        source_counts[label_name] += 1
        source_sequences = self.source_sequences.setdefault(record.source, set())
        source_sequences.add(record.sequence)
        source_counts["sequences"] = len(source_sequences)
        self.temporal_count += int(record.temporal)
        self.positive_count += int(label_state == LABEL_POSITIVE)
        self.negative_count += int(label_state == LABEL_NEGATIVE)

    def __len__(self) -> int:
        return len(self.samples["store"])

    def save(self, output_dir: Path) -> None:
        """Write numeric and string index files."""
        arrays: dict[str, np.ndarray] = {
            name: np.asarray(self.samples[name], dtype=dtype)
            for name, dtype in SAMPLE_FIELDS.items()
        }
        arrays.update(
            {
                name: np.asarray(self.instances[name], dtype=dtype)
                for name, dtype in INSTANCE_FIELDS.items()
            }
        )
        np.savez(output_dir / INDEX_FILE, **arrays)  # type: ignore[arg-type]
        save_json(
            {
                "schema": SCHEMA_VERSION,
                "sources": self.sources.values,
                "sequences": self.sequences.values,
                "paths": self.paths.values,
            },
            output_dir / STRINGS_FILE,
        )


def publish_store(build_dir: Path, output_dir: Path) -> None:
    """Replace the previous store only after conversion completes."""
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    if backup_dir.exists():
        if output_dir.exists():
            shutil.rmtree(backup_dir)
        else:
            backup_dir.rename(output_dir)
    if output_dir.exists():
        output_dir.rename(backup_dir)
    try:
        build_dir.rename(output_dir)
    except BaseException:
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def write_manifest(
    output_dir: Path,
    index: IndexBuilder,
    writer: ShardWriter,
) -> None:
    """Write a human-readable store summary."""
    split_sequence_counts = {
        split: sum(
            assigned_split == split for assigned_split in index.sequence_splits.values()
        )
        for split in SPLIT_CODES
    }
    payload = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_samples": len(index),
        "total_instances": len(index.instances["inst_x"]),
        "label_counts": {
            "positive": index.positive_count,
            "negative": index.negative_count,
            "unknown": 0,
        },
        "temporal_samples": index.temporal_count,
        "non_temporal_samples": len(index) - index.temporal_count,
        "sequence_count": len(index.sequence_splits),
        "split_sequence_counts": split_sequence_counts,
        "split_counts": index.split_counts,
        "source_counts": index.source_counts,
        "sequence_split_leaks": 0,
        "shard_count": writer.shard_count,
        "packed_bytes": writer.total_bytes,
        "referenced_files": len(index.paths.values),
    }
    save_json(payload, output_dir / MANIFEST_FILE)


def write_store_readme(output_dir: Path) -> None:
    """Document the generated store layout."""
    (output_dir / "README.md").write_text(
        "# Unified web ball-detection store\n\n"
        f"Schema: `{SCHEMA_VERSION}` (see "
        "`src/tasks/ball_detection/data/components/web/data_access_layer/"
        "web_store.py`).\n\n"
        "Generated by `python -m "
        "src.tasks.ball_detection.scripts.convert_web_dataset` from "
        "`data/tennis/web`. Positive frames and frames explicitly annotated "
        "as ball-absent are kept; unknown frames are excluded.\n\n"
        "- `shards/shard-*.bin`: packed JPEG bytes for video frames.\n"
        "- COCO still images are referenced in place.\n"
        "- `index.npz` / `index_strings.json`: columnar sample index.\n"
        "- `manifest.json`: human-readable summary.\n\n"
        "Every sample carries source, split-safe sequence, frame index, "
        "temporal, and label-state provenance.\n\n"
        "Load via `data=web_frames` (`WebBallDataModule`).\n",
        encoding="utf-8",
    )


__all__ = [
    "IndexBuilder",
    "ShardWriter",
    "WebFrameRecord",
    "publish_store",
    "write_manifest",
    "write_store_readme",
]
