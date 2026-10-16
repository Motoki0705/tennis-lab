"""Explicit, non-destructive conversion of sparse v1 stores to v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final

import numpy as np

from src.tasks.player_detection.data.store import (
    FRAME_FIELDS,
    INDEX_FILE,
    INSTANCE_FIELDS,
    METADATA_FILE,
    SCHEMA_VERSION,
    SHARDS_DIR,
    PlayerFrameStore,
    _require_columns,
)

LEGACY_SCHEMA_VERSION: Final = "player_detection_frames.v1"
LEGACY_FRAME_FIELDS: Final = {
    name: dtype for name, dtype in FRAME_FIELDS.items() if name != "confirmed_empty"
}


def migrate_v1_store(source: Path, destination: Path) -> Path:
    """Copy a v1 snapshot without adding rows, changing teachers or reencoding JPEGs.

    The destination must be new and outside the source directory. Missing v1
    frames remain unknown: every migrated row has ``confirmed_empty=False``.
    Validate the new v2 store before publishing it, leaving the source untouched.
    """
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("Migration source and destination must be absolute paths")
    source, destination = source.resolve(), destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Migration destination must be outside the source store")
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination}")

    metadata_bytes = (source / METADATA_FILE).read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata["schema_version"] != LEGACY_SCHEMA_VERSION:
        raise ValueError(f"Migration requires a {LEGACY_SCHEMA_VERSION} source")
    with np.load(source / INDEX_FILE, allow_pickle=False) as saved:
        columns = {name: saved[name] for name in saved.files}
    frame_count = _require_columns(columns, LEGACY_FRAME_FIELDS, extra={}, table="frame")
    _require_columns(columns, INSTANCE_FIELDS,
        extra={"bbox_xyxy": (np.float32, (4,))}, table="instance")
    if set(columns) != set(LEGACY_FRAME_FIELDS) | set(INSTANCE_FIELDS) | {"bbox_xyxy"}:
        raise ValueError("Legacy player store contains columns outside its declared schema")
    if (columns["inst_count"] <= 0).any():
        raise ValueError("Legacy player store cannot contain empty frames or negative counts")
    columns["confirmed_empty"] = np.zeros(frame_count, dtype=np.bool_)
    metadata.update(schema_version=SCHEMA_VERSION, version=destination.name)
    metadata["counts"].update(confirmed_empty_frames=0, ignored_empty_frames=0)
    metadata["migration"] = {
        "source_schema_version": LEGACY_SCHEMA_VERSION,
        "source_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".migrating-{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "store"
        shutil.copytree(source / SHARDS_DIR, staging / SHARDS_DIR)
        savez: Any = np.savez  # numpy stubs type **arrays as allow_pickle
        savez(staging / INDEX_FILE, **columns)
        (staging / METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "README.md").write_text(
            f"# player_detection dataset `{destination.name}`\n\n"
            "Migrated with `python -m src.tasks.player_detection.data.migration`.\n"
            "Schema and access API: `src/tasks/player_detection/data/store.py`.\n"
            "Sparse v1 frame membership, splits, JPEG bytes and teachers are preserved.\n"
            "Missing frames are unknown; no negative teachers were added.\n\n"
            f"```json\n{json.dumps(metadata['counts'], indent=2)}\n```\n",
            encoding="utf-8",
        )
        PlayerFrameStore(staging)
        if destination.exists():
            raise FileExistsError(f"Migration destination already exists: {destination}")
        staging.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing v1 store (absolute path)")
    parser.add_argument("destination", type=Path, help="New v2 store (absolute path)")
    args = parser.parse_args()
    print(migrate_v1_store(args.source, args.destination))


if __name__ == "__main__":
    main()
