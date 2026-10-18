"""Build an explicit sparse v1 fixture from tiny v2 test data."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.player_detection.data.migration import (
    LEGACY_FRAME_FIELDS,
    LEGACY_SCHEMA_VERSION,
)
from src.tasks.player_detection.data.store import (
    INDEX_FILE,
    METADATA_FILE,
    SHARDS_DIR,
    PlayerFrameStore,
    shard_name,
)


def legacy_store(directory: Path, destination: Path) -> Path:
    store = PlayerFrameStore(directory)
    rows = np.flatnonzero(store.frames["inst_count"] > 0)
    assert len(rows)
    columns = {name: store.frames[name][rows].copy() for name in LEGACY_FRAME_FIELDS}
    columns.update({name: values.copy() for name, values in store.instances.items()})
    (destination / SHARDS_DIR).mkdir(parents=True)
    metadata: dict[str, Any] = copy.deepcopy(store.metadata)
    clips: list[dict[str, Any]] = []
    for clip in store.clips:
        kept = np.flatnonzero(columns["clip"] == clip.index)
        if not len(kept):
            continue
        index = len(clips)
        source = (directory / SHARDS_DIR / shard_name(clip.index)).read_bytes()
        with (destination / SHARDS_DIR / shard_name(index)).open("wb") as output:
            for row in kept:
                offset, length = int(columns["offset"][row]), int(columns["length"][row])
                columns["offset"][row] = output.tell()
                output.write(source[offset:offset + length])
        columns["clip"][kept] = index
        record = copy.deepcopy(metadata["clips"][clip.index])
        record.update(index=index, stored_frames=len(kept))
        clips.append(record)
    metadata.update(schema_version=LEGACY_SCHEMA_VERSION, version=destination.name, clips=clips)
    counts = metadata["counts"]
    counts.pop("confirmed_empty_frames")
    counts.pop("ignored_empty_frames")
    counts.update(clips=len(clips), stored_frames=len(rows), annotated_frames=sum(c["frame_count"] for c in clips),
        clips_by_split={split: sum(c["split"] == split for c in clips) for split in ("train", "val", "test")},
        frames_by_split={split: sum(c["stored_frames"] for c in clips if c["split"] == split) for split in ("train", "val", "test")})
    np.savez(destination / INDEX_FILE, **columns)
    (destination / METADATA_FILE).write_text(json.dumps(metadata))
    return destination
