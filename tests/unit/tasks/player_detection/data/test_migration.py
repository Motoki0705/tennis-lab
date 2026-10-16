"""Migration preserves an immutable sparse snapshot instead of filling gaps."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.tasks.player_detection.configuration import FrameSelectionConfig
from src.tasks.player_detection.data.detection_dataset import select_detection_frames
from src.tasks.player_detection.data.migration import main, migrate_v1_store
from src.tasks.player_detection.data.store import (
    INDEX_FILE,
    METADATA_FILE,
    SCHEMA_VERSION,
    SHARDS_DIR,
    PlayerFrameStore,
    Split,
)
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from tests.support.tasks.player_detection.chat_root import SyntheticRoot
from tests.support.tasks.player_detection.store import legacy_store


def snapshot(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_migration_preserves_sparse_rows_splits_pixels_and_teachers(
    synthetic_root: SyntheticRoot, split: Split,
) -> None:
    original = PlayerFrameStore(build_dataset(synthetic_root.config))
    source = legacy_store(original.directory, original.directory.parent / "legacy-v1")
    metadata_path = source / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata["clips"][0]["split"] = split
    metadata["counts"]["clips_by_split"] = {name: int(name == split) for name in ("train", "val", "test")}
    metadata["counts"]["frames_by_split"] = {name: 3 * int(name == split) for name in ("train", "val", "test")}
    metadata_path.write_text(json.dumps(metadata))
    before = snapshot(source)

    destination = migrate_v1_store(source, source.parent / "migrated-v2")
    store = PlayerFrameStore(destination)
    assert store.schema_version == SCHEMA_VERSION
    assert store.metadata["clips"] == metadata["clips"]
    assert store.metadata["split"] == metadata["split"]
    assert store.metadata["counts"] == {
        **metadata["counts"], "confirmed_empty_frames": 0, "ignored_empty_frames": 0,
    }
    assert store.frames["frame_index"].tolist() == [1, 2, 4]
    assert store.split_frames(split).tolist() == [0, 1, 2]
    assert store.frames["confirmed_empty"].tolist() == [False, False, False]
    with np.load(source / INDEX_FILE) as old, np.load(destination / INDEX_FILE) as new:
        assert set(new.files) == set(old.files) | {"confirmed_empty"}
        for name in old.files:
            assert new[name].dtype == old[name].dtype
            np.testing.assert_array_equal(new[name], old[name])
    assert snapshot(destination / SHARDS_DIR) == snapshot(source / SHARDS_DIR)
    for row, frame in enumerate([1, 2, 4]):
        np.testing.assert_array_equal(store.read_bgr(row), original.read_bgr(frame))
    selection = select_detection_frames(store, split, FrameSelectionConfig(
        require_reviewed=True, require_located_players=True, min_visible_box_px=4.,
    ), frame_stride=1)
    assert selection.frames.tolist() == [0, 2]
    assert selection.stats["negative_frames"] == 0
    assert snapshot(source) == before
    with pytest.raises(FileExistsError, match="already exists"):
        migrate_v1_store(source, destination)


@pytest.mark.parametrize("fault, message", [
    ("empty", "Legacy.*empty"),
    ("extra_column", "outside its declared schema"),
    ("box", "NaN boxes"),
    ("shard", "Shard size"),
])
def test_invalid_legacy_store_is_not_published_or_modified(
    synthetic_root: SyntheticRoot, fault: str, message: str,
) -> None:
    current = build_dataset(synthetic_root.config)
    source = legacy_store(current, current.parent / "legacy-v1")
    with np.load(source / INDEX_FILE) as saved:
        columns = {name: saved[name] for name in saved.files}
    if fault == "empty":
        columns["inst_count"][0] = 0
    elif fault == "extra_column":
        columns["confirmed_empty"] = np.zeros(3, dtype=np.bool_)
    elif fault == "box":
        columns["bbox_xyxy"][0] = np.nan
    else:
        columns["length"][-1] -= 1
    np.savez(source / INDEX_FILE, **columns)
    before = snapshot(source)
    destination = source.parent / "migrated-v2"
    with pytest.raises(ValueError, match=message):
        migrate_v1_store(source, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob(".migrating-*"))
    assert snapshot(source) == before


def test_migration_requires_v1_and_a_new_destination_outside_source(synthetic_root: SyntheticRoot) -> None:
    current = build_dataset(synthetic_root.config)
    source = legacy_store(current, current.parent / "legacy-v1")
    before = snapshot(source)
    for destination in (source, source / "nested" / "v2"):
        with pytest.raises(ValueError, match="outside the source"):
            migrate_v1_store(source, destination)
    with pytest.raises(FileExistsError):
        migrate_v1_store(source, current)
    with pytest.raises(ValueError, match="requires a .*v1 source"):
        migrate_v1_store(current, current.parent / "invalid-v2")
    assert snapshot(source) == before


def test_migration_cli(synthetic_root: SyntheticRoot, monkeypatch: pytest.MonkeyPatch) -> None:
    current = build_dataset(synthetic_root.config)
    source = legacy_store(current, current.parent / "legacy-v1")
    destination = current.parent / "migrated-v2"
    monkeypatch.setattr(sys, "argv", ["migration", str(source), str(destination)])
    main()
    assert PlayerFrameStore(destination).frames["frame_index"].tolist() == [1, 2, 4]
