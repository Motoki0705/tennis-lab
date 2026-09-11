"""Tests for the append-only structured dataset index."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tennis_scene.generate_dataset.manifest import (
    load_dataset_manifest,
    register_exported_clip,
)
from src.utils.io import load_json, save_json_atomic


def test_registered_clip_round_trips(structured_dataset: Path) -> None:
    dataset = load_dataset_manifest(structured_dataset)
    record = dataset.clips["video_000/clip_000"]
    assert record.path == "videos/video_000/clips/clip_000"
    assert record.num_cameras == 1
    assert record.num_frames == 3


def test_same_clip_name_from_later_video_can_be_appended(
    structured_dataset: Path,
) -> None:
    first = load_json(
        structured_dataset / "videos" / "video_000" / "clips" / "clip_000" / "clip.json"
    )
    second = {**first, "clip_id": "video_001/clip_000", "video_id": "video_001"}
    path = (
        structured_dataset / "videos" / "video_001" / "clips" / "clip_000" / "clip.json"
    )
    save_json_atomic(second, path)
    (path.parent / "media").mkdir()
    (path.parent / "media" / "cam0.mp4").write_bytes(b"video")

    dataset = register_exported_clip(structured_dataset, path)
    assert sorted(dataset.clips) == [
        "video_000/clip_000",
        "video_001/clip_000",
    ]


def test_changed_contract_for_existing_clip_id_is_rejected(
    structured_dataset: Path,
) -> None:
    path = (
        structured_dataset / "videos" / "video_000" / "clips" / "clip_000" / "clip.json"
    )
    changed = load_json(path)
    changed["num_frames"] = 4
    save_json_atomic(changed, path)
    with pytest.raises(ValueError, match="clip_id collision"):
        register_exported_clip(structured_dataset, path)


def test_explicit_replace_updates_existing_clip_contract(
    structured_dataset: Path,
) -> None:
    path = (
        structured_dataset / "videos" / "video_000" / "clips" / "clip_000" / "clip.json"
    )
    changed = load_json(path)
    changed["num_frames"] = 4
    save_json_atomic(changed, path)
    dataset = register_exported_clip(structured_dataset, path, allow_replace=True)
    assert dataset.clips["video_000/clip_000"].num_frames == 4
