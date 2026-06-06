from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.annotation.session import filter_source_entries
from src.tasks.court_detection.scripts.prepare_youtube_dataset import (
    _initial_annotation_item,
    _normalize_youtube_annotation_item,
    _write_annotations,
)


def _frame(image_id: str = "yt_000001_f00000000") -> dict[str, object]:
    return {
        "id": image_id,
        "image_path": f"frames/video_000001/{image_id}.jpg",
        "width": 1920,
        "height": 1080,
        "video_id": "video_000001",
        "source_url": "https://www.youtube.com/watch?v=example",
        "source_title": "Example",
        "source_frame_index": 0,
        "timestamp_sec": 0.0,
    }


def _annotation_config() -> DictConfig:
    return OmegaConf.create({
        "schema_name": "court_youtube_keypoints_v2",
        "keypoint_format": "kp20",
        "merge_existing": True,
        "overwrite": False,
    })


def test_youtube_item_keeps_keypoint_metadata_out_of_source() -> None:
    item = _initial_annotation_item(_frame(), "train", _annotation_config())

    assert item["keypoint_format"] == "kp20"
    assert item["labeled_keypoint_indices"] == list(range(20))
    assert item["is_yastrebksv_kp15"] is False
    assert item["source"]["type"] == "youtube"
    assert "dataset" not in item["source"]
    assert "keypoint_format" not in item["source"]
    assert "labeled_keypoint_indices" not in item["source"]


def test_existing_youtube_item_is_normalized_without_losing_keypoints() -> None:
    existing = _initial_annotation_item(_frame(), "train", _annotation_config())
    existing["keypoint_format"] = "kp15"
    existing["labeled_keypoint_indices"] = list(range(15))
    existing["source"].update({
        "dataset": "yastrebksv_kp15",
        "keypoint_format": "kp15",
        "labeled_keypoint_indices": list(range(15)),
    })
    existing["annotation_status"] = "completed"
    existing["keypoints"][0]["x"] = 123.0

    item = _normalize_youtube_annotation_item(existing, _frame(), "train", _annotation_config())

    assert item["keypoint_format"] == "kp20"
    assert item["labeled_keypoint_indices"] == list(range(20))
    assert item["keypoints"][0]["x"] == 123.0
    assert item["annotation_status"] == "pending"
    assert item["source"]["type"] == "youtube"
    assert "dataset" not in item["source"]
    assert "keypoint_format" not in item["source"]
    assert "labeled_keypoint_indices" not in item["source"]


def test_merge_existing_preserves_non_youtube_items(tmp_path: Path) -> None:
    legacy_item = {
        "id": "legacy_001",
        "keypoint_format": "kp15",
        "is_yastrebksv_kp15": True,
        "source": {"type": "yastrebksv"},
    }
    (tmp_path / "train.json").write_text(
        json.dumps({"schema_name": "old", "items": [legacy_item]}),
        encoding="utf-8",
    )

    _write_annotations(
        tmp_path,
        {"train": [_frame()], "val": []},
        _annotation_config(),
    )

    payload = json.loads((tmp_path / "train.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in payload["items"]] == [
        "yt_000001_f00000000",
        "legacy_001",
    ]


def test_source_type_filter_excludes_legacy_kp15_items() -> None:
    entries = [
        {"id": "youtube", "source": {"type": "youtube"}},
        {"id": "legacy", "source": {"type": "yastrebksv"}},
    ]

    assert filter_source_entries(entries, ("youtube",)) == [entries[0]]
