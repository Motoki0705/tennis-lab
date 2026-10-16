"""End-to-end store build from a synthetic prepared chat-annotation root."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.tasks.player_detection.configuration import FrameSelectionConfig
from src.tasks.player_detection.data.detection_dataset import select_detection_frames
from src.tasks.player_detection.data.store import (
    BBOX_SOURCE_CODES,
    INDEX_FILE,
    PlayerFrameStore,
)
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tennis_scene.chat_annotation.runtime.contracts import read_json, write_json
from tests.support.tasks.player_detection.chat_root import (
    SyntheticRoot,
    frame_pixels,
)


def test_build_stores_only_player_frames_with_tracks_and_timing(synthetic_root: SyntheticRoot) -> None:
    destination = build_dataset(synthetic_root.config)
    store = PlayerFrameStore(destination)

    assert [clip.clip_id for clip in store.clips] == ["src_a__run__f000-006"]
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["clips_without_players"] == ["src_b__run__f010-016"]
    assert metadata["counts"]["stored_frames"] == 3
    # Frames without players (0, 3, 5) are not stored; clip-local indices stay.
    assert store.frames["frame_index"].tolist() == [1, 2, 4]
    expected_pts = [synthetic_root.manifest.frames[i].clip_pts for i in (1, 2, 4)]
    assert store.frames["clip_pts"].tolist() == expected_pts
    clip = store.clips[0]
    assert clip.track_ids == ("p1", "p2")
    assert clip.split == "train"  # fewer than three source groups
    assert store.clip_frames(clip).tolist() == [0, 1, 2]

    first = store.instances_of(0)
    assert [clip.track_ids[i] for i in first.track_index] == ["p2", "p1"]
    np.testing.assert_allclose(first.boxes_xyxy[1], [50, 10, 70, 60])
    second = store.instances_of(1)
    assert second.bbox_source.tolist() == [BBOX_SOURCE_CODES["observed"], BBOX_SOURCE_CODES["unresolved"]]
    assert np.isnan(second.boxes_xyxy[1]).all()
    assert store.instances_of(2).truncated.tolist() == [True]

    for row, frame_index in enumerate([1, 2, 4]):
        expected_bgr = cv2.cvtColor(frame_pixels(frame_index), cv2.COLOR_RGB2BGR).astype(np.int16)
        # Lossless clip -> JPEG q95: only small chroma/quantization error.
        assert np.abs(store.read_bgr(row).astype(np.int16) - expected_bgr).mean() < 4.0


def test_selection_drops_unresolved_frames_and_clips_truncated_boxes(synthetic_root: SyntheticRoot) -> None:
    store = PlayerFrameStore(build_dataset(synthetic_root.config))
    selection = select_detection_frames(
        store,
        "train",
        FrameSelectionConfig(require_reviewed=True, require_located_players=True, min_visible_box_px=4.0),
        frame_stride=1,
    )
    assert selection.frames.tolist() == [0, 2]
    assert selection.stats["dropped_unresolved_player"] == 1
    np.testing.assert_allclose(selection.boxes_xyxy[1], [[80, 20, 96, 64]])

    permissive = select_detection_frames(
        store,
        "train",
        FrameSelectionConfig(require_reviewed=True, require_located_players=False, min_visible_box_px=20.0),
        frame_stride=1,
    )
    # Frame 4's clipped box is 16 px wide (< 20): no visible box remains.
    assert permissive.frames.tolist() == [0, 1]
    assert permissive.stats["dropped_no_visible_box"] == 1


def test_build_refuses_existing_version_and_invalid_annotations(synthetic_root: SyntheticRoot) -> None:
    build_dataset(synthetic_root.config)
    with pytest.raises(FileExistsError):
        build_dataset(synthetic_root.config)

    path = synthetic_root.root / "annotated/processed/player/src_a__run__f000-006.json"
    annotation = read_json(path)
    annotation["frames"].pop()
    write_json(path, annotation)
    config = synthetic_root.config
    other = type(config)(**{**{f: getattr(config, f) for f in config.__slots__}, "dataset_dir": config.dataset_dir.parent / "v2"})
    with pytest.raises(ValueError, match="invalid annotation"):
        build_dataset(other)
    assert not other.dataset_dir.exists()


def test_store_rejects_inconsistent_index(synthetic_root: SyntheticRoot) -> None:
    destination = build_dataset(synthetic_root.config)
    with np.load(destination / INDEX_FILE) as data:
        columns = {name: data[name] for name in data.files}
    columns["bbox_xyxy"][0] = np.nan  # observed player without a box
    np.savez(destination / INDEX_FILE, **columns)
    with pytest.raises(ValueError, match="NaN boxes"):
        PlayerFrameStore(destination)
