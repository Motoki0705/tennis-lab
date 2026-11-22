"""Video target preflight rejects every malformed selection before output creation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tennis_scene.pipeline.clip_inputs import prepare_video_inputs
from src.utils.video import VideoInfo
from tests.support.tennis_scene.video_dataset import add_clip


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "dataset"
    for key in ("video_000/clip_000", "video_000/clip_001", "video_001/clip_000"):
        add_clip(root, key)
    monkeypatch.setattr("src.tennis_scene.pipeline.source.probe_video_info", lambda _: VideoInfo(30., 1280, 720, 24))
    return root


@pytest.mark.parametrize("selection,expected", [(None, ("video_000/clip_000", "video_000/clip_001")),
    (("video_000/clip_001",), ("video_000/clip_001",))])
def test_video_all_and_explicit_subset_are_frozen(dataset: Path, selection, expected) -> None:
    dataset_id, prepared = prepare_video_inputs(dataset, "video_000", selection)
    add_clip(dataset, "video_000/clip_002")
    assert dataset_id == "batch-fixture"
    assert tuple(clip.inputs.record.clip_id for clip in prepared) == expected
    assert all(clip.media.source.camera_ids == ("cam0", "cam1", "cam2") for clip in prepared)


@pytest.mark.parametrize("video,selection,message", [
    ("video_000", ("video_001/clip_000",), "another video"),
    ("video_000", ("video_000/unknown",), "Unknown"),
    ("video_000", ("video_000/clip_000", "video_000/clip_000"), "unique"),
    ("video_000", (), "nonempty"), ("unknown", None, "nonempty"), ("../video", None, "video_id"),
])
def test_invalid_targets_create_no_outputs(dataset: Path, video: str, selection, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_video_inputs(dataset, video, selection)
    assert not list(dataset.rglob("annotations"))


@pytest.mark.parametrize("change,message", [
    ({"num_frames": 23}, "timeline"), ({"video_paths": ["cam0.mp4"] * 3}, "one-to-one"),
    ({"camera_ids": ["cam0", "cam0", "cam2"]}, "duplicates"),
    ({"dataset_id": "foreign"}, "another dataset"), ({"video_paths": ["cam0.mp4", "cam1.mp4", "missing.mp4"]}, "missing"),
])
def test_invalid_clip_manifest_stops_preflight(dataset: Path, change: dict, message: str) -> None:
    path = dataset / "videos/video_000/clips/clip_001/clip.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError, match=message):
        prepare_video_inputs(dataset, "video_000")
    assert not list(dataset.rglob("annotations"))


@pytest.mark.parametrize("info,message", [
    (VideoInfo(30., 1280, 720, 23), "frame count"),
    (VideoInfo(29., 1280, 720, 24), "FPS"),
    (VideoInfo(30., 640, 360, 24), "image size"),
])
def test_actual_camera_media_must_be_synchronized(dataset: Path, monkeypatch: pytest.MonkeyPatch, info: VideoInfo, message: str) -> None:
    monkeypatch.setattr("src.tennis_scene.pipeline.source.probe_video_info",
        lambda path: info if path.parent.name == "clip_001" and path.stem == "cam2" else VideoInfo(30., 1280, 720, 24))
    with pytest.raises(ValueError, match=message):
        prepare_video_inputs(dataset, "video_000")
    assert not list(dataset.rglob("annotations"))


def test_actual_media_must_match_registered_timeline(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.tennis_scene.pipeline.source.probe_video_info", lambda _: VideoInfo(30., 1280, 720, 25))
    with pytest.raises(ValueError, match="Actual media timeline"):
        prepare_video_inputs(dataset, "video_000")
