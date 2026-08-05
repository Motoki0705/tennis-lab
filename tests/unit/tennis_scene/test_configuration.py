"""Tests for the strict tennis-scene runtime configuration boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tennis_scene.configuration import parse_clip_studio_config
from src.utils.configuration import UnknownConfigurationKeyError


def _clip_studio_config(root: Path) -> dict[str, object]:
    return {
        "paths": {
            "project_root": str(root.resolve()),
            "data_root": "data",
            "checkpoint_root": "ckpt",
            "artifact_root": "outputs",
            "output_root": "outputs",
            "cache_root": ".cache",
            "external_asset_root": "third_party",
        },
        "project_path": "tennis_scene/clip_studio/match01/project.json",
        "recording_id": "match01",
        "video_paths": [
            "tennis_multiview/raw/match01/cam0.mp4",
            "tennis_multiview/raw/match01/cam1.mp4",
        ],
        "camera_ids": ["cam0", "cam1"],
        "gui": {
            "canvas_width": 1600,
            "tile_width": 640,
            "cache_frames": 96,
            "seek_grab_threshold": 24,
            "window_name": "Tennis Clip Studio",
            "zoom_step": 1.5,
        },
        "audio_sync": {
            "sample_rate": 8000,
            "envelope_rate": 100.0,
            "max_seconds": None,
        },
        "export": {
            "output_dir": "tennis_scene/dataset",
            "fps": None,
            "width": None,
            "height": None,
            "crf": 17,
            "overwrite": False,
        },
    }


def test_clip_studio_config_resolves_each_path_from_its_declared_root(
    tmp_path: Path,
) -> None:
    runtime = parse_clip_studio_config(OmegaConf.create(_clip_studio_config(tmp_path)))

    assert runtime.export.project_path == (
        tmp_path / "outputs/tennis_scene/clip_studio/match01/project.json"
    ).resolve()
    assert runtime.export.output_dir == (
        tmp_path / "outputs/tennis_scene/dataset"
    ).resolve()
    assert runtime.video_paths == (
        (tmp_path / "data/tennis_multiview/raw/match01/cam0.mp4").resolve(),
        (tmp_path / "data/tennis_multiview/raw/match01/cam1.mp4").resolve(),
    )
    assert runtime.camera_ids == ("cam0", "cam1")


@pytest.mark.parametrize("former_key", ["match_id", "dataset_root"])
def test_clip_studio_config_rejects_removed_path_aliases(
    tmp_path: Path,
    former_key: str,
) -> None:
    config = _clip_studio_config(tmp_path)
    config[former_key] = "legacy"

    with pytest.raises(
        UnknownConfigurationKeyError,
        match=rf"tennis_scene\.clip_studio\.{former_key}",
    ):
        parse_clip_studio_config(OmegaConf.create(config))
