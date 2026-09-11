"""Tests for the strict tennis-scene runtime configuration boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.tennis_scene.configuration import parse_clip_studio_config
from src.utils.configuration import UnknownConfigurationKeyError


def _clip_studio_config(root: Path) -> dict[str, object]:
    source = root / "data/tennis_multivew/raw/match01/video_000"
    source.mkdir(parents=True, exist_ok=True)
    (source / "cam0.mp4").touch()
    (source / "cam1.mp4").touch()
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
        "source_directory": "tennis_multivew/raw/match01/video_000",
        "gui": {
            "canvas_width": 1600,
            "tile_width": 640,
            "cache_frames": 96,
            "seek_grab_threshold": 24,
            "window_name": "Tennis Clip Studio",
            "zoom_step": 1.5,
            "port": 8765,
        },
        "audio_sync": {
            "sample_rate": 8000,
            "envelope_rate": 100.0,
            "max_seconds": None,
        },
        "export": {
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

    assert (
        runtime.export.projects_path
        == (tmp_path / "data/tennis_multivew/processed/match01/projects.json").resolve()
    )
    assert (
        runtime.export.output_dir
        == (tmp_path / "data/tennis_multivew/processed/match01/dataset").resolve()
    )
    assert runtime.video_paths == (
        (tmp_path / "data/tennis_multivew/raw/match01/video_000/cam0.mp4").resolve(),
        (tmp_path / "data/tennis_multivew/raw/match01/video_000/cam1.mp4").resolve(),
    )
    assert runtime.camera_ids == ("cam0", "cam1")
    assert runtime.dataset_id == "match01"
    assert runtime.video_id == "video_000"


@pytest.mark.parametrize(
    "former_key",
    [
        "match_id",
        "dataset_root",
        "project_path",
        "recording_id",
        "video_paths",
        "camera_ids",
    ],
)
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


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_clip_studio_rejects_invalid_port(tmp_path: Path, port: int) -> None:
    config = _clip_studio_config(tmp_path)
    gui = config["gui"]
    assert isinstance(gui, dict)
    gui["port"] = port
    with pytest.raises(ValueError, match="gui.port"):
        parse_clip_studio_config(OmegaConf.create(config))
