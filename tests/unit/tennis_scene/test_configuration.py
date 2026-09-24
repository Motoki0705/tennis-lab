"""Tests for the strict tennis-scene runtime configuration boundaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.configuration import (
    PipelineRuntimeConfig,
    parse_clip_studio_config,
    parse_visualize_tasks_config,
)
from src.utils.configuration import (
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT

_TENNIS_SCENE_CONFIG_DIR = (PROJECT_ROOT / "src/tennis_scene/configs").resolve()


@contextmanager
def _composed(config_name: str, overrides: list[str]) -> Iterator[DictConfig]:
    with initialize_config_dir(
        config_dir=str(_TENNIS_SCENE_CONFIG_DIR), version_base="1.3"
    ):
        yield compose(config_name=config_name, overrides=overrides)


def _pipeline_config(root: Path, *overrides: str) -> DictConfig:
    """Compose the shipped pipeline config with a temporary project root."""
    with _composed(
        "pipeline",
        [
            f"paths.project_root={root.resolve()}",
            *overrides,
        ],
    ) as config:
        return config


def _visualize_tasks_config(root: Path, *overrides: str) -> DictConfig:
    """Compose the shipped per-task visualization config."""
    with _composed(
        "visualize_tasks",
        [f"paths.project_root={root.resolve()}", *overrides],
    ) as config:
        return config


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


def test_automatic_pipeline_defaults(tmp_path: Path) -> None:
    runtime = PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path))
    assert runtime.inference_policy.min_frames == 512
    assert runtime.inference_policy.max_frames == 1024
    assert runtime.camera_geometry.reference_camera is None
    assert runtime.people.runtime.static_cam
    assert runtime.enabled["plcs_reid"] and "blcs_association" not in runtime.enabled


@pytest.mark.parametrize("override", ["+player_motion.source=plcs", "+court_reference.view_half_turns=[false,false,true]", "+player_association.mode=manual_ui"])
def test_automatic_pipeline_rejects_removed_manual_and_3d_settings(tmp_path: Path, override: str) -> None:
    with pytest.raises(UnknownConfigurationKeyError):
        PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path, override))


@pytest.mark.parametrize("override", ["association.min_player_probability=0.0", "association.cosine_threshold=1.0", "association.max_frames=100", "camera_geometry.side_max_cost=-1", "person_observations.sideline_margin_m=-1"])
def test_automatic_pipeline_rejects_invalid_operating_thresholds(tmp_path: Path, override: str) -> None:
    with pytest.raises(ValueError):
        PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path, override))


def test_visualize_tasks_defaults_to_automatic_reconstruction(
    tmp_path: Path,
) -> None:
    runtime = parse_visualize_tasks_config(_visualize_tasks_config(tmp_path))

    assert "gvhmr_alignment" not in runtime.tasks
    assert "player_reconstruction" in runtime.tasks
    assert "ball_reconstruction" in runtime.tasks


def test_visualize_tasks_accepts_gvhmr_alignment_on_its_own(
    tmp_path: Path,
) -> None:
    runtime = parse_visualize_tasks_config(
        _visualize_tasks_config(tmp_path, "tasks=[gvhmr_alignment]")
    )

    assert runtime.tasks == ("gvhmr_alignment",)
