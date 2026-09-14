"""Tests for the strict tennis-scene runtime configuration boundaries."""

from __future__ import annotations

import math
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
    SemanticConfigurationError,
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
        [f"paths.project_root={root.resolve()}", *overrides],
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


def test_player_motion_defaults_select_plcs_and_reuse_the_bundled_regressor(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path))

    assert runtime.player_motion.source == "plcs"
    assert runtime.player_motion.scale_mode == "fixed"
    assert (
        runtime.player_motion.smpl_joint_regressor
        == runtime.gvhmr.bundled_assets.smpl_neutral_joint_regressor
    )
    similarity = runtime.player_motion.similarity
    assert similarity.sigma_position == pytest.approx(0.5)
    assert similarity.sigma_heading == pytest.approx(math.radians(30.0))
    assert similarity.heading_weight == pytest.approx(1.0)
    assert similarity.scale_prior == pytest.approx(1.0)
    assert similarity.min_scale == pytest.approx(0.5)
    assert similarity.max_scale == pytest.approx(2.0)
    assert similarity.huber_delta == pytest.approx(1.0)
    assert similarity.heading_resultant_threshold == pytest.approx(0.5)
    assert similarity.max_nfev == 500
    # The declared degree value is converted once; the scale mode is applied
    # when the module builds its fit config.
    assert similarity.fixed_scale is None
    assert runtime.player_motion.fit_config().fixed_scale == 1.0


def test_player_motion_accepts_the_aligned_source_and_free_scale(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntimeConfig.from_config(
        _pipeline_config(
            tmp_path,
            "player_motion.source=gvhmr_alignment",
            "player_motion.scale_mode=free",
        )
    )

    assert runtime.player_motion.source == "gvhmr_alignment"
    assert runtime.player_motion.scale_mode == "free"
    assert runtime.player_motion.fit_config().fixed_scale is None


@pytest.mark.parametrize(
    "override",
    [
        "player_motion.source=world",
        "player_motion.source=PLCS",
        "player_motion.scale_mode=huge",
        "player_motion.scale_mode=Fixed",
    ],
)
def test_player_motion_rejects_unknown_choices(
    tmp_path: Path, override: str
) -> None:
    with pytest.raises(SemanticConfigurationError, match="player_motion"):
        PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path, override))


@pytest.mark.parametrize(
    "override",
    [
        "player_motion.alignment.sigma_position_m=0.0",
        "player_motion.alignment.sigma_position_m=-1.0",
        "player_motion.alignment.sigma_heading_deg=0.0",
        "player_motion.alignment.sigma_heading_deg=-30.0",
        "player_motion.alignment.huber_delta=0.0",
    ],
)
def test_player_motion_rejects_non_positive_sigmas(
    tmp_path: Path, override: str
) -> None:
    with pytest.raises(SemanticConfigurationError, match="player_motion"):
        PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path, override))


def test_player_motion_rejects_inverted_scale_bounds(tmp_path: Path) -> None:
    with pytest.raises(SemanticConfigurationError, match="min_scale"):
        PipelineRuntimeConfig.from_config(
            _pipeline_config(
                tmp_path,
                "player_motion.alignment.min_scale=3.0",
                "player_motion.alignment.max_scale=2.0",
            )
        )


def test_player_motion_rejects_an_unknown_alignment_key(tmp_path: Path) -> None:
    with pytest.raises(UnknownConfigurationKeyError, match="alignment"):
        PipelineRuntimeConfig.from_config(
            _pipeline_config(
                tmp_path, "+player_motion.alignment.unexpected_value=1.0"
            )
        )


def test_visualize_tasks_accepts_gvhmr_alignment_by_default(
    tmp_path: Path,
) -> None:
    runtime = parse_visualize_tasks_config(_visualize_tasks_config(tmp_path))

    assert "gvhmr_alignment" in runtime.tasks


def test_visualize_tasks_accepts_gvhmr_alignment_on_its_own(
    tmp_path: Path,
) -> None:
    runtime = parse_visualize_tasks_config(
        _visualize_tasks_config(tmp_path, "tasks=[gvhmr_alignment]")
    )

    assert runtime.tasks == ("gvhmr_alignment",)
