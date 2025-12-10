import pytest
from omegaconf import OmegaConf
from pathlib import Path

import src.wasb.scripts.generate_game as generate_game_module

from src.wasb.pipeline import PipelineConfig
from src.wasb.scripts.generate_game import (
    process_single_video,
    process_video_directory,
    run_from_config,
    _create_pipeline_config_from_cfg,
    apply_completion,
)


def test_run_from_config_invalid_mode_raises() -> None:
    cfg = OmegaConf.create({"mode": "invalid_mode"})

    with pytest.raises(ValueError):
        run_from_config(cfg)


def test_process_single_video_requires_video() -> None:
    # Missing video should fail early without touching filesystem.
    cfg = OmegaConf.create({"mode": "single_video"})

    exit_code = process_single_video(cfg)

    assert exit_code == 1


def test_process_video_directory_requires_video_dir() -> None:
    # Missing video_dir should fail early without touching filesystem.
    cfg = OmegaConf.create({"mode": "batch"})

    exit_code = process_video_directory(cfg)

    assert exit_code == 1


def test_create_pipeline_config_uses_pipeline_section() -> None:
    cfg = OmegaConf.create(
        {
            "pipeline": {
                "score_threshold": 0.7,
                "min_clip_length": 40,
                "min_detection_rate": 0.8,
                "max_gap": 12,
                "clip_padding": 7,
                "batch_size": 256,
                "frame_format": "frame_{:06d}.jpg",
                "jpeg_quality": 80,
                "streaming_batch_size": 32,
                "streaming_queue_size": 8,
                "use_streaming": False,
                "streaming_threshold": 1000,
                "use_completion": False,
                "completion_method": "physics",
                "completion_checkpoint": "checkpoint.pth",
                "physics_gap_threshold": 3,
                "max_completion_gap": 20,
            }
        }
    )

    config = _create_pipeline_config_from_cfg(cfg)

    assert isinstance(config, PipelineConfig)
    assert config.score_threshold == 0.7
    assert config.min_clip_length == 40
    assert config.min_detection_rate == 0.8
    assert config.max_gap == 12
    assert config.clip_padding == 7
    assert config.batch_size == 256
    assert config.frame_format == "frame_{:06d}.jpg"
    assert config.jpeg_quality == 80
    assert config.streaming_batch_size == 32
    assert config.streaming_queue_size == 8
    assert config.use_streaming is False
    assert config.streaming_threshold == 1000
    assert config.use_completion is False
    assert config.completion_method == "physics"
    assert config.completion_checkpoint == "checkpoint.pth"
    assert config.physics_gap_threshold == 3
    assert config.max_completion_gap == 20


def test_create_pipeline_config_requires_pipeline_section() -> None:
    cfg = OmegaConf.create({})

    with pytest.raises(ValueError):
        _create_pipeline_config_from_cfg(cfg)


def test_apply_completion_uses_pipeline_config(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_resolve_path(path_str: str) -> Path:
        return Path("dummy")

    def fake_resolve_games(output_dir: Path, games_cfg: list[str]) -> list[str]:
        return ["game11"]

    def fake_create_completer(
        *,
        method: str,
        checkpoint_path,
        physics_gap_threshold: int,
        max_gap: int,
        score_threshold: float,
    ) -> object:
        called["method"] = method
        called["checkpoint_path"] = checkpoint_path
        called["physics_gap_threshold"] = physics_gap_threshold
        called["max_gap"] = max_gap
        called["score_threshold"] = score_threshold
        return object()

    def fake_apply_completion_to_game(
        output_dir: Path,
        game_name: str,
        completer: object,
        score_threshold: float,
        verbose: bool = True,
    ) -> dict:
        called["score_threshold_used_in_apply"] = score_threshold
        return {"clips": 1, "frames": 10, "detected": 8, "completed": 2, "missing": 0}

    def fake_update_meta_completion(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        called["update_meta_called"] = True

    monkeypatch.setattr(generate_game_module, "_resolve_path", fake_resolve_path)
    monkeypatch.setattr(generate_game_module, "_resolve_games", fake_resolve_games)
    monkeypatch.setattr(generate_game_module, "create_completer", fake_create_completer)
    monkeypatch.setattr(
        generate_game_module,
        "_apply_completion_to_game",
        fake_apply_completion_to_game,
    )
    monkeypatch.setattr(
        generate_game_module,
        "_update_meta_completion",
        fake_update_meta_completion,
    )

    cfg = OmegaConf.create(
        {
            "output_dir": "data/tennis",
            "apply_completion": ["game11"],
            "pipeline": {
                "score_threshold": 0.6,
                "completion_method": "physics",
                "completion_checkpoint": "path/to/checkpoint.pth",
                "physics_gap_threshold": 7,
                "max_completion_gap": 21,
            },
        }
    )

    exit_code = apply_completion(cfg)

    assert exit_code == 0
    assert called["method"] == "physics"
    assert called["checkpoint_path"] == "path/to/checkpoint.pth"
    assert called["physics_gap_threshold"] == 7
    assert called["max_gap"] == 21
    assert called["score_threshold"] == 0.6
    assert called["score_threshold_used_in_apply"] == 0.6
    assert called["update_meta_called"] is True
