import pytest
from omegaconf import OmegaConf

from src.wasb.scripts.generate_game import (
    process_single_video,
    process_video_directory,
    run_from_config,
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
