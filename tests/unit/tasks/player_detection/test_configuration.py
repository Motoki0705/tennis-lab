"""Explicit configuration for validation-clip qualitative logging."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.player_detection.configuration import PlayerTrainingConfig

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/player_detection/configs"


@pytest.mark.parametrize("override", [
    "qualitative.max_frames=1", "qualitative.frame_stride=0", "qualitative.batch_size=0",
    "qualitative.display_width=0", "qualitative.annotation_root=../escape",
])
def test_invalid_clip_configuration_is_rejected(override: str) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(config_name="train", overrides=[override])
    with pytest.raises(ValueError):
        PlayerTrainingConfig.from_config(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_config_requires_clip_settings_only_when_enabled(enabled: bool) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(config_name="train")
    with open_dict(config):
        del config.qualitative
    config.training.qualitative_logging.enabled = enabled
    if enabled:
        with pytest.raises(ValueError, match="requires an explicit qualitative configuration"):
            PlayerTrainingConfig.from_config(config)
    else:
        assert PlayerTrainingConfig.from_config(config).qualitative is None
