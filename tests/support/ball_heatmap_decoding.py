"""Decoder test settings from the same Hydra composition as the scene pipeline."""

from functools import lru_cache
from typing import cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.ball_detection.inference.heatmap_decoding import HeatmapDecodingConfig
from src.tennis_scene.configuration import build_heatmap_decoding_config
from src.utils.paths import PROJECT_ROOT


@lru_cache
def compose_heatmap_decoding(*overrides: str) -> HeatmapDecodingConfig:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        config = compose(config_name="pipeline", overrides=list(overrides))
    settings = OmegaConf.to_container(config.ball_detection.decoding, resolve=True)
    return build_heatmap_decoding_config(cast(dict[str, object], settings))
