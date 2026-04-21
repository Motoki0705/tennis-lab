"""Helpers for resolving the default BLCS generator configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from src.tasks.blcs.scripts.generate_dataset import build_generator_config

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig


@lru_cache(maxsize=1)
def build_default_generator_config() -> GeneratorConfig:
    """Build the default BLCS generator config without Hydra composition."""
    config_root = Path(__file__).resolve().parents[1] / "configs"
    cfg = OmegaConf.create(
        {
            "physics": OmegaConf.load(config_root / "physics" / "default.yaml"),
            "rally": OmegaConf.load(config_root / "rally" / "default.yaml"),
            "camera": OmegaConf.load(config_root / "camera" / "default.yaml"),
            "targeted_velocity": OmegaConf.load(
                config_root / "targeted_velocity" / "default.yaml"
            ),
            "generator": OmegaConf.load(config_root / "generator" / "default.yaml"),
        }
    )
    return build_generator_config(cfg)