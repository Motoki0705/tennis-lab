"""BLCS dataset generation settings and boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.tasks.ball_refiner_3d.configuration.core import parse_section, resolved_config
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.utils.configuration import PathRole


@dataclass(frozen=True)
class CameraSampling:
    views: int
    width: int
    height: int
    z_min: float
    z_max: float
    hfov_min: float
    hfov_max: float
    look_x: float
    look_y: float
    look_z_min: float
    look_z_max: float
    maximum_attempts: int

    def __post_init__(self) -> None:
        if self.views < 2 or self.views % 2 or min(self.width, self.height) < 2:
            raise ValueError("Use an even camera count >=2 and valid image dimensions")
        if (
            not 0 < self.z_min <= self.z_max
            or not 0 < self.hfov_min <= self.hfov_max < 170
        ):
            raise ValueError("Invalid height/field-of-view ranges")
        if (
            min(self.look_x, self.look_y, self.look_z_min) < 0
            or self.look_z_min > self.look_z_max
        ):
            raise ValueError("Invalid look-at ranges")
        if self.maximum_attempts < 1:
            raise ValueError(
                "Camera sampling requires a finite positive attempt budget"
            )


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str
    train_rallies: int
    val_rallies: int
    test_rallies: int
    seed: int
    workers: int
    min_frames: int
    maximum_attempts: int

    def __post_init__(self) -> None:
        if (
            min(
                self.train_rallies,
                self.val_rallies,
                self.test_rallies,
                self.workers,
                self.min_frames,
                self.maximum_attempts,
            )
            < 1
            or self.seed < 0
        ):
            raise ValueError("Require positive generation counts and nonnegative seed")


def generation_config(
    config: DictConfig,
) -> tuple[dict[str, Any], Path, GenerationConfig, CameraSampling, GeneratorConfig]:
    raw, resolver = resolved_config(
        config,
        {
            "paths",
            "generation",
            "camera_sampling",
            "physics",
            "rally",
            "camera",
            "generator",
            "targeted_velocity",
            "court_keypoints",
        },
    )
    generation = parse_section(GenerationConfig, raw["generation"])
    camera = parse_section(CameraSampling, raw["camera_sampling"])
    physics = build_generator_config(config)
    if (
        physics.rally.sim_fps % physics.rally.output_fps
        or physics.rally.output_fps <= 0
    ):
        raise ValueError("BLCS simulation FPS must be divisible by output FPS")
    if not np.isclose(physics.physics.dt, 1 / physics.rally.sim_fps):
        raise ValueError("Physics dt must match simulation FPS")
    raw["paths"] = {name: str(value) for name, value in asdict(resolver.roots).items()}
    return (
        raw,
        resolver.resolve(PathRole.DATA, generation.dataset),
        generation,
        camera,
        physics,
    )


def validate_generation(config: DictConfig) -> None:
    generation_config(config)
