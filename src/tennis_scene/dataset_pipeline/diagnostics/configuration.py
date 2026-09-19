"""Typed diagnostic requests, resolved before model loading or output creation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.submodules.configuration import ViTPoseHeadConfig
from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
from src.tennis_scene.dataset_pipeline.refinement import RefinementSettings
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    StrictConfigSchema,
)
from src.utils.configuration.output_layout import task_output_path
from src.utils.paths import PROJECT_ROOT


def _mapping(cfg: DictConfig, fields: dict[str, ConfigField], name: str) -> None:
    value = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict):
        raise TypeError("Diagnostic configuration must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("Diagnostic configuration keys must be strings")
    StrictConfigSchema(fields=fields, name=name).validate(
        cast(dict[str, object], value)
    )


def resolver_for(cfg: DictConfig) -> PathResolver:
    return PathResolver(
        RuntimePathRoots.from_mapping(dict(cfg.paths), repository_root=PROJECT_ROOT)
    )


def evaluation_output(resolver: PathResolver, fragment: str, task: str) -> Path:
    parts = fragment.split("/")
    if len(parts) != 4 or parts[:2] != [task, "evaluate"]:
        raise ValueError(f"output must be {task}/evaluate/<experiment>/<run-id>")
    task_output_path(*parts)
    result: Path = resolver.resolve(PathRole.OUTPUT, fragment)
    if result.exists():
        raise FileExistsError(f"Choose a new evaluation run directory: {result}")
    return result


def compose_recipe(
    name: str, resolver: PathResolver, overrides: list[str]
) -> DictConfig:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("recipe must be a configuration name, not a path")
    if any(
        value.lstrip("+~").split("=", 1)[0] == "paths"
        or value.lstrip("+~").startswith("paths.")
        for value in overrides
    ):
        raise ValueError("Set roots through diagnostic paths, not recipe overrides")
    directory = resolver.resolve(PathRole.PROJECT, "src/tennis_scene/configs")
    if GlobalHydra.instance().is_initialized():
        cfg = compose(config_name=name, overrides=overrides)
    else:
        with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
            cfg = compose(config_name=name, overrides=overrides)
    cfg.paths = OmegaConf.create(dict(resolver.roots.as_mapping()))
    return cfg


@dataclass(frozen=True)
class ViTPoseBenchmarkConfig:
    clip: Path
    observations: Path
    checkpoint: Path
    output: Path
    device: str
    cameras: tuple[str, ...]
    people: tuple[int, ...]
    frames: int
    warmup_frames: int
    batch_size: int
    flip_test: bool
    crop_enlarge: float
    head: ViTPoseHeadConfig

    @classmethod
    def from_config(cls, cfg: DictConfig) -> ViTPoseBenchmarkConfig:
        _mapping(
            cfg,
            {
                **{
                    key: ConfigField.of(str)
                    for key in (
                        "clip",
                        "observations",
                        "checkpoint",
                        "output",
                        "device",
                    )
                },
                "paths": ConfigField.of(dict),
                "head": ConfigField.of(dict),
                "cameras": ConfigField.sequence(ConfigField.of(str)),
                "people": ConfigField.sequence(ConfigField.of(int)),
                **{
                    key: ConfigField.of(int)
                    for key in ("frames", "warmup_frames", "batch_size")
                },
                "flip_test": ConfigField.of(bool),
                "crop_enlarge": ConfigField.of(float, int),
            },
            "vitpose_benchmark",
        )
        resolver = resolver_for(cfg)
        if cfg.device != "cuda:0":
            raise ValueError("Precision benchmark requires explicit cuda:0")
        if not 0 < cfg.warmup_frames <= cfg.frames or cfg.batch_size <= 0:
            raise ValueError(
                "Frame counts and batch size must be positive; warmup <= frames"
            )
        if not math.isfinite(cfg.crop_enlarge) or cfg.crop_enlarge <= 0:
            raise ValueError("crop_enlarge must be finite and positive")
        if not cfg.cameras or len(set(cfg.cameras)) != len(cfg.cameras):
            raise ValueError("cameras must be nonempty and unique")
        if any(
            not name or "/" in name or "\\" in name or name in {".", ".."}
            for name in cfg.cameras
        ):
            raise ValueError("camera IDs must be single path components")
        if (
            not cfg.people
            or len(set(cfg.people)) != len(cfg.people)
            or min(cfg.people) < 0
        ):
            raise ValueError("people must be unique nonnegative indices")
        head_fields = dict(cfg.head)
        for key in ("num_deconv_filters", "num_deconv_kernels", "num_conv_kernels"):
            head_fields[key] = tuple(head_fields[key])
        return cls(
            resolver.resolve(PathRole.DATA, cfg.clip),
            resolver.resolve(PathRole.OUTPUT, cfg.observations),
            resolver.resolve(PathRole.EXTERNAL_ASSET, cfg.checkpoint),
            evaluation_output(resolver, cfg.output, "tennis_scene"),
            cfg.device,
            tuple(cfg.cameras),
            tuple(cfg.people),
            cfg.frames,
            cfg.warmup_frames,
            cfg.batch_size,
            cfg.flip_test,
            float(cfg.crop_enlarge),
            ViTPoseHeadConfig(**head_fields),
        )


@dataclass(frozen=True)
class RefinementEvaluationConfig:
    scene: Path
    court: Path
    output: Path
    view_half_turns: tuple[bool, ...]
    settings: RefinementSettings
    refinement_config: Any

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RefinementEvaluationConfig:
        _mapping(
            cfg,
            {
                **{
                    key: ConfigField.of(str)
                    for key in ("scene", "court", "output", "recipe")
                },
                "paths": ConfigField.of(dict),
                "overrides": ConfigField.sequence(ConfigField.of(str)),
            },
            "refinement_evaluation",
        )
        resolver = resolver_for(cfg)
        recipe = compose_recipe(cfg.recipe, resolver, list(cfg.overrides))
        if recipe.coordinate_mode not in {"reference", "physical"}:
            raise ValueError("coordinate_mode must be reference or physical")
        turns = (
            tuple(recipe.view_half_turns)
            if recipe.coordinate_mode == "reference"
            else (False,)
        )
        if not turns or any(type(turn) is not bool for turn in turns):
            raise ValueError("view_half_turns must be nonempty booleans")
        return cls(
            resolver.resolve(PathRole.OUTPUT, cfg.scene),
            resolver.resolve(PathRole.OUTPUT, cfg.court),
            evaluation_output(resolver, cfg.output, "tennis_scene"),
            turns,
            RefinementSettings.from_config(recipe.refinement),
            OmegaConf.to_container(recipe.refinement, resolve=True),
        )


@dataclass(frozen=True)
class CourtProbeConfig:
    runtime: DatasetBuildConfig
    build: DictConfig
    samples: int
    margins: tuple[float, ...]
    checkpoints: tuple[str, ...]

    @classmethod
    def from_config(cls, cfg: DictConfig) -> CourtProbeConfig:
        _mapping(
            cfg,
            {
                "paths": ConfigField.of(dict),
                "recipe": ConfigField.of(str),
                "overrides": ConfigField.sequence(ConfigField.of(str)),
                "samples": ConfigField.of(int),
                "margins": ConfigField.sequence(ConfigField.of(float, int)),
                "checkpoints": ConfigField.sequence(ConfigField.of(str)),
                "output": ConfigField.of(str),
            },
            "court_probe",
        )
        if (
            cfg.samples <= 0
            or not cfg.margins
            or any(not math.isfinite(m) or m < 0 for m in cfg.margins)
        ):
            raise ValueError("samples must be positive and margins finite nonnegative")
        if (
            len(set(cfg.margins)) != len(cfg.margins)
            or not cfg.checkpoints
            or len(set(cfg.checkpoints)) != len(cfg.checkpoints)
        ):
            raise ValueError("margins and checkpoints must be unique and nonempty")
        resolver = resolver_for(cfg)
        output = evaluation_output(resolver, cfg.output, "tennis_scene")
        build = compose_recipe(cfg.recipe, resolver, list(cfg.overrides))
        build.output_dir = cfg.output
        runtime = DatasetBuildConfig.from_config(build)
        if len(runtime.clip_ids) != 1 or runtime.output != output:
            raise ValueError("Court probe requires exactly one configured clip")
        for checkpoint in cfg.checkpoints:
            resolver.resolve(PathRole.OUTPUT, checkpoint)
        return cls(
            runtime,
            build,
            cfg.samples,
            tuple(float(m) for m in cfg.margins),
            tuple(cfg.checkpoints),
        )


def validate_vitpose(cfg: DictConfig) -> None:
    ViTPoseBenchmarkConfig.from_config(cfg)


def validate_refinement(cfg: DictConfig) -> None:
    RefinementEvaluationConfig.from_config(cfg)


def validate_court(cfg: DictConfig) -> None:
    CourtProbeConfig.from_config(cfg)
