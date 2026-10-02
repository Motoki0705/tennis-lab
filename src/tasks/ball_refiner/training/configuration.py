"""Strict, composition-owned settings for the detector-only pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.data.store import SOURCES
from src.tasks.ball_refiner.refiner_2d.config import Refiner2DConfig, parse_model_config
from src.tasks.base.configuration import CompileConfig
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

TRAINING_SCHEMA = StrictConfigSchema(fields={
    **{key: ConfigField((int,)) for key in (
        "batch_size", "epochs", "steps_per_epoch", "max_steps", "num_workers",
    )},
    **{key: ConfigField((int, float)) for key in ("learning_rate", "weight_decay", "gradient_clip", "gap_probability")},
    "gap_lengths": ConfigField((list,), sequence_item=ConfigField((int,))),
}, name="training")
DATA_SCHEMA = StrictConfigSchema(fields={
    "store": ConfigField((str,)), "evidence": ConfigField((str,)),
    "sources": ConfigField((list,), sequence_item=ConfigField((str,))),
    **{key: ConfigField((int,)) for key in ("window_length", "stride", "partition_seed")},
    "short_clip_policy": ConfigField((str,)),
}, name="data")
RUN_SCHEMA = StrictConfigSchema(fields={
    "output_dir": ConfigField((str,)), "device": ConfigField((str,)),
    "seed": ConfigField((int,)), "dry_run": ConfigField((bool,)),
}, name="run")
PILOT_SCHEMA = StrictConfigSchema(fields={
    "paths": ConfigField((dict,)), "model": ConfigField((dict,)), "compile": ConfigField((dict,)),
    "data": ConfigField((dict,), mapping_schema=DATA_SCHEMA),
    "training": ConfigField((dict,), mapping_schema=TRAINING_SCHEMA),
    "run": ConfigField((dict,), mapping_schema=RUN_SCHEMA),
}, name="ball_refiner.pilot")


@dataclass(frozen=True)
class PilotTrainingConfig:
    batch_size: int
    epochs: int
    steps_per_epoch: int
    max_steps: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    gap_probability: float
    gap_lengths: tuple[int, ...]


@dataclass(frozen=True)
class PilotConfig:
    model: Refiner2DConfig
    training: PilotTrainingConfig
    compilation: CompileConfig
    store: Path
    evidence: Path
    output: Path
    sources: tuple[str, ...]
    window_length: int
    stride: int
    partition_seed: int
    seed: int
    device: str
    dry_run: bool
    resolved: dict[str, Any]

    @classmethod
    def from_config(cls, config: DictConfig) -> PilotConfig:
        raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(raw, dict):
            raise TypeError("Pilot config must be a mapping")
        validated = cast(dict[str, Any], dict(PILOT_SCHEMA.validate(raw)))
        data, run, training = validated["data"], validated["run"], dict(validated["training"])
        model = parse_model_config(validated["model"])
        if model.use_pose or model.use_court or not model.use_detector:
            raise ValueError("Pilot requires explicit detector-only configuration; context is not generated")
        compilation = CompileConfig.from_mapping(validated["compile"])
        if compilation.mode != "default":
            raise ValueError("Pilot supports only compile mode=default; CUDA Graphs lifecycle is not implemented")
        training["gap_lengths"] = tuple(training["gap_lengths"])
        train = PilotTrainingConfig(**training)
        for name in ("batch_size", "epochs", "steps_per_epoch", "max_steps"):
            if getattr(train, name) < 1:
                raise ValueError(f"training.{name} must be positive")
        if train.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        for name in ("learning_rate", "weight_decay", "gradient_clip", "gap_probability"):
            if not math.isfinite(getattr(train, name)):
                raise ValueError(f"Nonfinite training.{name}")
        if train.learning_rate <= 0 or train.weight_decay < 0 or train.gradient_clip <= 0 or not 0 <= train.gap_probability <= 1:
            raise ValueError("Invalid optimizer or gap settings")
        length, stride = data["window_length"], data["stride"]
        if length < 2 or not 1 <= stride <= length or data["short_clip_policy"] != "exclude_and_report":
            raise ValueError("Require real windows and short_clip_policy=exclude_and_report")
        if not train.gap_lengths or len(set(train.gap_lengths)) != len(train.gap_lengths) or not all(0 < gap < length for gap in train.gap_lengths):
            raise ValueError("Gap lengths must be distinct, positive, and shorter than the window")
        sources = tuple(data["sources"])
        if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(SOURCES) or "meiji" not in sources:
            raise ValueError("Pilot sources must be unique known sources including Meiji validation")
        if min(run["seed"], data["partition_seed"]) < 0 or run["device"] not in ("cpu", "cuda"):
            raise ValueError("Require nonnegative seeds and explicit cpu/cuda device")
        roots = RuntimePathRoots.from_mapping(validated["paths"], repository_root=PROJECT_ROOT)
        resolver = PathResolver(roots)
        store = resolver.resolve(PathRole.DATA, data["store"])
        evidence = resolver.resolve(PathRole.CACHE, data["evidence"])
        output = resolver.resolve(PathRole.OUTPUT, run["output_dir"])
        if any(output == path or output.is_relative_to(path) or path.is_relative_to(output) for path in (store, evidence)):
            raise ValueError("Pilot output must be separate from input store/cache")
        return cls(model, train, compilation, store, evidence, output, sources, length, stride,
                   data["partition_seed"], run["seed"], run["device"], run["dry_run"], cast(dict[str, Any], raw))


def validate_training_boundary(config: DictConfig) -> None:
    PilotConfig.from_config(config)
