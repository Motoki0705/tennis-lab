"""Explicit validation-only diagnostic configuration, independent of training input IO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_refiner.evaluation.hdr import validate_hdr_settings
from src.tasks.base.configuration import CompileConfig
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    RuntimePathRoots,
    StrictConfigSchema,
)
from src.utils.paths import PROJECT_ROOT

EVALUATE_SCHEMA = StrictConfigSchema(fields={
    "training_run": ConfigField((str,)), "partition": ConfigField((str,)),
    "levels": ConfigField((list,), sequence_item=ConfigField((int, float))),
    **{key: ConfigField((int,)) for key in ("samples", "seed", "chunk_size", "bootstrap_repetitions")},
    "bootstrap_confidence": ConfigField((int, float)),
}, name="evaluate")
SCHEMA = StrictConfigSchema(fields={
    "paths": ConfigField((dict,)), "compile": ConfigField((dict,)),
    "evaluate": ConfigField((dict,), mapping_schema=EVALUATE_SCHEMA),
    "run": ConfigField((dict,), mapping_schema=StrictConfigSchema(fields={
        "output_dir": ConfigField((str,)), "device": ConfigField((str,)),
    }, name="run")),
}, name="ball_refiner.evaluate_pilot")


@dataclass(frozen=True)
class EvaluationConfig:
    training_run: Path
    output: Path
    partition: str
    levels: tuple[float, ...]
    samples: int
    seed: int
    chunk_size: int
    bootstrap_repetitions: int
    bootstrap_confidence: float
    device: str
    compilation: CompileConfig
    resolved: dict[str, Any]

    @classmethod
    def from_config(cls, config: DictConfig) -> EvaluationConfig:
        raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(raw, dict):
            raise TypeError("Evaluation config must be a mapping")
        data = cast(dict[str, Any], dict(SCHEMA.validate(raw)))
        evaluate, run = data["evaluate"], data["run"]
        levels = tuple(float(value) for value in evaluate["levels"])
        validate_hdr_settings(levels, evaluate["samples"], evaluate["seed"], evaluate["chunk_size"])
        confidence = float(evaluate["bootstrap_confidence"])
        if evaluate["bootstrap_repetitions"] < 2 or not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("Invalid bootstrap settings")
        if evaluate["partition"] not in ("selection", "calibration"):
            raise ValueError("Pilot diagnostics allow only selection/calibration validation, never test")
        if run["device"] not in ("cpu", "cuda"):
            raise ValueError("Evaluation device must be explicitly cpu or cuda")
        compilation = CompileConfig.from_mapping(data["compile"])
        if compilation.mode != "default":
            raise ValueError("Only default compile mode has a supported inference lifecycle")
        roots = RuntimePathRoots.from_mapping(data["paths"], repository_root=PROJECT_ROOT)
        resolver = PathResolver(roots)
        training = resolver.resolve(PathRole.ARTIFACT, evaluate["training_run"])
        output = resolver.resolve(PathRole.OUTPUT, run["output_dir"])
        if training == output or output.is_relative_to(training) or training.is_relative_to(output):
            raise ValueError("Evaluation output must be separate from the training input")
        return cls(training, output, evaluate["partition"], levels, evaluate["samples"], evaluate["seed"],
                   evaluate["chunk_size"], evaluate["bootstrap_repetitions"], confidence, run["device"],
                   compilation, cast(dict[str, Any], raw))


def validate_evaluation_boundary(config: DictConfig) -> None:
    EvaluationConfig.from_config(config)
