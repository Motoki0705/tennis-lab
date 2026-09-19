"""Strict held-out evaluation boundary; checkpoint selection is external."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.tasks.blcs.training.runner import BLCSTrainingRunner
from src.utils.configuration import (
    ConfigField,
    PathResolver,
    PathRole,
    StrictConfigSchema,
)
from src.utils.configuration.output_layout import task_output_path


@dataclass(frozen=True)
class RealEvaluationConfig:
    training: DictConfig
    resolver: PathResolver
    checkpoint: Path
    output: Path
    device: str

    @classmethod
    def from_config(cls, cfg: DictConfig) -> RealEvaluationConfig:
        value = OmegaConf.to_container(
            cfg.evaluation, resolve=True, throw_on_missing=True
        )
        if not isinstance(value, dict):
            raise TypeError("evaluation must be a mapping")
        StrictConfigSchema(
            name="blcs.real_evaluation",
            fields={
                "checkpoint": ConfigField.of(str),
                "device": ConfigField.of(str),
            },
        ).validate(value)
        if cfg.evaluation.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        training = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        if not isinstance(training, DictConfig):
            raise TypeError("training configuration must be a mapping")
        del training.evaluation
        runtime = BLCSTrainingRunner().validate_runtime_config(training)
        parts = str(training.run.output_dir).split("/")
        if len(parts) != 4 or parts[:2] != ["blcs", "evaluate"]:
            raise ValueError(
                "run.output_dir must be blcs/evaluate/<experiment>/<run-id>"
            )
        task_output_path(*parts)
        output = runtime.resolver.resolve(PathRole.OUTPUT, training.run.output_dir)
        if output.exists():
            raise FileExistsError(f"Choose a new evaluation run: {output}")
        return cls(
            training,
            runtime.resolver,
            runtime.resolver.resolve(PathRole.CHECKPOINT, cfg.evaluation.checkpoint),
            output,
            cfg.evaluation.device,
        )


def validate_real_evaluation(cfg: DictConfig) -> None:
    RealEvaluationConfig.from_config(cfg)
