"""Task-owned losses, update budget, and strict training boundary."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from src.tasks.ball_refiner_3d.configuration.core import parse_section, resolved_config
from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.configuration.model import (
    DiscriminatorConfig,
    ModelConfig,
)
from src.tasks.base.configuration import TrainingRuntimeConfig
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class EventConfig:
    weight: float
    sigma_frames: float

    def __post_init__(self) -> None:
        if self.weight <= 0 or self.sigma_frames <= 0:
            raise ValueError("Event weight and Gaussian sigma must be positive")


@dataclass(frozen=True)
class GANConfig:
    enabled: bool
    schedule_enabled: bool
    target_weight: float
    start_step: int
    warmup_steps: int

    def __post_init__(self) -> None:
        if self.target_weight < 0 or self.start_step < 0 or self.warmup_steps < 1:
            raise ValueError("Invalid GAN target or transition schedule")
        if self.enabled and self.target_weight == 0:
            raise ValueError("Enabled GAN requires positive target_weight")


@dataclass(frozen=True)
class ReconstructionConfig:
    enabled: bool
    initial_weight: float
    final_weight: float
    start_step: int
    decay_steps: int

    def __post_init__(self) -> None:
        if (
            self.initial_weight <= 0
            or not 0 <= self.final_weight <= self.initial_weight
        ):
            raise ValueError(
                "Reconstruction weights must satisfy 0 <= final <= positive initial"
            )
        if self.start_step < 0 or self.decay_steps < 1:
            raise ValueError("Invalid reconstruction decay schedule")


def parse_gan(raw: dict[str, Any]) -> tuple[GANConfig, DiscriminatorConfig]:
    if set(raw) != {
        "enabled",
        "schedule_enabled",
        "target_weight",
        "transition",
        "warmup_steps",
        "discriminator",
        "warmup_epochs",
        "generator_gradient_clip_val",
        "discriminator_gradient_clip_val",
    } or set(raw["transition"]) != {"start_step", "start_epoch"}:
        raise ValueError(
            "Require explicit GAN transition and discriminator configuration"
        )
    return parse_section(
        GANConfig,
        {
            key: raw[key]
            for key in ("enabled", "schedule_enabled", "target_weight", "warmup_steps")
        }
        | {"start_step": raw["transition"]["start_step"]},
    ), parse_section(DiscriminatorConfig, raw["discriminator"])


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    evaluate_every: int
    log_every: int
    cpu_threads: int

    def __post_init__(self) -> None:
        if (
            min(
                self.steps,
                self.batch_size,
                self.evaluate_every,
                self.log_every,
                self.cpu_threads,
            )
            < 1
        ):
            raise ValueError("Training counts must be positive")
        if min(self.learning_rate, self.gradient_clip) <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer/GAN settings")


@dataclass(frozen=True)
class RefinerTrainingSettings:
    raw: dict[str, Any]
    runtime: TrainingRuntimeConfig
    dataset: Path
    model: ModelConfig
    updates: TrainingConfig
    augmentation: CorruptionConfig
    event: EventConfig
    reconstruction: ReconstructionConfig
    gan: GANConfig
    discriminator: DiscriminatorConfig


def training_config(config: DictConfig) -> RefinerTrainingSettings:
    raw, resolver = resolved_config(
        config, {"paths", "data", "augmentation", "model", "training", "loss", "run"}
    )
    expected_training = {field.name for field in fields(TrainingConfig)} | {
        "trainer",
        "warmup_steps",
        "warmup_epochs",
        "min_lr",
        "steps_per_epoch",
        "optimizer",
        "checkpoint",
        "early_stopping",
        "lr_monitor",
        "qualitative_logging",
        "gan",
        "compile",
        "matmul_precision",
        "allow_tf32",
    }
    if set(raw["training"]) != expected_training or set(raw["loss"]) != {
        "event",
        "reconstruction",
    }:
        raise ValueError("Unknown or missing training/loss settings")
    if set(raw["data"]) != {
        "dataset",
        "evaluation_event_probability",
        "evaluation_seed",
        "num_workers",
        "pin_memory",
    }:
        raise ValueError("Unknown or missing data settings")
    updates = parse_section(
        TrainingConfig,
        {field.name: raw["training"][field.name] for field in fields(TrainingConfig)},
    )
    model = parse_section(ModelConfig, raw["model"])
    augmentation = parse_section(CorruptionConfig, raw["augmentation"])
    event = parse_section(EventConfig, raw["loss"]["event"])
    reconstruction = parse_section(ReconstructionConfig, raw["loss"]["reconstruction"])
    gan, discriminator = parse_gan(raw["training"]["gan"])
    runtime = TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    trainer = runtime.training.trainer
    if runtime.run.gpus > 1:
        raise ValueError(
            "Refiner window sampling currently supports one training device"
        )
    if runtime.training.optimizer.steps_per_epoch != updates.evaluate_every:
        raise ValueError("steps_per_epoch must equal evaluate_every")
    raw_gan = raw["training"]["gan"]
    if (
        raw_gan["transition"]["start_epoch"] != gan.start_step // updates.evaluate_every
        or raw_gan["warmup_epochs"]
        != (gan.warmup_steps + updates.evaluate_every - 1) // updates.evaluate_every
    ):
        raise ValueError(
            "GAN epoch fields must be derived from the generator update schedule"
        )
    if (
        raw_gan["generator_gradient_clip_val"] != updates.gradient_clip
        or raw_gan["discriminator_gradient_clip_val"] != updates.gradient_clip
    ):
        raise ValueError("GAN clipping must match the task gradient_clip setting")
    expected_epochs = (
        updates.steps + updates.evaluate_every - 1
    ) // updates.evaluate_every
    if trainer.max_epochs != expected_epochs or trainer.check_val_every_n_epoch != 1:
        raise ValueError(
            "Epochs must represent evaluate_every generator updates, including the final partial epoch"
        )
    if (
        trainer.accumulate_grad_batches != 1
        or trainer.reload_dataloaders_every_n_epochs != 1
    ):
        raise ValueError(
            "Refiner requires one generator update per batch and a refreshed loader each epoch"
        )
    if trainer.precision != "32-true" or trainer.gradient_clip_val is not None:
        raise ValueError(
            "Refiner uses explicit float32 optimization and task-owned gradient clipping"
        )
    if (
        runtime.training.optimizer.warmup_steps != 0
        or runtime.training.optimizer.warmup_epochs is not None
    ):
        raise ValueError(
            "Refiner cosine LR is indexed by generator updates without optimizer warmup"
        )
    if (
        runtime.training.early_stopping.enabled
        or runtime.training.qualitative_logging.enabled
    ):
        raise ValueError("Use the fixed update budget and the dedicated Dataset Review")
    if (
        not runtime.training.checkpoint.enabled
        or runtime.training.checkpoint.monitor != "val/rmse"
        or runtime.training.checkpoint.mode != "min"
        or runtime.training.checkpoint.save_top_k != 1
        or not runtime.training.checkpoint.save_last
    ):
        raise ValueError(
            "Require validation RMSE best selection and a separate last checkpoint"
        )
    if model.architecture == "flow" and gan.enabled:
        raise ValueError("GAN is only supported by direct regression")
    if (
        gan.enabled
        and gan.schedule_enabled
        and gan.start_step + gan.warmup_steps > updates.steps
    ):
        raise ValueError(
            "GAN transition and warmup must finish within the update budget"
        )
    if (
        reconstruction.enabled
        and reconstruction.start_step + reconstruction.decay_steps > updates.steps
    ):
        raise ValueError("Reconstruction schedule must finish within the update budget")
    if (
        reconstruction.enabled
        and reconstruction.final_weight == 0
        and (
            not gan.enabled
            or (
                gan.schedule_enabled
                and gan.start_step
                >= reconstruction.start_step + reconstruction.decay_steps
            )
        )
    ):
        raise ValueError("Zero reconstruction weight requires an active GAN objective")
    if (
        discriminator.hidden_dim != model.width
        or discriminator.max_seq_len != model.window_length
    ):
        raise ValueError("Discriminator width/window must match the generator")
    if (
        not 0 <= raw["data"]["evaluation_event_probability"] <= 1
        or type(raw["data"]["evaluation_seed"]) is not int
        or raw["data"]["evaluation_seed"] < 0
    ):
        raise ValueError("Invalid fixed evaluation corruption/seed")
    if (
        raw["data"]["num_workers"] != 0
        or type(raw["data"]["num_workers"]) is not int
        or type(raw["data"]["pin_memory"]) is not bool
    ):
        raise ValueError(
            "The stateful window sampler requires num_workers=0 and explicit pin_memory"
        )
    dataset = resolver.resolve(PathRole.DATA, raw["data"]["dataset"])
    output = runtime.run.output_dir
    if (
        output == dataset
        or output.is_relative_to(dataset)
        or dataset.is_relative_to(output)
    ):
        raise ValueError("Training output must be separate from the dataset")
    return RefinerTrainingSettings(
        raw,
        runtime,
        dataset,
        model,
        updates,
        augmentation,
        event,
        reconstruction,
        gan,
        discriminator,
    )


def validate_training(config: DictConfig) -> None:
    training_config(config)
