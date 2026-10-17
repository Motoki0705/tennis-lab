"""Resolve every decision before opening data or writing training artifacts."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.data.windows.hybrid import AbsencePolicy
from src.tasks.ball_detection.models.mdd import MDDHybridConfig
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.hydra import register_boundary_validator
from src.utils.paths import PROJECT_ROOT


def validate_loss(loss: dict[str, Any]) -> None:
    if set(loss) != {"heatmap_weight", "presence_weight", "sigma_ratio", "focal_gamma"}:
        raise ValueError("Declare both hybrid loss weights, sigma_ratio and focal_gamma")
    for key, value in loss.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "focal_gamma" and value == 0):
            raise ValueError(f"Invalid hybrid loss parameter: {key}")


def validate_threshold(value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("presence_threshold must be explicitly set in (0,1)")


def resolve_recipe(cfg: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict):
        raise ValueError("Hybrid recipe must be a mapping")
    expected = {"paths", "run", "model", "manifest", "initialization", "resume", "stop_after_epoch", "presence_threshold",
                "loss", "freeze", "absence", "epochs", "windows_per_epoch", "batch_size", "learning_rate",
                "encoder_lr_ratio", "query_lr_ratio", "warmup_updates", "seed", "selection_scope", "device", "runtime"}
    if set(raw) != expected:
        raise ValueError(f"Hybrid recipe fields differ: {set(raw) ^ expected}")
    config: dict[str, Any] = dict(raw)
    model = dict(config["model"])
    if model.pop("name", None) != "mdd_hybrid":
        raise ValueError("Hybrid stage requires model.name=mdd_hybrid")
    if set(model) != set(MDDHybridConfig.__dataclass_fields__):
        raise ValueError("Hybrid recipe must declare every model field")
    for key in ("stem_channels", "mixed_channels", "residual_blocks"):
        model[key] = tuple(model[key])
    MDDHybridConfig(**model)
    config["model"] = model
    validate_threshold(config["presence_threshold"])
    validate_loss(config["loss"])
    AbsencePolicy.from_mapping(config["absence"])
    for key in ("epochs", "windows_per_epoch", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if set(config["freeze"]) != {"encoder_epochs", "query_epochs"}:
        raise ValueError("Declare encoder and query freeze epochs separately")
    for value in config["freeze"].values():
        if type(value) is not int or not 0 <= value <= config["epochs"]:
            raise ValueError("Freeze epochs must be integers from zero to the full epoch budget")
    for key in ("learning_rate", "encoder_lr_ratio", "query_lr_ratio"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid learning rate: {key}")
    total = config["epochs"] * math.ceil(config["windows_per_epoch"] / config["batch_size"])
    if type(config["warmup_updates"]) is not int or not 0 <= config["warmup_updates"] < total:
        raise ValueError("Invalid warmup budget")
    stop = config["stop_after_epoch"]
    if stop is not None and (type(stop) is not int or not 0 <= stop < config["epochs"]):
        raise ValueError("Diagnostic stop lies outside the original epoch budget")
    initial = config["initialization"]
    if set(initial) != {"mode", "checkpoint"} or initial["mode"] not in {"query", "continual"}:
        raise ValueError("Initialization must explicitly select query or continual and its checkpoint")
    roots = RuntimePathRoots.from_mapping(config.pop("paths"), repository_root=PROJECT_ROOT)
    resolver = PathResolver(roots)
    run = config.pop("run")
    if not isinstance(run, dict) or set(run) != {"output_dir"} or not isinstance(run["output_dir"], str):
        raise ValueError("Hybrid run must declare only a relative output_dir")
    config["output"] = str(resolver.resolve(PathRole.OUTPUT, run["output_dir"]))
    for name, role in (("manifest", PathRole.ARTIFACT), ("resume", PathRole.OUTPUT)):
        value = config[name]
        if name == "resume" and value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a role-relative path")
        config[name] = str(resolver.resolve(role, value))
    if not isinstance(initial["checkpoint"], str):
        raise ValueError("Initialization checkpoint must be a checkpoint-relative path")
    initial["checkpoint"] = str(resolver.resolve(PathRole.CHECKPOINT, initial["checkpoint"]))
    if config["device"] not in {"cpu", "cuda"} or config["selection_scope"] not in {"common", "full"}:
        raise ValueError("Explicit CPU/CUDA device and common/full scope required")
    if set(config["runtime"]) != set(CoordinateRuntime.__dataclass_fields__):
        raise ValueError("Hybrid runtime must declare every execution field")
    runtime = CoordinateRuntime(**config["runtime"])
    if config["device"] == "cuda" and runtime.precision != "bf16":
        raise ValueError("CUDA hybrid training requires BF16")
    return config


def validate_training_boundary(cfg: DictConfig) -> None:
    resolve_recipe(cfg)


register_boundary_validator("ball.hybrid", validate_training_boundary)
