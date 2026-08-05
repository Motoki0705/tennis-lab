"""Checkpoint and dataset configuration resolution for evaluation jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.evaluation.contracts import DatasetSpec, MetricsSpec
from src.utils.configuration import PathResolver, PathRole

_CONFIG_ROOT = Path("src/tasks/ball_detection/configs")


def _require_dict_config(value: object, *, context: str) -> DictConfig:
    if not isinstance(value, DictConfig):
        raise TypeError(f"{context} must resolve to a mapping.")
    return value


def read_checkpoint_config(checkpoint_path: str | Path) -> DictConfig:
    """Read the saved Hydra config without constructing the model."""
    resolved = Path(checkpoint_path)
    if not resolved.is_absolute():
        raise ValueError("Checkpoint path must be absolute.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    try:
        checkpoint = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except RuntimeError as error:
        if "mmap" not in str(error).lower():
            raise
        checkpoint = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    if not isinstance(checkpoint, dict):
        raise TypeError("Lightning checkpoint root must be a mapping.")
    try:
        hyper_parameters = checkpoint["hyper_parameters"]
    except KeyError as error:
        raise ValueError("Checkpoint does not contain hyper_parameters.") from error
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Checkpoint does not contain hyper_parameters.")
    try:
        config = hyper_parameters["config"]
    except KeyError as error:
        raise ValueError(
            "Checkpoint hyper_parameters do not contain config."
        ) from error
    return _require_dict_config(
        OmegaConf.create(config), context="Checkpoint hyper_parameters.config"
    )


def validate_checkpoint_model_name(
    checkpoint_config: DictConfig,
    *,
    expected_model_name: str,
) -> str:
    """Fail early when a manifest entry names the wrong architecture."""
    actual_name = str(checkpoint_config.model.name)
    if actual_name != expected_model_name:
        raise ValueError(
            "Checkpoint/model config mismatch: "
            f"expected model.name={expected_model_name!r}, got {actual_name!r}."
        )
    return actual_name


def build_evaluation_config(
    *,
    checkpoint_config: DictConfig,
    dataset_spec: DatasetSpec,
    metrics_spec: MetricsSpec,
    resolver: PathResolver,
) -> DictConfig:
    """Combine checkpoint model settings with a fixed evaluation dataset."""
    data_config = load_data_config(
        dataset_spec.config,
        overrides=dataset_spec.overrides,
        resolver=resolver,
    )
    evaluation_config = _require_dict_config(
        OmegaConf.create(OmegaConf.to_container(checkpoint_config, resolve=True)),
        context="Checkpoint evaluation config",
    )
    evaluation_config.data = data_config
    evaluation_config.metrics = OmegaConf.create(
        {
            "peak_threshold": metrics_spec.peak_threshold,
            "ball_distance_threshold": metrics_spec.ball_distance_threshold,
            "nms_kernel": metrics_spec.nms_kernel,
            "max_predictions_per_frame": metrics_spec.max_predictions_per_frame,
            "subpixel_refine": metrics_spec.subpixel_refine,
        }
    )
    return evaluation_config


def load_data_config(
    config_name_or_path: str,
    *,
    overrides: dict[str, Any] | None = None,
    resolver: PathResolver,
) -> DictConfig:
    """Load one ball-detection data config including its augmentation default."""
    candidate = Path(config_name_or_path)
    if (
        not config_name_or_path.strip()
        or config_name_or_path != config_name_or_path.strip()
        or candidate.name != config_name_or_path
        or candidate.suffix
    ):
        raise ValueError(
            "Dataset config must be a canonical config name without a path or suffix."
        )
    config_path = resolver.resolve(
        PathRole.PROJECT, _CONFIG_ROOT / "data" / f"{config_name_or_path}.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Data config not found: {config_path}")

    loaded = _require_dict_config(
        OmegaConf.load(config_path), context=f"Data config {config_path}"
    )
    container = OmegaConf.to_container(loaded, resolve=False)
    if not isinstance(container, dict):
        raise TypeError(f"Data config must be a mapping: {config_path}")
    if "defaults" not in container:
        raise ValueError(f"Data config must declare defaults: {config_path}")
    defaults = container.pop("defaults")
    data_config = _require_dict_config(
        OmegaConf.create(container), context=f"Data config {config_path}"
    )

    augmentation_name = _augmentation_default_name(defaults)
    if augmentation_name is not None:
        augmentation_path = (
            config_path.parent / "augmentation" / (f"{augmentation_name}.yaml")
        )
        if not augmentation_path.is_file():
            raise FileNotFoundError(
                f"Data augmentation config not found: {augmentation_path}"
            )
        augmentation = _require_dict_config(
            OmegaConf.load(augmentation_path),
            context=f"Data augmentation config {augmentation_path}",
        )
        data_config = _require_dict_config(
            OmegaConf.merge({"augmentation": augmentation}, data_config),
            context="Merged data config",
        )
    if overrides:
        data_config = _require_dict_config(
            OmegaConf.merge(data_config, OmegaConf.create(overrides)),
            context="Data config with manifest overrides",
        )
    return data_config


def _augmentation_default_name(defaults: object) -> str | None:
    if not isinstance(defaults, list):
        raise TypeError("data config defaults must be a list.")
    for entry in defaults:
        if isinstance(entry, dict) and "augmentation" in entry:
            return str(entry["augmentation"])
    return None


__all__ = [
    "build_evaluation_config",
    "load_data_config",
    "read_checkpoint_config",
    "validate_checkpoint_model_name",
]
