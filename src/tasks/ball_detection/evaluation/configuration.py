"""Checkpoint and dataset configuration resolution for evaluation jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_detection.evaluation.contracts import DatasetSpec, MetricsSpec
from src.utils.paths import resolve_project_path

_CONFIG_ROOT = Path("src/tasks/ball_detection/configs")


def read_checkpoint_config(checkpoint_path: str | Path) -> DictConfig:
    """Read the saved Hydra config without constructing the model."""
    resolved = resolve_project_path(checkpoint_path)
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
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Checkpoint does not contain hyper_parameters.")
    config = hyper_parameters.get("config")
    if config is None:
        raise ValueError("Checkpoint hyper_parameters do not contain config.")
    return OmegaConf.create(config)


def validate_checkpoint_model_name(
    checkpoint_config: DictConfig,
    *,
    expected_model_name: str,
) -> str:
    """Fail early when a manifest entry names the wrong architecture."""
    actual_name = str((checkpoint_config.get("model", {}) or {}).get("name", ""))
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
) -> DictConfig:
    """Combine checkpoint model settings with a fixed evaluation dataset."""
    data_config = load_data_config(
        dataset_spec.config,
        overrides=dataset_spec.overrides,
    )
    evaluation_config = OmegaConf.create(
        OmegaConf.to_container(checkpoint_config, resolve=True)
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
) -> DictConfig:
    """Load one ball-detection data config including its augmentation default."""
    candidate = Path(config_name_or_path)
    if candidate.suffix:
        config_path = resolve_project_path(candidate)
    else:
        config_path = resolve_project_path(
            _CONFIG_ROOT / "data" / f"{config_name_or_path}.yaml"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"Data config not found: {config_path}")

    loaded = OmegaConf.load(config_path)
    container = OmegaConf.to_container(loaded, resolve=False)
    if not isinstance(container, dict):
        raise TypeError(f"Data config must be a mapping: {config_path}")
    defaults = container.pop("defaults", [])
    data_config = OmegaConf.create(container)

    augmentation_name = _augmentation_default_name(defaults)
    if augmentation_name is not None:
        augmentation_path = config_path.parent / "augmentation" / (
            f"{augmentation_name}.yaml"
        )
        if not augmentation_path.is_file():
            raise FileNotFoundError(
                f"Data augmentation config not found: {augmentation_path}"
            )
        data_config = OmegaConf.merge(
            {"augmentation": OmegaConf.load(augmentation_path)},
            data_config,
        )
    if overrides:
        data_config = OmegaConf.merge(data_config, OmegaConf.create(overrides))
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
