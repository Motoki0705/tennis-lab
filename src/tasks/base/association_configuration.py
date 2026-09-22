"""Validation of shared association data fields, called by task boundaries."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import OmegaConf

from src.tasks.base.configuration import TrainingRuntimeConfig, exact_config_mapping
from src.tasks.base.data.observation_tracking import ObservationTrackingConfig
from src.tasks.base.models.view_association import ViewQueryModelConfig
from src.utils.paths import PROJECT_ROOT


def validate_association_configuration(
    config: Any, *, model_name: str
) -> ViewQueryModelConfig:
    root = OmegaConf.to_container(config, resolve=True)
    required = {
        "court_keypoints",
        "paths",
        "model",
        "data",
        "training",
        "loss",
        "metrics",
        "run",
    }
    exact_config_mapping(
        root, path="association training", required_keys=frozenset(required)
    )
    if (
        config.model.name != model_name
        or config.court_keypoints.selector != "camera_view_v2"
    ):
        raise ValueError("Association requires its task model and raw camera_view_v2")
    model = ViewQueryModelConfig.from_mapping(
        {k: v for k, v in config.model.items() if k != "name"}
    )
    data = config.data
    keys = {
        "augmentation",
        "association",
        "backend",
        "scene_dir",
        "camera_mode",
        "camera_candidates",
        "num_views_range",
        "seq_len_range",
        "batch_size",
        "num_workers",
        "pin_memory",
        "evaluation_reference_camera_id",
        "validation_cache_size",
        "persistent_workers",
        "prefetch_factor",
    }
    exact_config_mapping(data, path="association data", required_keys=frozenset(keys))
    if data.backend != "default":
        raise ValueError("Association currently requires a fixed dataset")
    for name in ("validation_cache_size", "prefetch_factor"):
        value = data[name]
        if type(value) is not int or value < (1 if name == "prefetch_factor" else 0):
            raise ValueError(f"Invalid data.{name}")
    if type(data.persistent_workers) is not bool:
        raise ValueError("data.persistent_workers must be boolean")
    ObservationTrackingConfig.from_mapping(data.association)
    exact_config_mapping(
        config.loss,
        path="association loss",
        required_keys=frozenset({"identity_weight", "side_weight"}),
    )
    for value in config.loss.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Association loss weights must be finite and nonnegative")
    if sum(config.loss.values()) <= 0:
        raise ValueError("At least one association loss must be enabled")
    exact_config_mapping(
        config.metrics,
        path="association metrics",
        required_keys=frozenset({"side_threshold"}),
    )
    if not 0 < config.metrics.side_threshold < 1:
        raise ValueError("side_threshold must be between 0 and 1")
    TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    return model
