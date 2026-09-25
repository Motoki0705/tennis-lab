"""Strict PLCS-only training configuration for independent person models."""

from __future__ import annotations

import math
from typing import Any

from omegaconf import OmegaConf

from src.tasks.base.configuration import (
    BaseTrainingConfig,
    TrainingRuntimeConfig,
    exact_config_mapping,
)
from src.tasks.plcs.model_io.person_association import MODEL_CONTRACTS, REID_MODEL
from src.tasks.plcs.models.person_tokens import PersonModelConfig
from src.utils.paths import PROJECT_ROOT


def validate_person_config(config: Any) -> PersonModelConfig:
    root = OmegaConf.to_container(config, resolve=True)
    exact_config_mapping(root, path="PLCS person training", required_keys=frozenset({"court_keypoints", "paths", "model", "data", "training", "loss", "metrics", "run"}))
    name = str(config.model.name)
    if name not in MODEL_CONTRACTS or config.court_keypoints.selector != "camera_view_v2":
        raise ValueError("PLCS person models require raw camera_view_v2 observations")
    model = PersonModelConfig.from_mapping({k: v for k, v in config.model.items() if k != "name"})
    exact_config_mapping(config.data, path="PLCS person data", required_keys=frozenset({
        "augmentation", "backend", "scene_dir", "camera_mode", "camera_candidates", "num_views_range", "seq_len_range",
        "target_fps", "batch_size", "num_workers", "pin_memory", "evaluation_reference_camera_id",
        "validation_cache_size", "persistent_workers", "prefetch_factor"}))
    if config.data.backend != "default":
        raise ValueError("PLCS person training requires a fixed scene dataset")
    for field, minimum in (("validation_cache_size", 0), ("prefetch_factor", 1)):
        if type(config.data[field]) is not int or config.data[field] < minimum:
            raise ValueError(f"Invalid data.{field}")
    if type(config.data.persistent_workers) is not bool or not math.isfinite(config.data.target_fps) or config.data.target_fps <= 0:
        raise ValueError("Invalid loader/timebase configuration")
    aug = config.data.augmentation
    exact_config_mapping(aug, path="PLCS person augmentation", required_keys=frozenset({"pose_noise", "court_noise", "joint_dropout", "court_dropout", "track_dropout"}))
    for key, value in aug.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or ("noise" not in key and value > 1):
            raise ValueError(f"Invalid augmentation.{key}")
    reid = name == REID_MODEL
    exact_config_mapping(config.loss, path="PLCS person loss", required_keys=frozenset({"temperature", "margin"} if reid else {"weight"}))
    exact_config_mapping(config.metrics, path="PLCS person metrics", required_keys=frozenset({"cosine_threshold"} if reid else {"side_threshold"}))
    for value in config.loss.values():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Person loss settings must be finite numbers")
    if reid:
        if config.loss.temperature <= 0 or not -1 < config.loss.margin < 1:
            raise ValueError("Invalid Re-ID loss settings")
        if not -1 < config.metrics.cosine_threshold < 1:
            raise ValueError("Invalid Re-ID decision thresholds")
    elif config.loss.weight <= 0 or not 0 < config.metrics.side_threshold < 1:
        raise ValueError("Invalid side loss/threshold")
    from src.tasks.plcs.training.mcmc import MCMCConfig

    mcmc = MCMCConfig.from_dict(dict(config.training.mcmc))
    if mcmc.enabled:
        raise ValueError("MCMC is not supported by the independent person models")
    BaseTrainingConfig.from_mapping({key: value for key, value in root["training"].items() if key != "mcmc"})
    TrainingRuntimeConfig.from_config(config, repository_root=PROJECT_ROOT)
    return model
