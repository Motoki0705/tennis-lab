"""Inference-only compatibility for historical Court checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

_LOCAL_ARTIFACT_STORE: dict[str, object] = {
    "mode": "local",
    "remote": None,
    "remote_root": None,
    "sync_interval_seconds": None,
}


def migrate_court_inference_config(config: object) -> object | None:
    """Add inference-safe defaults missing from historical training configs.

    Training entry points remain strict and require every current field. This
    migration is deliberately scoped to checkpoint inference, where artifact
    publication settings do not affect model construction or predictions.

    Returns ``None`` when no known migration is required.
    """
    if isinstance(config, DictConfig):
        run = config.get("run")
        if not isinstance(run, DictConfig) or "artifact_store" in run:
            return None
        migrated = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        migrated.run.artifact_store = deepcopy(_LOCAL_ARTIFACT_STORE)
        return migrated

    if not isinstance(config, Mapping):
        return None
    run = config.get("run")
    if not isinstance(run, Mapping) or "artifact_store" in run:
        return None
    migrated_mapping = deepcopy(dict(cast("Mapping[str, Any]", config)))
    migrated_run = deepcopy(dict(cast("Mapping[str, Any]", run)))
    migrated_run["artifact_store"] = deepcopy(_LOCAL_ARTIFACT_STORE)
    migrated_mapping["run"] = migrated_run
    return migrated_mapping


def load_court_inference_config_override(checkpoint_path: Path) -> object | None:
    """Read only checkpoint metadata needed for known inference migrations.

    ``mmap=True`` keeps tensor storages lazy while the small hyperparameter
    mapping is inspected. The checkpoint is trusted already: Lightning will
    load the same file immediately afterward to restore model weights.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    try:
        if not isinstance(checkpoint, Mapping):
            return None
        hyper_parameters = checkpoint.get("hyper_parameters")
        if not isinstance(hyper_parameters, Mapping):
            return None
        return migrate_court_inference_config(hyper_parameters.get("config"))
    finally:
        del checkpoint


__all__ = [
    "load_court_inference_config_override",
    "migrate_court_inference_config",
]
