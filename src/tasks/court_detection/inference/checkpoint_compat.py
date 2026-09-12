"""Inference-only compatibility for historical Court checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.configuration import LINE_TARGET_SCHEMA

_LOCAL_ARTIFACT_STORE: dict[str, object] = {
    "mode": "local",
    "remote": None,
    "remote_root": None,
    "sync_interval_seconds": None,
}
_LEGACY_LINE_TARGET_SCHEMAS = frozenset(
    {
        "court_line_binary_75mm_150mm_v2",
    }
)


def _migrate_plain_config(config: MutableMapping[str, Any]) -> bool:
    changed = False

    run = config.get("run")
    if isinstance(run, MutableMapping) and "artifact_store" not in run:
        run["artifact_store"] = deepcopy(_LOCAL_ARTIFACT_STORE)
        changed = True

    data = config.get("data")
    processing = data.get("processing") if isinstance(data, Mapping) else None
    targets = processing.get("targets") if isinstance(processing, Mapping) else None
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, MutableMapping) or target.get("kind") != "line":
                continue
            if target.get("target_schema") in _LEGACY_LINE_TARGET_SCHEMAS:
                target["target_schema"] = LINE_TARGET_SCHEMA
                changed = True

    return changed


def migrate_court_inference_config(config: object) -> object | None:
    """Normalize historical training metadata for model-only inference.

    Training entry points remain strict and require current configuration.
    This migration is scoped to checkpoint inference and only changes fields
    irrelevant to model weights or forward semantics:

    - missing artifact publication settings become a local-only store;
    - retired line-target generator schema names become the current one.

    Returns ``None`` when no known migration is required.
    """
    if isinstance(config, DictConfig):
        raw = OmegaConf.to_container(config, resolve=False)
        if not isinstance(raw, dict):
            return None
        migrated = cast("dict[str, Any]", raw)
        if not _migrate_plain_config(migrated):
            return None
        return OmegaConf.create(migrated)

    if not isinstance(config, Mapping):
        return None
    migrated_mapping = deepcopy(dict(cast("Mapping[str, Any]", config)))
    if not _migrate_plain_config(migrated_mapping):
        return None
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
