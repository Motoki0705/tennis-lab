"""Inference-only compatibility for historical Court checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeAlias, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.court_detection.configuration import LINE_TARGET_SCHEMA
from src.tasks.court_detection.models.checkpoint_dense_head import (
    build_checkpoint_dense_heads,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)

CourtInferenceConfig: TypeAlias = dict[str, Any] | DictConfig

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
_REQUIRED_PATH_ROOTS = frozenset(
    {
        "project_root",
        "data_root",
        "checkpoint_root",
        "artifact_root",
        "output_root",
        "cache_root",
        "external_asset_root",
    }
)


def _normalized_runtime_path_roots(
    value: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    missing = _REQUIRED_PATH_ROOTS.difference(value)
    extra = set(value).difference(_REQUIRED_PATH_ROOTS)
    if missing or extra:
        raise ValueError(
            "Court inference runtime roots must contain exactly "
            f"{sorted(_REQUIRED_PATH_ROOTS)}; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )
    normalized: dict[str, str] = {}
    for name in sorted(_REQUIRED_PATH_ROOTS):
        raw = value[name]
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise ValueError(
                f"Court inference runtime root {name!r} must be a trimmed string."
            )
        normalized[name] = raw
    return normalized


def _migrate_plain_config(
    config: MutableMapping[str, Any],
    *,
    runtime_path_roots: Mapping[str, str] | None,
) -> bool:
    changed = False

    # Checkpoints can move between worktrees, containers, and machines. Runtime
    # path authority comes from the caller rather than serialized absolute roots.
    normalized_roots = _normalized_runtime_path_roots(runtime_path_roots)
    if normalized_roots is not None and config.get("paths") != normalized_roots:
        config["paths"] = normalized_roots
        changed = True

    # Mixed-source loader composition is irrelevant when reconstructing only
    # the serialized model and is outside the standard Court config contract.
    if "mixed" in config:
        config.pop("mixed")
        changed = True

    run = config.get("run")
    if isinstance(run, MutableMapping) and "artifact_store" not in run:
        run["artifact_store"] = deepcopy(_LOCAL_ARTIFACT_STORE)
        changed = True

    # Residual head metadata is consumed separately before standard model
    # construction, then the exact serialized head hierarchy is restored.
    model = config.get("model")
    if isinstance(model, MutableMapping) and "dense_head" in model:
        model.pop("dense_head")
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


def _to_plain_mapping(config: object) -> dict[str, Any] | None:
    if isinstance(config, DictConfig):
        raw = OmegaConf.to_container(config, resolve=False)
        return cast("dict[str, Any]", raw) if isinstance(raw, dict) else None
    if isinstance(config, Mapping):
        return deepcopy(dict(cast("Mapping[str, Any]", config)))
    return None


def extract_court_checkpoint_dense_head_config(
    config: object,
) -> Mapping[str, object] | None:
    """Return a detached serialized dense-head config, when present."""
    plain = _to_plain_mapping(config)
    if plain is None:
        return None
    model = plain.get("model")
    dense_head = model.get("dense_head") if isinstance(model, Mapping) else None
    if not isinstance(dense_head, Mapping):
        return None
    return deepcopy(dict(cast("Mapping[str, object]", dense_head)))


def migrate_court_inference_config(
    config: object,
    *,
    runtime_path_roots: Mapping[str, str] | None = None,
) -> CourtInferenceConfig | None:
    """Normalize historical training metadata for model-only inference.

    Training entry points remain strict and require current configuration.
    This migration is scoped to checkpoint inference and only changes fields
    irrelevant to the restored forward semantics:

    - serialized roots are replaced by caller-authorized runtime roots;
    - mixed-source data-loader composition is removed;
    - missing artifact publication settings become a local-only store;
    - serialized dense-head metadata is consumed by the checkpoint loader;
    - retired line-target generator schema names become the current one.

    Returns ``None`` when no known migration is required.
    """
    plain = _to_plain_mapping(config)
    if plain is None or not _migrate_plain_config(
        plain,
        runtime_path_roots=runtime_path_roots,
    ):
        return None
    return OmegaConf.create(plain) if isinstance(config, DictConfig) else plain


def load_court_inference_config_override(
    checkpoint_path: Path,
    *,
    runtime_path_roots: Mapping[str, str] | None = None,
) -> CourtInferenceConfig | None:
    """Read only checkpoint metadata needed for known inference migrations."""
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
        return migrate_court_inference_config(
            hyper_parameters.get("config"),
            runtime_path_roots=runtime_path_roots,
        )
    finally:
        del checkpoint


def load_court_inference_lightning_module(
    checkpoint_path: Path,
    *,
    config_override: object | None = None,
    runtime_path_roots: Mapping[str, str] | None = None,
    strict: bool = True,
) -> CourtDetectionLightningModule:
    """Restore a Court module, including checkpoint-defined residual heads.

    Lightning cannot instantiate checkpoints whose historical config contains
    fields removed from the current strict training schema. This loader keeps
    training strict, constructs the current standard module from an
    inference-safe config, restores the exact serialized dense-head hierarchy,
    and finally performs a strict state-dict load.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Court checkpoint root must be a mapping.")
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise KeyError("Court checkpoint is missing hyper_parameters.")
    source_config = hyper_parameters.get("config")
    selected_config = source_config if config_override is None else config_override
    migrated_config = migrate_court_inference_config(
        selected_config,
        runtime_path_roots=runtime_path_roots,
    )
    runtime_config = selected_config if migrated_config is None else migrated_config
    if runtime_config is None:
        raise KeyError("Court checkpoint is missing its serialized config.")

    target_bundle_state = hyper_parameters.get("target_bundle_state")
    if not isinstance(target_bundle_state, Mapping):
        raise KeyError("Court checkpoint is missing target_bundle_state.")
    module = CourtDetectionLightningModule(
        runtime_config,
        target_bundle_state=cast("Mapping[str, object]", target_bundle_state),
    )

    dense_head_config = extract_court_checkpoint_dense_head_config(source_config)
    if dense_head_config is not None:
        module.model.heads = build_checkpoint_dense_heads(
            dense_head_config,
            input_channels=int(module.model.decoder.output_channels),
            output_channels=module.model.target_bundle_spec.head_channels,
        )

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise KeyError("Court checkpoint is missing state_dict.")
    module.load_state_dict(
        cast("Mapping[str, Tensor]", state_dict),
        strict=strict,
    )
    module.eval()
    return module


__all__ = [
    "CourtInferenceConfig",
    "extract_court_checkpoint_dense_head_config",
    "load_court_inference_config_override",
    "load_court_inference_lightning_module",
    "migrate_court_inference_config",
]
