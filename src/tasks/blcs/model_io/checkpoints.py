"""Fail-closed BLCS checkpoint metadata contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, open_dict

from src.tasks.base.configuration import CourtCoordinateNormalizationConfig
from src.tasks.base.data.court_coordinate_contract import (
    CourtCoordinateContractMismatchError,
)
from src.tasks.base.model_io import (
    resolve_checkpoint_court_coordinate_contract,
    validate_checkpoint_court_coordinate_contract,
)
from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)


@dataclass(frozen=True, slots=True)
class BLCSCheckpointRuntime:
    """Checkpoint composition and its validated/restored normalization."""

    config: Any
    normalization: CourtCoordinateNormalization
    legacy_metadata_free: bool


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("BLCS checkpoint root must be a mapping.")
    return checkpoint


def _checkpoint_config(checkpoint: Mapping[str, object]) -> Any:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise RuntimeError(
            "BLCS checkpoint is incompatible: hyper_parameters.config is required "
            "to compose its typed model I/O contract."
        )
    return hyper_parameters["config"]


def _inject_legacy_v1_config(config: Any) -> Any:
    copied = deepcopy(config)
    section = {"version": "v1"}
    if isinstance(copied, DictConfig):
        with open_dict(copied):
            copied["court_coordinate_normalization"] = section
        return copied
    if isinstance(copied, Mapping):
        result = dict(copied)
        result["court_coordinate_normalization"] = section
        return result
    raise RuntimeError(
        "BLCS checkpoint hyper_parameters.config must be a mapping-like config."
    )


def _has_normalization_section(config: Any) -> bool:
    if isinstance(config, (DictConfig, Mapping)):
        return "court_coordinate_normalization" in config
    return False


def load_checkpoint_config(path: Path) -> Any:
    """Load the explicit configuration required to compose a BLCS checkpoint."""
    return _checkpoint_config(_load_checkpoint(path))


def load_checkpoint_runtime(
    path: Path,
    *,
    runtime_normalization: CourtCoordinateNormalization | str | None = None,
) -> BLCSCheckpointRuntime:
    """Restore and validate checkpoint/config normalization before model load.

    Metadata-bearing checkpoints restore their own contract when no runtime is
    supplied. A metadata-free legacy checkpoint requires the caller to select
    ``v1`` explicitly; its old saved config is copied and qualified as v1 for
    composition without modifying the checkpoint.
    """
    checkpoint = _load_checkpoint(path)
    requested = (
        None
        if runtime_normalization is None
        else (
            runtime_normalization
            if isinstance(runtime_normalization, CourtCoordinateNormalization)
            else resolve_court_coordinate_normalization(runtime_normalization)
        )
    )
    if requested is None:
        checkpoint_contract = resolve_checkpoint_court_coordinate_contract(
            checkpoint,
            location=str(path),
        )
    else:
        checkpoint_contract = validate_checkpoint_court_coordinate_contract(
            checkpoint,
            requested,
            location=str(path),
        )

    config = _checkpoint_config(checkpoint)
    if not _has_normalization_section(config):
        if not checkpoint_contract.legacy_metadata_free:
            raise RuntimeError(
                f"{path}: metadata-bearing BLCS checkpoint config must include "
                "court_coordinate_normalization."
            )
        config = _inject_legacy_v1_config(config)
    config_contract = CourtCoordinateNormalizationConfig.from_config(config).contract
    if config_contract != checkpoint_contract.contract:
        raise CourtCoordinateContractMismatchError(
            f"{path}: checkpoint config normalization "
            f"{config_contract.version!r}/{config_contract.scale_xyz!r} does not "
            "match checkpoint/runtime normalization "
            f"{checkpoint_contract.contract.version!r}/"
            f"{checkpoint_contract.contract.scale_xyz!r}."
        )
    return BLCSCheckpointRuntime(
        config=config,
        normalization=checkpoint_contract.contract,
        legacy_metadata_free=checkpoint_contract.legacy_metadata_free,
    )


def validate_checkpoint_path(
    path: Path,
    runtime_normalization: CourtCoordinateNormalization,
) -> None:
    """Validate a resume/init checkpoint contract without composing a model."""
    load_checkpoint_runtime(
        path,
        runtime_normalization=runtime_normalization,
    )


__all__ = [
    "BLCSCheckpointRuntime",
    "load_checkpoint_config",
    "load_checkpoint_runtime",
    "validate_checkpoint_path",
]
