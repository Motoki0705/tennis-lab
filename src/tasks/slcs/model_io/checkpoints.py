"""SLCS checkpoint configuration and normalization-contract validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import CourtCoordinateNormalizationConfig
from src.tasks.base.data import (
    CourtCoordinateContractMismatchError,
    MissingCourtCoordinateMetadataError,
)
from src.tasks.base.model_io import validate_checkpoint_court_coordinate_contract
from src.utils.schema.court_normalization import CourtCoordinateNormalization

__all__ = [
    "load_slcs_checkpoint_mapping",
    "prepare_slcs_checkpoint_config",
]


def load_slcs_checkpoint_mapping(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, object]:
    """Load one trusted local Lightning checkpoint as a mapping."""
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"SLCS checkpoint must contain a mapping, got {type(checkpoint).__name__}."
        )
    if any(not isinstance(key, str) for key in checkpoint):
        raise TypeError("SLCS checkpoint keys must all be strings.")
    return cast("Mapping[str, object]", checkpoint)


def _stored_config(checkpoint: Mapping[str, object]) -> DictConfig:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError(
            "SLCS checkpoint is missing the required hyper_parameters mapping."
        )
    config = hyper_parameters.get("config")
    if isinstance(config, DictConfig):
        copied = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    elif isinstance(config, Mapping):
        copied = OmegaConf.create(dict(config))
    else:
        raise ValueError(
            "SLCS checkpoint hyper_parameters is missing the required config mapping."
        )
    if not isinstance(copied, DictConfig):
        raise TypeError("SLCS checkpoint config must compose to a mapping.")
    return copied


def prepare_slcs_checkpoint_config(
    checkpoint: Mapping[str, object],
    runtime_contract: CourtCoordinateNormalization,
    *,
    location: str = "SLCS checkpoint",
) -> DictConfig:
    """Validate checkpoint metadata/config and return its constructor config.

    Metadata-free legacy checkpoints may omit the normalization config because
    they predate the versioned contract.  Only an explicit v1 runtime can load
    them; a v1 section is added to the in-memory config passed to Lightning.
    Persisted checkpoint contents are never rewritten.
    """
    compatibility = validate_checkpoint_court_coordinate_contract(
        checkpoint,
        runtime_contract,
        location=location,
    )
    config = _stored_config(checkpoint)
    if "court_coordinate_normalization" not in config:
        if not compatibility.legacy_metadata_free:
            raise MissingCourtCoordinateMetadataError(
                f"{location}: checkpoint metadata is versioned but its saved "
                "config has no court_coordinate_normalization section."
            )
        return cast(
            "DictConfig",
            OmegaConf.merge(
                config,
                {
                    "court_coordinate_normalization": {
                        "version": runtime_contract.version
                    }
                },
            ),
        )

    stored_contract = CourtCoordinateNormalizationConfig.from_config(config).contract
    if stored_contract != runtime_contract:
        raise CourtCoordinateContractMismatchError(
            f"{location}: saved config normalization "
            f"{stored_contract.version!r}/{stored_contract.scale_xyz!r} does not "
            f"match runtime {runtime_contract.version!r}/"
            f"{runtime_contract.scale_xyz!r}."
        )
    return config
