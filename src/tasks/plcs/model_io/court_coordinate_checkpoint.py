"""PLCS checkpoint binding for the shared court-coordinate contract."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import CourtCoordinateNormalizationConfig
from src.tasks.base.data import (
    CourtCoordinateContractMismatchError,
    MissingCourtCoordinateMetadataError,
)
from src.tasks.base.model_io import (
    resolve_checkpoint_court_coordinate_contract,
    validate_checkpoint_court_coordinate_contract,
    write_checkpoint_court_coordinate_contract,
)
from src.utils.schema.court_normalization import CourtCoordinateNormalization


def write_plcs_checkpoint_normalization(
    checkpoint: MutableMapping[str, object],
    contract: CourtCoordinateNormalization,
) -> None:
    """Persist the resolved PLCS normalization contract at checkpoint root."""
    write_checkpoint_court_coordinate_contract(
        checkpoint,
        contract,
        location="PLCS checkpoint",
    )


def validate_plcs_checkpoint_normalization(
    checkpoint: Mapping[str, object],
    contract: CourtCoordinateNormalization,
) -> None:
    """Reject a checkpoint/runtime mismatch before state is restored."""
    validate_checkpoint_court_coordinate_contract(
        checkpoint,
        contract,
        location="PLCS checkpoint",
    )


def load_plcs_checkpoint_mapping(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, object]:
    """Load one trusted local PLCS Lightning checkpoint as a mapping."""
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"PLCS checkpoint must contain a mapping, got {type(checkpoint).__name__}."
        )
    if any(not isinstance(key, str) for key in checkpoint):
        raise TypeError("PLCS checkpoint keys must all be strings.")
    return cast("Mapping[str, object]", checkpoint)


def _stored_config(checkpoint: Mapping[str, object]) -> DictConfig:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping):
        raise ValueError(
            "PLCS checkpoint is missing the required hyper_parameters mapping."
        )
    config = hyper_parameters.get("config")
    if isinstance(config, DictConfig):
        copied = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    elif isinstance(config, Mapping):
        copied = OmegaConf.create(dict(config))
    else:
        raise ValueError(
            "PLCS checkpoint hyper_parameters is missing the required config mapping."
        )
    if not isinstance(copied, DictConfig):
        raise TypeError("PLCS checkpoint config must compose to a mapping.")
    return copied


def prepare_plcs_checkpoint_config(
    checkpoint: Mapping[str, object],
    runtime_contract: CourtCoordinateNormalization | None,
    *,
    location: str = "PLCS checkpoint",
) -> tuple[DictConfig, CourtCoordinateNormalization]:
    """Resolve checkpoint normalization and return its constructor config.

    Versioned checkpoints restore their own contract when no runtime contract
    is supplied. Metadata-free checkpoints require an explicit legacy ``v1``
    runtime; their missing saved config section is injected only in memory.
    """
    compatibility = (
        resolve_checkpoint_court_coordinate_contract(
            checkpoint,
            location=location,
        )
        if runtime_contract is None
        else validate_checkpoint_court_coordinate_contract(
            checkpoint,
            runtime_contract,
            location=location,
        )
    )
    contract = compatibility.contract
    config = _stored_config(checkpoint)
    if "court_coordinate_normalization" not in config:
        if not compatibility.legacy_metadata_free:
            raise MissingCourtCoordinateMetadataError(
                f"{location}: checkpoint metadata is versioned but its saved "
                "config has no court_coordinate_normalization section."
            )
        return (
            cast(
                "DictConfig",
                OmegaConf.merge(
                    config,
                    {
                        "court_coordinate_normalization": {
                            "version": contract.version
                        }
                    },
                ),
            ),
            contract,
        )

    stored_contract = CourtCoordinateNormalizationConfig.from_config(config).contract
    if stored_contract != contract:
        raise CourtCoordinateContractMismatchError(
            f"{location}: saved config normalization "
            f"{stored_contract.version!r}/{stored_contract.scale_xyz!r} does not "
            f"match checkpoint/runtime {contract.version!r}/{contract.scale_xyz!r}."
        )
    return config, contract


__all__ = [
    "load_plcs_checkpoint_mapping",
    "prepare_plcs_checkpoint_config",
    "validate_plcs_checkpoint_normalization",
    "write_plcs_checkpoint_normalization",
]
