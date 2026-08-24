"""Fail-closed BLCS checkpoint metadata contract."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, open_dict

from src.tasks.base.generate_dataset import (
    CourtKeypointContract,
    CourtKeypointContractMismatchError,
    resolve_court_keypoint_contract,
)
from src.tasks.base.model_io import (
    resolve_model_artifact_court_keypoint_contract,
    validate_model_artifact_court_keypoint_contract,
)
from src.tasks.blcs.configuration import parse_court_keypoint_contract
from src.utils.schema.court_normalization import load_and_validate_checkpoint


@dataclass(frozen=True, slots=True)
class BLCSCheckpointRuntime:
    """Checkpoint composition and its validated CourtKP20 contract."""

    config: Any
    court_keypoint_contract: CourtKeypointContract
    legacy_metadata_free: bool


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint: Mapping[str, Any] = load_and_validate_checkpoint(path)
    return checkpoint


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> Any:
    hyper_parameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyper_parameters, Mapping) or "config" not in hyper_parameters:
        raise RuntimeError(
            "BLCS checkpoint is incompatible: hyper_parameters.config is required "
            "to compose its typed model I/O contract."
        )
    return hyper_parameters["config"]


def _overlay_legacy_physical_config(config: Any) -> Any:
    """Qualify an explicitly selected metadata-free checkpoint as physical v1."""
    copied = deepcopy(config)
    if isinstance(copied, DictConfig):
        with open_dict(copied):
            if "court_keypoints" not in copied:
                copied["court_keypoints"] = {"selector": "physical_v1"}
        return copied
    if isinstance(copied, Mapping):
        result = dict(copied)
        result.setdefault("court_keypoints", {"selector": "physical_v1"})
        return result
    raise RuntimeError(
        "BLCS checkpoint hyper_parameters.config must be a mapping-like config."
    )


def _has_court_keypoint_section(config: Any) -> bool:
    return isinstance(config, (DictConfig, Mapping)) and "court_keypoints" in config


def load_checkpoint_config(path: Path) -> Any:
    """Load the explicit configuration required to compose a BLCS checkpoint."""
    return _checkpoint_config(_load_checkpoint(path))


def load_checkpoint_runtime(
    path: Path,
    *,
    runtime_court_keypoints: CourtKeypointContract | str | None = None,
) -> BLCSCheckpointRuntime:
    """Restore exact CourtKP semantics or require explicit physical-v1 legacy use."""
    checkpoint = _load_checkpoint(path)
    requested = (
        None
        if runtime_court_keypoints is None
        else (
            runtime_court_keypoints
            if isinstance(runtime_court_keypoints, CourtKeypointContract)
            else resolve_court_keypoint_contract(runtime_court_keypoints)
        )
    )
    if requested is None:
        checkpoint_contract = resolve_model_artifact_court_keypoint_contract(
            checkpoint,
            location=str(path),
        )
    else:
        checkpoint_contract = validate_model_artifact_court_keypoint_contract(
            checkpoint,
            requested,
            location=str(path),
        )

    config = _checkpoint_config(checkpoint)
    if checkpoint_contract.legacy_metadata_free:
        config = _overlay_legacy_physical_config(config)
    elif not _has_court_keypoint_section(config):
        raise RuntimeError(
            f"{path}: metadata-bearing BLCS checkpoint config must include "
            "court_keypoints."
        )
    config_contract = parse_court_keypoint_contract(config)
    if config_contract != checkpoint_contract.contract:
        raise CourtKeypointContractMismatchError(
            f"{path}: checkpoint config CourtKP contract "
            f"{config_contract.contract_id!r} does not match checkpoint/runtime "
            f"{checkpoint_contract.contract.contract_id!r}."
        )
    return BLCSCheckpointRuntime(
        config=config,
        court_keypoint_contract=checkpoint_contract.contract,
        legacy_metadata_free=checkpoint_contract.legacy_metadata_free,
    )


def validate_checkpoint_path(
    path: Path,
    runtime_court_keypoints: CourtKeypointContract | str,
) -> None:
    """Validate a resume/init checkpoint CourtKP contract before model loading."""
    load_checkpoint_runtime(
        path,
        runtime_court_keypoints=runtime_court_keypoints,
    )


__all__ = [
    "BLCSCheckpointRuntime",
    "load_checkpoint_config",
    "load_checkpoint_runtime",
    "validate_checkpoint_path",
]
