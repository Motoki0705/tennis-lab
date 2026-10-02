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
from src.tasks.base.model_io.court_keypoint_contract import (
    validate_neural_court_observation_order,
)
from src.tasks.blcs.configuration import (
    parse_court_keypoint_contract,
    parse_model_config,
)
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


def _qualify_metadata_free_config(config: Any) -> Any:
    """Add explicit physical-v1 config only for non-tracking checkpoints."""
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
    """Restore an axial checkpoint with exact physical CourtKP semantics."""
    checkpoint = _load_checkpoint(path)
    requested = (
        runtime_court_keypoints
        if isinstance(runtime_court_keypoints, CourtKeypointContract)
        else resolve_court_keypoint_contract(runtime_court_keypoints)
        if runtime_court_keypoints is not None
        else None
    )
    compatibility = (
        resolve_model_artifact_court_keypoint_contract(checkpoint, location=str(path))
        if requested is None
        else validate_model_artifact_court_keypoint_contract(
            checkpoint, requested, location=str(path)
        )
    )
    validate_neural_court_observation_order(checkpoint, compatibility.contract)
    config = _checkpoint_config(checkpoint)
    if compatibility.legacy_metadata_free:
        config = _qualify_metadata_free_config(config)
    elif not _has_court_keypoint_section(config):
        raise RuntimeError(f"{path}: checkpoint config must include court_keypoints.")
    parse_model_config(config)
    contract = parse_court_keypoint_contract(config)
    if contract != compatibility.contract:
        raise CourtKeypointContractMismatchError(
            f"{path}: checkpoint config and artifact CourtKP contracts disagree."
        )
    return BLCSCheckpointRuntime(
        config=config,
        court_keypoint_contract=contract,
        legacy_metadata_free=compatibility.legacy_metadata_free,
    )


def validate_checkpoint_path(
    path: Path, runtime_court_keypoints: CourtKeypointContract | str
) -> None:
    """Validate the axial model and coordinates before resume or initialization."""
    load_checkpoint_runtime(path, runtime_court_keypoints=runtime_court_keypoints)


__all__ = [
    "BLCSCheckpointRuntime",
    "load_checkpoint_config",
    "load_checkpoint_runtime",
    "validate_checkpoint_path",
]
