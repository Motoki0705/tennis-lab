"""Inference-only PLCS checkpoint loading for the visualization UI.

``PLCSPredictor.load_from_checkpoint`` and
``PLCSTrackingPredictor.load_from_checkpoint`` restore a checkpoint through
``LightningModule.load_from_checkpoint``, which composes the *complete*
``PLCSTrainingConfig``. That training boundary keeps growing: it now demands
sections such as ``run.artifact_store`` that a curated inference checkpoint
never stored. Fabricating those values would silently invent training state, so
this loader instead rebuilds only the slice the model factory consumes.

Construction mirrors the part of ``PLCSTrainingConfig.from_config`` that runs
*before* the training sections are validated, using the very same functions:

1. ``load_and_validate_checkpoint`` (court-coordinate contract),
2. ``prepare_plcs_checkpoint_court_keypoint_config`` (saved selector + CourtKP20),
3. ``PLCSModelConfig.from_mapping`` for the model variant,
4. ``PLCSPathConfig.from_config`` for the saved path roots,
5. ``PLCSDataConfig.from_mapping`` for the resolved data contract.

The three resulting values are handed to the unchanged ``build_plcs_model_io``
factory. Every ``on_load_checkpoint`` marker check the Lightning modules run is
repeated here, and only the explicit ``model.`` state-dict entries are restored
with ``strict=True`` so a missing or surplus parameter is rejected instead of
patched up.

This module is inference-only: optimizer, scheduler, loss, and metric state are
never read or restored. Training and the existing predictors keep their current
behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from src.tasks.base.configuration import require_config_mapping
from src.tasks.base.generate_dataset import CourtKeypointContract
from src.tasks.plcs.configuration import PLCSDataConfig, PLCSModelConfig
from src.tasks.plcs.configuration_contracts import PLCSPathConfig
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.inference.tracking_predictor import PLCSTrackingPredictor
from src.tasks.plcs.model_io import (
    PLCSBoundModelIO,
    PLCSModelIOAdapter,
    PLCSTrackQueryIOAdapter,
    build_plcs_model_io,
    prepare_plcs_checkpoint_court_keypoint_config,
    resolve_plcs_track_query_reference_contract,
    validate_plcs_checkpoint_court_keypoints,
    validate_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.model_io.axial_reference import validate_axial_reference_checkpoint
from src.utils.configuration import PathResolver, PathRole
from src.utils.device import resolve_device
from src.utils.schema.court_normalization import (
    load_and_validate_checkpoint,
    validate_court_coordinate_normalization,
)

#: Checkpoints store Lightning module state under this prefix; only these
#: entries are model weights.
MODEL_STATE_PREFIX = "model."


class InferenceCheckpointError(ValueError):
    """Raised when a curated checkpoint cannot drive PLCS inference."""


@dataclass(frozen=True, slots=True)
class _InferenceModelConfig:
    """Minimal read-only config that satisfies :class:`PLCSModelIOConfig`."""

    model: PLCSModelConfig
    data: PLCSDataConfig
    court_keypoint_contract: CourtKeypointContract


@dataclass(frozen=True, slots=True)
class InferenceCheckpoint:
    """Everything the UI loader resolved from one curated checkpoint."""

    path: Path
    raw: Mapping[str, Any]
    config: DictConfig
    model_config: PLCSModelConfig
    data_config: PLCSDataConfig
    court_keypoint_contract: CourtKeypointContract

    def model_io(self) -> PLCSBoundModelIO:
        """Build the bound model and I/O adapter through the shared factory."""
        return build_plcs_model_io(
            _InferenceModelConfig(
                model=self.model_config,
                data=self.data_config,
                court_keypoint_contract=self.court_keypoint_contract,
            )
        )


def _resolve_single_checkpoint(
    checkpoint_path: str | Path, resolver: PathResolver
) -> Path:
    """Validate exactly one checkpoint path inside the caller's root boundary."""
    candidate = Path(checkpoint_path)
    resolved: Path = (
        resolver.validate(PathRole.CHECKPOINT, candidate)
        if candidate.is_absolute()
        else resolver.resolve(PathRole.CHECKPOINT, candidate)
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def _model_state_dict(
    checkpoint: Mapping[str, Any], *, location: str
) -> dict[str, Any]:
    """Extract only the explicit ``model.`` weights and reject empty payloads."""
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise InferenceCheckpointError(
            f"{location}: checkpoint has no state_dict mapping."
        )
    weights = {
        str(key)[len(MODEL_STATE_PREFIX) :]: value
        for key, value in raw_state.items()
        if isinstance(key, str) and key.startswith(MODEL_STATE_PREFIX)
    }
    if not weights:
        raise InferenceCheckpointError(
            f"{location}: checkpoint state_dict carries no "
            f"{MODEL_STATE_PREFIX!r} model weights."
        )
    return weights


def _restore_model_weights(
    model: nn.Module, checkpoint: Mapping[str, Any], *, location: str
) -> None:
    """Strictly restore the model parameters and buffers from one checkpoint.

    ``strict=True`` rejects both missing and unexpected entries, so an
    architecture that no longer matches the stored weights fails loudly rather
    than being silently patched or key-translated.
    """
    weights = _model_state_dict(checkpoint, location=location)
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise InferenceCheckpointError(
            f"{location}: model state_dict does not match the architecture: {error}"
        ) from error


def load_inference_checkpoint(
    *,
    checkpoint_path: str | Path,
    resolver: PathResolver,
    court_keypoint_contract: CourtKeypointContract | None = None,
) -> InferenceCheckpoint:
    """Resolve a curated checkpoint into the factory-ready inference slice."""
    path = _resolve_single_checkpoint(checkpoint_path, resolver)
    checkpoint = load_and_validate_checkpoint(path)
    config, contract = prepare_plcs_checkpoint_court_keypoint_config(
        checkpoint,
        court_keypoint_contract,
        location=str(path),
    )
    model_config = PLCSModelConfig.from_mapping(
        require_config_mapping(config, "model", path="configuration")
    )
    paths = PLCSPathConfig.from_config(config)
    data_config = PLCSDataConfig.from_mapping(
        require_config_mapping(config, "data", path="configuration"),
        resolver=paths.resolver,
        model=model_config,
    )
    return InferenceCheckpoint(
        path=path,
        raw=checkpoint,
        config=config,
        model_config=model_config,
        data_config=data_config,
        court_keypoint_contract=contract,
    )


def _validate_standard_markers(checkpoint: InferenceCheckpoint) -> None:
    """Repeat ``PLCSLightningModule.on_load_checkpoint`` exactly."""
    validate_axial_reference_checkpoint(
        checkpoint.raw, model_name=checkpoint.model_config.name
    )
    validate_court_coordinate_normalization(checkpoint.raw, artifact="PLCS checkpoint")
    validate_plcs_checkpoint_court_keypoints(
        checkpoint.raw, checkpoint.court_keypoint_contract
    )


def _validate_tracking_markers(checkpoint: InferenceCheckpoint) -> None:
    """Repeat ``PLCSTrackingLightningModule.on_load_checkpoint`` exactly."""
    validate_court_coordinate_normalization(
        checkpoint.raw, artifact="PLCS tracking checkpoint"
    )
    validate_plcs_checkpoint_court_keypoints(
        checkpoint.raw, checkpoint.court_keypoint_contract
    )
    reference_contract = resolve_plcs_track_query_reference_contract(
        checkpoint.model_config,
        checkpoint.court_keypoint_contract,
    )
    validate_plcs_checkpoint_track_query_reference(checkpoint.raw, reference_contract)


def load_standard_predictor(
    *,
    checkpoint_path: str | Path,
    resolver: PathResolver,
    device: str | torch.device,
    court_keypoint_contract: CourtKeypointContract | None = None,
) -> PLCSPredictor:
    """Build a standard PLCS predictor without restoring training state."""
    checkpoint = load_inference_checkpoint(
        checkpoint_path=checkpoint_path,
        resolver=resolver,
        court_keypoint_contract=court_keypoint_contract,
    )
    location = f"PLCS inference checkpoint {checkpoint.path}"
    _validate_standard_markers(checkpoint)
    bound = checkpoint.model_io()
    adapter = bound.adapter
    if not isinstance(adapter, PLCSModelIOAdapter):
        raise InferenceCheckpointError(
            f"{location}: checkpoint does not contain a standard PLCS adapter."
        )
    _restore_model_weights(bound.model, checkpoint.raw, location=location)
    return PLCSPredictor(
        model=bound.model,
        adapter=adapter,
        device=resolve_device(device),
        court_keypoint_contract=checkpoint.court_keypoint_contract,
    )


def load_tracking_predictor(
    *,
    checkpoint_path: str | Path,
    resolver: PathResolver,
    device: str | torch.device,
    court_keypoint_contract: CourtKeypointContract | None = None,
) -> PLCSTrackingPredictor:
    """Build a track-query predictor without restoring training state."""
    checkpoint = load_inference_checkpoint(
        checkpoint_path=checkpoint_path,
        resolver=resolver,
        court_keypoint_contract=court_keypoint_contract,
    )
    location = f"PLCS inference checkpoint {checkpoint.path}"
    _validate_tracking_markers(checkpoint)
    bound = checkpoint.model_io()
    adapter = bound.adapter
    if not isinstance(adapter, PLCSTrackQueryIOAdapter):
        raise InferenceCheckpointError(
            f"{location}: checkpoint does not contain a PLCS track-query adapter."
        )
    _restore_model_weights(bound.model, checkpoint.raw, location=location)
    return PLCSTrackingPredictor(
        model=bound.model,
        adapter=adapter,
        device=resolve_device(device),
        court_keypoint_contract=checkpoint.court_keypoint_contract,
    )


def model_state_keys(checkpoint: Mapping[str, Any]) -> Sequence[str]:
    """Return the ``model.`` state-dict keys a checkpoint declares."""
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping):
        return ()
    return tuple(
        str(key)
        for key in raw_state
        if isinstance(key, str) and key.startswith(MODEL_STATE_PREFIX)
    )


__all__ = [
    "MODEL_STATE_PREFIX",
    "InferenceCheckpoint",
    "InferenceCheckpointError",
    "load_inference_checkpoint",
    "load_standard_predictor",
    "load_tracking_predictor",
    "model_state_keys",
]
