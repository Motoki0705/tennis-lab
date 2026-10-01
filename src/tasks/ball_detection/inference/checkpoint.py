"""Strict model-only checkpoint restoration shared by ball inference consumers."""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.tasks.ball_detection.model_io.contracts import BallModelIOError
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_detection.model_io.normalization import BallImageNormalization
from src.tasks.base.model_io import BoundModelIO
from src.utils.configuration import PathResolver

MODEL_STATE_PREFIX = "model."
_LOGGER = logging.getLogger(__name__)


class BallInferenceCheckpointError(ValueError):
    """A saved checkpoint cannot supply the declared inference model."""


@dataclass(frozen=True, slots=True)
class LoadedBallCheckpoint:
    model_io: BoundModelIO[Tensor, Tensor, Tensor]
    config: DictConfig
    image_normalization: BallImageNormalization
    backbone_asset_migration: dict[str, str] | None


def _runtime_backbone_config(
    config: DictConfig, resolver: PathResolver | None,
) -> tuple[DictConfig, dict[str, str] | None]:
    """Migrate the known external DINOv3 layout without changing saved metadata."""
    if config.model.name != "dinov3_rope":
        return config, None
    runtime = copy.deepcopy(config)
    saved_path = str(config.model.backbone.checkpoint_path)
    parts = Path(saved_path).parts
    migration = None
    if parts[:2] == ("dinov3", "checkpoints"):
        if len(parts) != 3 or parts[2] in {".", ".."}:
            raise BallInferenceCheckpointError("Legacy DINOv3 asset path must name exactly one checkpoint file")
        runtime_path = str(Path("dinov3") / parts[2])
        runtime.model.backbone.checkpoint_path = runtime_path
        migration = {"layout": "external_dinov3_to_checkpoint", "saved_path": saved_path, "runtime_path": runtime_path}
        if resolver is None:
            # The legacy checkpoint root described learned outputs, not this backbone.
            runtime.paths.checkpoint_root = "ckpt"
    if resolver is not None:
        runtime.paths = dict(resolver.roots.as_mapping())
    if migration is not None:
        from src.tasks.ball_detection.configuration import BallRuntimePaths
        migration["checkpoint_root"] = str(BallRuntimePaths.from_config(runtime).resolver.roots.checkpoint_root)
        _LOGGER.warning("Migrating saved DINOv3 asset path %s -> %s under %s",
                        saved_path, migration["runtime_path"], migration["checkpoint_root"])
    return runtime, migration


def load_ball_checkpoint(
    path: str | Path,
    *,
    strict: bool = True,
    weights_only: bool = False,
    resolver: PathResolver | None = None,
) -> LoadedBallCheckpoint:
    """Restore one model on CPU without constructing a Lightning training module.

    Callers authorize the path. Only saved model/input configuration and explicit
    ``model.`` state entries participate; training options are not synthesized.
    """
    if strict is not True:
        raise ValueError("Ball inference requires strict checkpoint loading")
    if type(weights_only) is not bool:
        raise TypeError("weights_only must be a boolean")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    location = f"Ball detection checkpoint {path}"
    checkpoint = torch.load(path, map_location="cpu", weights_only=weights_only)
    if not isinstance(checkpoint, Mapping):
        raise BallInferenceCheckpointError(f"{location}: checkpoint root must be a mapping.")
    hyper = checkpoint.get("hyper_parameters")
    if not isinstance(hyper, Mapping) or not isinstance(hyper.get("config"), Mapping):
        raise BallInferenceCheckpointError(
            f"{location}: hyper_parameters.config must be a mapping."
        )
    config = OmegaConf.create(hyper["config"])
    if not isinstance(config, DictConfig):
        raise BallInferenceCheckpointError(f"{location}: config must be a mapping.")
    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise BallInferenceCheckpointError(f"{location}: checkpoint has no state_dict mapping.")
    weights = {
        key[len(MODEL_STATE_PREFIX) :]: value
        for key, value in raw_state.items()
        if isinstance(key, str) and key.startswith(MODEL_STATE_PREFIX)
    }
    if not weights:
        raise BallInferenceCheckpointError(
            f"{location}: checkpoint state_dict carries no {MODEL_STATE_PREFIX!r} model weights."
        )
    runtime_config, migration = _runtime_backbone_config(config, resolver)
    bound = build_ball_detection_pair(runtime_config)
    try:
        bound.model.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise BallInferenceCheckpointError(
            f"{location}: model state_dict does not match the architecture: {error}"
        ) from error
    try:
        normalization = BallImageNormalization.from_config(config)
    except BallModelIOError as error:
        raise BallInferenceCheckpointError(f"{location}: {error}") from error
    return LoadedBallCheckpoint(bound, config, normalization, migration)
