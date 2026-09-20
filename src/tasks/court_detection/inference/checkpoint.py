"""Checkpoint-authoritative inference, independent of training run/data options."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.configuration import CourtLossConfig, CourtModelConfig
from src.tasks.court_detection.data.bundle_state import (
    deserialize_target_bundle,
    serialize_target_bundle,
)
from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.model_io.contracts import CourtModelIOError
from src.tasks.court_detection.model_io.factory import (
    CourtDetectionBoundModelIO,
    build_court_inference_pair,
)
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CourtModelIOError(f"Checkpoint {name} must be a mapping")
    return value


def _plain(value: object) -> dict[str, Any]:
    mapping = _mapping(value, "configuration section")
    config = value if isinstance(value, DictConfig) else OmegaConf.create(dict(mapping))
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))


@dataclass(frozen=True)
class CourtInferenceSpec:
    model: CourtModelConfig
    loss: CourtLossConfig
    target_bundle: CourtTargetBundleSpec
    short_side: int
    architecture: dict[str, Any]

    @classmethod
    def from_checkpoint_config(
        cls,
        config: object,
        bundle_state: object,
        *,
        resolver: PathResolver | None = None,
    ) -> CourtInferenceSpec:
        """Validate only fields that determine the saved inference computation.

        Training config validation remains strict and unchanged. In particular,
        no artifact_store, source-scope or augmentation keys are synthesized to
        make an old checkpoint look like a modern training run.
        """
        values = _mapping(config, "config")
        if resolver is None:
            resolver = PathResolver(
                RuntimePathRoots.from_mapping(
                    _mapping(values.get("paths"), "paths"),
                    repository_root=PROJECT_ROOT,
                )
            )
        model = CourtModelConfig.from_mapping(values.get("model"), resolver=resolver)
        loss = CourtLossConfig.from_mapping(values.get("loss"))
        bundle = deserialize_target_bundle(
            _mapping(bundle_state, "target_bundle_state")
        )
        augmentation = _mapping(
            _mapping(values.get("data"), "data").get("augmentation"),
            "data.augmentation",
        )
        short_side = augmentation.get("val_short_side")
        if type(short_side) is not int or short_side <= 0:
            raise CourtModelIOError(
                "Checkpoint val_short_side must be a positive integer"
            )
        if set(bundle.kinds) - set(loss.dense_weights):
            raise CourtModelIOError("Checkpoint loss/head definitions disagree")
        if model.dense_head.name == "residual" and set(bundle.kinds) - set(
            model.dense_head.branches
        ):
            raise CourtModelIOError(
                "Checkpoint has no saved architecture for a declared head"
            )
        return cls(model, loss, bundle, short_side, _plain(values["model"]))


@dataclass(frozen=True)
class LoadedCourtModel:
    model_io: CourtDetectionBoundModelIO
    spec: CourtInferenceSpec
    identity: dict[str, Any]


def load_court_checkpoint(
    path: Path,
    *,
    resolver: PathResolver | None = None,
) -> LoadedCourtModel:
    """Strictly load the complete model on CPU, then let the predictor move it.

    Callers own path authorization: the normal factory uses PathResolver and
    the inference UI resolves its allowed checkpoint catalog before this call.
    """
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    checkpoint = _mapping(
        torch.load(path, map_location="cpu", weights_only=False, mmap=True), "body"
    )
    hyper = _mapping(checkpoint.get("hyper_parameters"), "hyper_parameters")
    spec = CourtInferenceSpec.from_checkpoint_config(
        hyper.get("config"),
        hyper.get("target_bundle_state"),
        resolver=resolver,
    )
    state = _mapping(checkpoint.get("state_dict"), "state_dict")
    if not state or any(
        not key.startswith("model.") or not isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise CourtModelIOError(
            "Expected the complete tensor state_dict with model. prefixes"
        )
    pair = build_court_inference_pair(
        model_config=spec.model,
        loss_config=spec.loss,
        short_side=spec.short_side,
        target_bundle=spec.target_bundle,
    )
    pair.model.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items()}, strict=True
    )
    if file_sha256(path) != digest:
        raise CourtModelIOError("Checkpoint changed while being loaded")
    backbone = spec.model.encoder.checkpoint_path
    identity = {
        "schema": "court_inference_checkpoint_v1",
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "backbone_sha256": file_sha256(backbone) if backbone is not None else None,
        "architecture": spec.architecture,
        "target_bundle": serialize_target_bundle(spec.target_bundle),
        "short_side": spec.short_side,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    return LoadedCourtModel(pair, spec, identity)
