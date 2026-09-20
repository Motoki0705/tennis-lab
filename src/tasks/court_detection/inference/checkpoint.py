"""Checkpoint-authoritative inference, independent of training run/data options."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.configuration import (
    CourtInferenceConfig,
    CourtLossConfig,
    CourtModelConfig,
)
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
    pose_long_side: bool
    patch_size: int
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
        runtime = CourtInferenceConfig.from_config(config, resolver=resolver)
        model, loss = runtime.model, runtime.loss
        bundle = deserialize_target_bundle(
            _mapping(bundle_state, "target_bundle_state")
        )
        if set(bundle.kinds) - set(loss.dense_weights):
            raise CourtModelIOError("Checkpoint loss/head definitions disagree")
        if model.dense_head.name == "residual" and set(bundle.kinds) - set(
            model.dense_head.branches
        ):
            raise CourtModelIOError(
                "Checkpoint has no saved architecture for a declared head"
            )
        return cls(
            model,
            loss,
            bundle,
            runtime.short_side,
            runtime.pose_long_side,
            runtime.patch_size,
            _plain(values["model"]),
        )


@dataclass(frozen=True)
class LoadedCourtModel:
    model_io: CourtDetectionBoundModelIO
    spec: CourtInferenceSpec
    identity: dict[str, Any]


def load_court_checkpoint(
    path: Path,
    *,
    resolver: PathResolver | None = None,
    strict: bool = True,
    weights_only: bool = False,
) -> LoadedCourtModel:
    """Strictly load the complete model on CPU, then let the predictor move it.

    Callers own path authorization: the normal factory uses PathResolver and
    the inference UI resolves its allowed checkpoint catalog before this call.
    """
    if not strict:
        raise ValueError("Court inference requires strict checkpoint loading")
    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    checkpoint = _mapping(
        torch.load(path, map_location="cpu", weights_only=weights_only, mmap=True),
        "body",
    )
    hyper = _mapping(checkpoint.get("hyper_parameters"), "hyper_parameters")
    spec = CourtInferenceSpec.from_checkpoint_config(
        hyper.get("config"),
        hyper.get("target_bundle_state"),
        resolver=resolver,
    )
    state = _mapping(checkpoint.get("state_dict"), "state_dict")
    # Lightning may also persist criterion buffers. Only the model namespace
    # participates in inference; every model parameter is still loaded strictly.
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not model_state or any(
        not isinstance(value, torch.Tensor) for value in model_state.values()
    ):
        raise CourtModelIOError(
            "Expected the complete tensor state_dict with model. prefixes"
        )
    pair = build_court_inference_pair(
        model_config=spec.model,
        loss_config=spec.loss,
        short_side=spec.short_side,
        pose_long_side=spec.pose_long_side,
        patch_size=spec.patch_size,
        target_bundle=spec.target_bundle,
    )
    pair.model.load_state_dict(model_state, strict=True)
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
        "pose_long_side": spec.pose_long_side,
        "patch_size": spec.patch_size,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
    return LoadedCourtModel(pair, spec, identity)


def load_court_pair(
    path: Path,
    *,
    resolver: PathResolver,
    strict: bool = True,
    weights_only: bool = False,
) -> CourtDetectionBoundModelIO:
    """Load the same strict checkpoint for consumers needing only the model pair."""
    return load_court_checkpoint(
        path, resolver=resolver, strict=strict, weights_only=weights_only
    ).model_io
