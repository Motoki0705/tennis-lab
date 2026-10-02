"""Inference-only loader: config slicing, marker parity, and strict weights."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.court_keypoint_contract import PLCSCourtKeypointRuntimeConfig
from src.tasks.plcs.model_io import build_plcs_model_io
from src.tasks.plcs.visualization.inference.loader import (
    MODEL_STATE_PREFIX,
    load_standard_predictor,
)
from src.utils.configuration import (
    PathResolver,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

_CONFIG_DIR = PROJECT_ROOT / "src/tasks/plcs/configs"

_STANDARD = "standard"
_TRACKING = "tracking"

_TINY_OVERRIDES: dict[str, tuple[str, ...]] = {
    _STANDARD: (
        "model.hidden_dim=16",
        "model.num_layers=1",
        "model.num_heads=2",
        "model.ffn_dim=32",
        "model.rope_dim=6",
        "model.dropout=0.0",
        "training.compile.enabled=false",
    ),
    _TRACKING: (
        "model.hidden_dim=16",
        "model.num_heads=2",
        "model.ffn_dim=32",
        "model.rope_dim=6",
        "model.num_stages=4",
        "model.dropout=0.0",
        "model.mhc.coefficient_dim=8",
        "model.mhc.sinkhorn_iters=3",
        "model.cswa.compression_ratio=2",
        "model.cswa.window_radius=1",
        "model.cswa.backend=reference",
        "model.num_queries=4",
    ),
}

_CONFIG_NAME = {_STANDARD: "train_axial_reference", _TRACKING: "train_tracking"}


@lru_cache(maxsize=4)
def _resolved_config(kind: str) -> dict[str, Any]:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        composed = compose(
            config_name=_CONFIG_NAME[kind],
            overrides=list(_TINY_OVERRIDES[kind]),
        )
    return cast("dict[str, Any]", OmegaConf.to_container(composed, resolve=True))


def _training_config(kind: str) -> PLCSTrainingConfig:
    return PLCSTrainingConfig.from_config(OmegaConf.create(_resolved_config(kind)))


@lru_cache(maxsize=4)
def _runtime_contract(kind: str):
    return PLCSCourtKeypointRuntimeConfig.from_config(
        OmegaConf.create(_resolved_config(kind))
    ).contract


@lru_cache(maxsize=4)
def _checkpoint_state_dict(kind: str) -> dict[str, torch.Tensor]:
    """Return a real ``model.``-prefixed state dict for the tiny architecture."""
    bound = build_plcs_model_io(_training_config(kind))
    return {
        f"{MODEL_STATE_PREFIX}{key}": value
        for key, value in bound.model.state_dict().items()
    }


def _curated_config(kind: str) -> dict[str, Any]:
    """Mimic a curated checkpoint: no training-only ``run.artifact_store``."""
    config = copy.deepcopy(_resolved_config(kind))
    run = cast("dict[str, Any]", config["run"])
    removed = run.pop("artifact_store", None)
    assert removed is not None, "expected the composed run section to have it"
    return config


def _resolver(tmp_path: Path) -> PathResolver:
    checkpoint_root = tmp_path / "ckpt"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=(PROJECT_ROOT / "data").resolve(),
        checkpoint_root=checkpoint_root,
        artifact_root=checkpoint_root,
        output_root=checkpoint_root,
        cache_root=PROJECT_ROOT / ".cache",
        external_asset_root=PROJECT_ROOT,
    )
    return PathResolver(roots)


# --------------------------------------------------------------- config slice


# ------------------------------------------------------------- strict loading


# ------------------------------------------------------------------ markers


# ------------------------------------------------------------------- boundary


def test_loader_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_standard_predictor(
            checkpoint_path=tmp_path / "ckpt" / "absent.ckpt",
            resolver=_resolver(tmp_path),
            device="cpu",
        )


# ------------------------------------------------------- real curated checkpoint

_DATA_CANDIDATES = (
    PROJECT_ROOT / "data" / "plcs",
    Path("/home/kamimura/projects/tennis-lab/data/plcs"),
)
_DATASET_ROOT = next((path for path in _DATA_CANDIDATES if path.is_dir()), None)
_REPO_ROOT = None if _DATASET_ROOT is None else _DATASET_ROOT.parents[1]
_CURATED = (
    None
    if _REPO_ROOT is None
    else _REPO_ROOT
    / "ckpt"
    / "plcs"
    / "plcs-axial-reference-corners-v3-4-t128-seed42-epoch47.ckpt"
)
