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
from src.tasks.plcs.model_io import (
    build_plcs_model_io,
    resolve_plcs_track_query_reference_contract,
    write_plcs_checkpoint_court_keypoints,
    write_plcs_checkpoint_track_query_reference,
)
from src.tasks.plcs.model_io.axial_reference import write_axial_reference_checkpoint
from src.tasks.plcs.visualization.inference.loader import (
    MODEL_STATE_PREFIX,
    InferenceCheckpointError,
    load_standard_predictor,
    load_tracking_predictor,
    model_state_keys,
)
from src.utils.configuration import (
    PathContractError,
    PathResolver,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import add_court_coordinate_normalization

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


def _write_checkpoint(
    path: Path,
    *,
    kind: str,
    state_dict: dict[str, Any] | None = None,
    drop_marker: str | None = None,
) -> Path:
    config = _curated_config(kind)
    checkpoint: dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "state_dict": (
            dict(_checkpoint_state_dict(kind)) if state_dict is None else state_dict
        ),
        "hyper_parameters": {"config": OmegaConf.create(config)},
    }
    tracking = kind == _TRACKING
    write_axial_reference_checkpoint(
        checkpoint, model_name=_training_config(kind).model.name
    )
    add_court_coordinate_normalization(
        checkpoint,
        artifact="PLCS tracking checkpoint" if tracking else "PLCS checkpoint",
    )
    write_plcs_checkpoint_court_keypoints(checkpoint, _runtime_contract(kind))
    if tracking:
        write_plcs_checkpoint_track_query_reference(
            checkpoint,
            resolve_plcs_track_query_reference_contract(
                _training_config(kind).model, _runtime_contract(kind)
            ),
        )
    if drop_marker is not None:
        checkpoint.pop(drop_marker, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def _resolver(tmp_path: Path) -> PathResolver:
    checkpoint_root = tmp_path / "ckpt"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        checkpoint_root=checkpoint_root,
        artifact_root=checkpoint_root,
        output_root=checkpoint_root,
        cache_root=PROJECT_ROOT / ".cache",
        external_asset_root=PROJECT_ROOT,
    )
    return PathResolver(roots)


# --------------------------------------------------------------- config slice


@pytest.mark.parametrize("kind", [_STANDARD, _TRACKING])
def test_loader_ignores_training_only_sections(tmp_path: Path, kind: str) -> None:
    """The whole point: a curated checkpoint without training sections loads."""
    # The training boundary refuses the curated config; the loader must not.
    with pytest.raises(Exception, match="artifact_store"):
        PLCSTrainingConfig.from_config(OmegaConf.create(_curated_config(kind)))
    path = _write_checkpoint(tmp_path / "ckpt" / "best.ckpt", kind=kind)
    if kind == _TRACKING:
        predictor = load_tracking_predictor(
            checkpoint_path=path,
            resolver=_resolver(tmp_path),
            device="cpu",
        )
    else:
        predictor = load_standard_predictor(
            checkpoint_path=path,
            resolver=_resolver(tmp_path),
            device="cpu",
        )
    assert predictor is not None
    assert predictor.device == torch.device("cpu")
    # Every stored ``model.`` weight is restored exactly, nothing dropped.
    stored = _checkpoint_state_dict(kind)
    reloaded = {
        f"{MODEL_STATE_PREFIX}{key}": value
        for key, value in predictor.model.state_dict().items()
    }
    assert set(reloaded) == set(stored)
    for key, value in stored.items():
        torch.testing.assert_close(reloaded[key], value)


def test_loader_preserves_model_and_adapter_types(tmp_path: Path) -> None:
    path = _write_checkpoint(tmp_path / "ckpt" / "best.ckpt", kind=_STANDARD)
    predictor = load_standard_predictor(
        checkpoint_path=path, resolver=_resolver(tmp_path), device="cpu"
    )
    assert type(predictor.model).__name__ == "PLCSMultiViewAxialReferenceModel"
    assert type(predictor.io_adapter).__name__ == "PLCSAxialReferenceIOAdapter"
    path = _write_checkpoint(tmp_path / "ckpt" / "track.ckpt", kind=_TRACKING)
    tracking = load_tracking_predictor(
        checkpoint_path=path, resolver=_resolver(tmp_path), device="cpu"
    )
    assert type(tracking.model).__name__ == "PLCSTrackQueryModel"


# ------------------------------------------------------------- strict loading


def test_loader_rejects_missing_and_surplus_model_weights(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    full = _checkpoint_state_dict(_STANDARD)

    missing = dict(full)
    removed_key = next(iter(missing))
    del missing[removed_key]
    missing_path = _write_checkpoint(
        tmp_path / "ckpt" / "missing.ckpt", kind=_STANDARD, state_dict=missing
    )
    with pytest.raises(InferenceCheckpointError, match="architecture"):
        load_standard_predictor(
            checkpoint_path=missing_path, resolver=resolver, device="cpu"
        )

    surplus = dict(full)
    surplus[f"{MODEL_STATE_PREFIX}unexpected.weight"] = torch.zeros(1)
    surplus_path = _write_checkpoint(
        tmp_path / "ckpt" / "surplus.ckpt", kind=_STANDARD, state_dict=surplus
    )
    with pytest.raises(InferenceCheckpointError, match="architecture"):
        load_standard_predictor(
            checkpoint_path=surplus_path, resolver=resolver, device="cpu"
        )


def test_loader_requires_the_explicit_model_prefix(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    state = {
        key.removeprefix(MODEL_STATE_PREFIX): value
        for key, value in _checkpoint_state_dict(_STANDARD).items()
    }
    path = _write_checkpoint(
        tmp_path / "ckpt" / "unprefixed.ckpt", kind=_STANDARD, state_dict=state
    )
    with pytest.raises(InferenceCheckpointError, match="model weights"):
        load_standard_predictor(checkpoint_path=path, resolver=resolver, device="cpu")


def test_loader_ignores_non_model_training_state(tmp_path: Path) -> None:
    """Optimizer/metric entries are dropped rather than mistaken for weights."""
    state = dict(_checkpoint_state_dict(_STANDARD))
    state["train_metrics.position_error"] = torch.zeros(1)
    state["optimizer_state"] = {"state": {}}
    path = _write_checkpoint(
        tmp_path / "ckpt" / "training.ckpt", kind=_STANDARD, state_dict=state
    )
    predictor = load_standard_predictor(
        checkpoint_path=path, resolver=_resolver(tmp_path), device="cpu"
    )
    assert predictor is not None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    keys = model_state_keys(raw)
    assert keys
    assert all(key.startswith(MODEL_STATE_PREFIX) for key in keys)
    assert "optimizer_state" not in keys


# ------------------------------------------------------------------ markers


def test_loader_rejects_mismatched_markers(tmp_path: Path) -> None:
    resolver = _resolver(tmp_path)
    standard = _write_checkpoint(
        tmp_path / "ckpt" / "no_axial.ckpt",
        kind=_STANDARD,
        drop_marker="axial_reference",
    )
    with pytest.raises(Exception, match="axial reference"):
        load_standard_predictor(
            checkpoint_path=standard, resolver=resolver, device="cpu"
        )
    tracking = _write_checkpoint(
        tmp_path / "ckpt" / "no_trackref.ckpt",
        kind=_TRACKING,
        drop_marker="track_query_reference",
    )
    with pytest.raises(Exception, match="track"):
        load_tracking_predictor(
            checkpoint_path=tracking, resolver=resolver, device="cpu"
        )


# ------------------------------------------------------------------- boundary


def test_loader_rejects_checkpoint_outside_its_root(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "best.ckpt"
    _write_checkpoint(outside, kind=_STANDARD)
    inside = tmp_path / "ckpt"
    inside.mkdir(parents=True, exist_ok=True)
    roots = RuntimePathRoots(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        checkpoint_root=inside,
        artifact_root=inside,
        output_root=inside,
        cache_root=PROJECT_ROOT / ".cache",
        external_asset_root=PROJECT_ROOT,
    )
    with pytest.raises(PathContractError):
        load_standard_predictor(
            checkpoint_path=outside,
            resolver=PathResolver(roots),
            device="cpu",
        )


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


@pytest.mark.skipif(
    _CURATED is None or not _CURATED.is_file(),
    reason="curated PLCS reference checkpoint is unavailable",
)
def test_curated_reference_checkpoint_predicts_16_frames_on_cpu() -> None:
    """The reported failure: a curated checkpoint must run through the UI."""
    from src.tasks.plcs.visualization.inference.service import (
        InferenceService,
        PredictionRequest,
    )

    assert _REPO_ROOT is not None
    service = InferenceService(
        data_root=_REPO_ROOT / "data" / "plcs",
        checkpoint_root=_REPO_ROOT / "outputs" / "plcs",
        checkpoint_roots=(_REPO_ROOT / "ckpt" / "plcs",),
        device="cpu",
        project_root=_REPO_ROOT,
    )
    request = PredictionRequest(
        checkpoint="ckpt/plcs/plcs-axial-reference-corners-v3-4-t128-seed42-epoch47.ckpt",
        family="single_object_camera_view_v2",
        scene="scene_000000",
        cameras=(0, 1, 2),
        reference_camera_id="camera_0",
        window_start=0,
        window_length=16,
        canonical_pose_source="gt",
        device="cpu",
    )
    service.validate_prediction_request(request)
    result = service.predict(request)
    header = result.header
    assert header["mode"] == "single"
    assert header["tracks"][0]["position"]["shape"] == [16, 3]
    assert header["tracks"][1]["has_joints"] is True
    assert header["metrics"]["position_error_m"]["mean"] is not None
    assert result.payload.size == header["payload_elements"]
    assert bool(torch.isfinite(torch.as_tensor(result.payload)).all())
