"""Unit tests for the inference-only ball checkpoint loader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.inference.checkpoint import (
    BallInferenceCheckpointError,
    load_ball_checkpoint,
)
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.visualization.inference.loader import load_ball_model
from src.utils.configuration import PathResolver, RuntimePathRoots


def test_loads_tiny_checkpoint_strictly(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    path = make_tiny_checkpoint(tmp_path / "outputs" / "run.ckpt", num_frames=2)
    loaded = load_ball_model(path, device="cpu")
    assert loaded.model_name == "conv_next_unet"
    assert loaded.num_frames == 2
    assert loaded.minimum_frames == 1
    assert loaded.image_size_hw == (64, 128)
    assert loaded.device == torch.device("cpu")
    assert not loaded.model.training


def test_missing_weight_is_rejected(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    def drop_one(state: dict[str, object]) -> dict[str, object]:
        key = sorted(state)[-1]
        return {name: value for name, value in state.items() if name != key}

    path = make_tiny_checkpoint(tmp_path / "run.ckpt", state_transform=drop_one)
    with pytest.raises(BallInferenceCheckpointError, match="does not match the architecture"):
        load_ball_model(path, device="cpu")


def test_checkpoint_without_model_prefix_is_rejected(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    def strip_prefix(state: dict[str, object]) -> dict[str, object]:
        return {name.removeprefix("model."): value for name, value in state.items()}

    path = make_tiny_checkpoint(tmp_path / "run.ckpt", state_transform=strip_prefix)
    with pytest.raises(BallInferenceCheckpointError, match="carries no 'model.'"):
        load_ball_model(path, device="cpu")


def test_checkpoint_without_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "run.ckpt"
    torch.save({"state_dict": {"model.a": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="hyper_parameters"):
        load_ball_model(path, device="cpu")


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ball_model(tmp_path / "absent.ckpt", device="cpu")


def test_cuda_request_without_cuda_is_rejected(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    path = make_tiny_checkpoint(tmp_path / "run.ckpt")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        load_ball_model(path, device="cuda")


@pytest.mark.parametrize("normalize", [False, True])
def test_predictor_restores_model_only_checkpoint_and_matches_ui(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, normalize: bool
) -> None:
    path = make_tiny_checkpoint(tmp_path / "ckpt" / "run.ckpt", num_frames=2)
    raw = torch.load(path, weights_only=False)
    raw["hyper_parameters"]["config"]["data"]["augmentation"]["normalize_imagenet"] = {
        "enabled": normalize, "mean": [.1, .2, .3], "std": [.2, .4, .8],
    }
    torch.save(raw, path)
    roots = RuntimePathRoots(
        project_root=tmp_path,
        checkpoint_root=tmp_path / "ckpt",
        data_root=tmp_path / "data",
        artifact_root=tmp_path / "artifacts",
        output_root=tmp_path / "outputs",
        cache_root=tmp_path / "cache",
        external_asset_root=tmp_path / "external",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Inference must not construct a Lightning training module")

    monkeypatch.setattr(BallDetectionPredictor, "_load_single_lightning_module", forbidden)
    predictor = BallDetectionPredictor.load_from_checkpoint(
        path, resolver=PathResolver(roots), device="cpu", subpixel_refine=True,
        strict=True, weights_only=False,
    )
    ui = load_ball_model(path, device="cpu")
    assert predictor.configured_frames == ui.num_frames == 2
    assert predictor.image_normalization.enabled is normalize
    if normalize:
        assert predictor.image_normalization.mean == (.1, .2, .3)
        assert predictor.image_normalization.std == (.2, .4, .8)
    assert not predictor.model.training
    images = torch.rand(1, 2, 3, 64, 128)
    actual = predictor.predict(images)
    with torch.no_grad():
        call = ui.adapter.prepare_model_call(images, image_normalization=ui.image_normalization)
        expected = ui.adapter.prediction(ui.model(*call.model_args), call, subpixel_refine=True)
    torch.testing.assert_close(actual.heatmaps, expected.heatmaps)
    torch.testing.assert_close(actual.coords, expected.coords)


def test_core_checkpoint_requires_model_and_declared_preprocessing(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    path = make_tiny_checkpoint(tmp_path / "run.ckpt")
    raw = torch.load(path, weights_only=False)
    raw["hyper_parameters"]["config"].pop("data")
    torch.save(raw, path)
    with pytest.raises(BallInferenceCheckpointError, match="normalize_imagenet"):
        load_ball_checkpoint(path)


def test_surplus_model_weight_is_rejected(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    path = make_tiny_checkpoint(
        tmp_path / "run.ckpt",
        state_transform=lambda state: {**state, "model.unexpected": torch.zeros(1)},
    )
    with pytest.raises(BallInferenceCheckpointError, match="Unexpected key"):
        load_ball_checkpoint(path)


def test_core_checkpoint_forbids_non_strict_loading(
    tmp_path: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    path = make_tiny_checkpoint(tmp_path / "run.ckpt")
    with pytest.raises(ValueError, match="requires strict"):
        load_ball_checkpoint(path, strict=False)
