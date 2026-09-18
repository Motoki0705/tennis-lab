"""Unit tests for the inference-only ball checkpoint loader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.visualization.inference.loader import (
    BallInferenceCheckpointError,
    load_ball_model,
)


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
