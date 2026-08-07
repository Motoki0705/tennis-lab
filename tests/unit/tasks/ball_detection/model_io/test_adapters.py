"""Unit tests for ball model-I/O boundary contracts."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import (
    BallModelInputSpec,
    BallModelIOError,
    BallPrediction,
)
from src.tasks.base.model_io import bind_model_io


class _CountingBallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.num_classes = 1
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        batch_size, frame_count, _, height, width = images.shape
        return images.new_zeros(batch_size, 1, frame_count, height, width)


def _rgb_adapter() -> BallModelIOAdapter:
    return BallModelIOAdapter(
        BallModelInputSpec(
            model_name="test_rgb",
            input_mode="rgb",
            input_layout="btchw",
            in_channels=3,
            num_classes=1,
            configured_frames=2,
            image_size_hw=None,
            minimum_spatial_size=None,
            mdd_gain=1.0,
            mdd_offset=0.0,
        ),
        expected_model_type=_CountingBallModel,
        minimum_frames=2,
    )


def _valid_training_batch() -> dict[str, torch.Tensor]:
    return {
        "images": torch.zeros(1, 2, 3, 8, 8),
        "heatmaps": torch.zeros(1, 2, 8, 8),
        "coords": torch.zeros(1, 2, 1, 2),
        "visibility": torch.ones(1, 2, 1, dtype=torch.bool),
        "original_size": torch.tensor([[8, 8]], dtype=torch.int64),
    }


def _run_training_boundary(
    adapter: BallModelIOAdapter,
    model: _CountingBallModel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    call = adapter.prepare_training_batch(batch)
    logits = model(*call.model_call.model_args)
    return adapter.training_logits(logits, call)


def test_bound_lifecycle_rejects_invalid_input_before_forward() -> None:
    model = _CountingBallModel()
    adapter = _rgb_adapter()
    adapter.validate_model_pair(model)
    pair = bind_model_io(model, adapter)

    with pytest.raises(BallModelIOError, match="floating dtype"):
        pair.run(torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8))

    assert model.calls == 0


@pytest.mark.parametrize(
    ("images", "message"),
    [
        (torch.zeros(1, 2, 3, 8, 8, dtype=torch.float64), "torch.float32"),
        (torch.full((1, 2, 3, 8, 8), -0.01), r"values must be in \[0, 1\]"),
        (torch.full((1, 2, 3, 8, 8), 1.01), r"values must be in \[0, 1\]"),
    ],
)
def test_image_semantics_fail_before_forward(
    images: torch.Tensor,
    message: str,
) -> None:
    model = _CountingBallModel()
    pair = bind_model_io(model, _rgb_adapter())

    with pytest.raises(BallModelIOError, match=message):
        pair.run(images)

    assert model.calls == 0


def test_image_range_inclusive_boundaries_enter_forward_once() -> None:
    model = _CountingBallModel()
    images = torch.zeros(1, 2, 3, 8, 8)
    images[:, 1] = 1.0

    output = bind_model_io(model, _rgb_adapter()).run(images)

    assert output.shape == (1, 2, 8, 8)
    assert model.calls == 1


def test_training_boundary_runs_validated_batch_once() -> None:
    model = _CountingBallModel()

    logits = _run_training_boundary(_rgb_adapter(), model, _valid_training_batch())

    assert logits.shape == (1, 2, 8, 8)
    assert model.calls == 1


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("missing", "missing required field 'coords'"),
        ("dtype", "images must use a floating dtype"),
        ("rank", "heatmaps must be rank 4"),
        ("shape", "coords must have shape"),
        ("semantic", "visibility values must be in \[0, 1\]"),
    ],
)
def test_training_contract_violations_fail_before_forward(
    violation: str,
    message: str,
) -> None:
    batch = _valid_training_batch()
    if violation == "missing":
        del batch["coords"]
    elif violation == "dtype":
        batch["images"] = batch["images"].to(torch.uint8)
    elif violation == "rank":
        batch["heatmaps"] = batch["heatmaps"].squeeze(0)
    elif violation == "shape":
        batch["coords"] = torch.zeros(1, 2, 1, 3)
    else:
        batch["visibility"] = torch.full((1, 2, 1), 2.0)
    model = _CountingBallModel()

    with pytest.raises(BallModelIOError, match=message):
        _run_training_boundary(_rgb_adapter(), model, batch)

    assert model.calls == 0


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_training_heatmap_range_fails_before_forward(value: float) -> None:
    batch = _valid_training_batch()
    batch["heatmaps"][0, 0, 0, 0] = value
    model = _CountingBallModel()

    with pytest.raises(BallModelIOError, match=r"heatmaps values must be in \[0, 1\]"):
        _run_training_boundary(_rgb_adapter(), model, batch)

    assert model.calls == 0


def test_prediction_decodes_stable_cpu_fields() -> None:
    adapter = _rgb_adapter()
    call = adapter.prepare_images(torch.zeros(1, 2, 3, 4, 5))
    logits = torch.zeros(1, 1, 2, 4, 5)

    prediction = adapter.prediction(logits, call, subpixel_refine=False)

    assert isinstance(prediction, BallPrediction)
    assert prediction.coords.shape == (1, 2, 2)
    assert prediction.confidence.shape == (1, 2)
    assert prediction.heatmaps.shape == (1, 2, 4, 5)
    assert prediction.heatmaps.device.type == "cpu"


def test_mdd_adapter_constructs_two_channel_temporal_input() -> None:
    adapter = BallModelIOAdapter(
        BallModelInputSpec(
            model_name="test_mdd",
            input_mode="mdd",
            input_layout="bcthw",
            in_channels=2,
            num_classes=1,
            configured_frames=3,
            image_size_hw=None,
            minimum_spatial_size=None,
            mdd_gain=1.0,
            mdd_offset=0.0,
        ),
        expected_model_type=nn.Identity,
        minimum_frames=2,
    )

    call = adapter.prepare_images(torch.rand(2, 3, 3, 6, 7))

    assert call.model_input.shape == (2, 2, 3, 6, 7)
    torch.testing.assert_close(
        call.model_input[:, :, 0],
        torch.zeros_like(call.model_input[:, :, 0]),
    )
