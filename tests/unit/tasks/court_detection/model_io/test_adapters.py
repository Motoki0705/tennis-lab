"""Unit tests for court task adapter contracts."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.model_io.adapters import (
    CourtKeypointModelIO,
    CourtLineModelIO,
    CourtSegmentationModelIO,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtLinePrediction,
    CourtModelIOError,
    CourtModelSpec,
    CourtSegmentationPrediction,
    CourtTask,
    CourtTrainingResult,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


class _CountingCourtModel(CourtHierarchicalModel):
    def __init__(self, *, output_channels: int) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.num_classes = output_channels
        self.calls = 0

    def forward(
        self,
        images: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert feature_1 is None
        assert feature_2 is None
        assert feature_3 is None
        assert feature_4 is None
        self.calls += 1
        return images.new_zeros(
            images.shape[0],
            self.num_classes,
            images.shape[-2],
            images.shape[-1],
        )


def _spec(task: str, output_channels: int) -> CourtModelSpec:
    return CourtModelSpec(
        task=cast(CourtTask, task),
        in_channels=3,
        output_channels=output_channels,
        short_side=32,
    )


def _valid_keypoint_batch() -> dict[str, torch.Tensor]:
    return {
        "image": torch.zeros(1, 3, 8, 8),
        "heatmap": torch.zeros(1, 2, 8, 8),
        "keypoints": torch.zeros(1, 2, 2),
        "kp_visible": torch.ones(1, 2, dtype=torch.bool),
    }


def _run_training_boundary(
    adapter: CourtKeypointModelIO | CourtSegmentationModelIO | CourtLineModelIO,
    model: _CountingCourtModel,
    batch: dict[str, torch.Tensor],
) -> CourtTrainingResult:
    call = adapter.prepare_training_batch(batch)
    logits = model(*call.model_call.model_args)
    return adapter.training_result(logits, call)


def test_training_lifecycle_rejects_missing_fields_before_forward() -> None:
    adapter = CourtKeypointModelIO(_spec("kp", 2), focal_gamma=2.0)
    model = _CountingCourtModel(output_channels=2)
    adapter.validate_model_pair(model)
    pair = bind_model_io(model, adapter)

    with pytest.raises(CourtModelIOError, match="heatmap"):
        pair.run({"image": torch.zeros(1, 3, 8, 8)})

    assert model.calls == 0


def test_adapter_rejects_non_rgb_model_spec() -> None:
    with pytest.raises(CourtModelIOError, match="exactly three RGB channels"):
        CourtKeypointModelIO(
            CourtModelSpec(
                task="kp",
                in_channels=1,
                output_channels=2,
                short_side=32,
            ),
            focal_gamma=2.0,
        )


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (torch.zeros(1, 3, 8, 8, dtype=torch.float64), "torch.float32"),
        (
            torch.full((1, 3, 8, 8), -3.0),
            "ImageNet-normalized values",
        ),
        (
            torch.full((1, 3, 8, 8), 3.0),
            "ImageNet-normalized values",
        ),
    ],
)
def test_image_semantics_fail_before_forward(
    image: torch.Tensor,
    message: str,
) -> None:
    adapter = CourtKeypointModelIO(_spec("kp", 2), focal_gamma=2.0)
    model = _CountingCourtModel(output_channels=2)
    batch = _valid_keypoint_batch()
    batch["image"] = image

    with pytest.raises(CourtModelIOError, match=message):
        _run_training_boundary(adapter, model, batch)

    assert model.calls == 0


def test_keypoint_training_boundary_runs_validated_batch_once() -> None:
    adapter = CourtKeypointModelIO(_spec("kp", 2), focal_gamma=2.0)
    model = _CountingCourtModel(output_channels=2)

    result = _run_training_boundary(adapter, model, _valid_keypoint_batch())

    assert result.logits.shape == (1, 2, 8, 8)
    assert result.loss.ndim == 0
    assert model.calls == 1


@pytest.mark.parametrize(
    ("violation", "message"),
    [
        ("missing", "missing required field 'keypoints'"),
        ("dtype", "torch.float32"),
        ("rank", "Keypoint heatmap must have shape"),
        ("shape", "keypoints must have shape"),
        ("semantic", "kp_visible values must be in \[0, 1\]"),
    ],
)
def test_keypoint_contract_violations_fail_before_forward(
    violation: str,
    message: str,
) -> None:
    batch = _valid_keypoint_batch()
    if violation == "missing":
        del batch["keypoints"]
    elif violation == "dtype":
        batch["image"] = batch["image"].to(torch.uint8)
    elif violation == "rank":
        batch["heatmap"] = batch["heatmap"].squeeze(0)
    elif violation == "shape":
        batch["keypoints"] = torch.zeros(1, 2, 3)
    else:
        batch["kp_visible"] = torch.full((1, 2), 2.0)
    model = _CountingCourtModel(output_channels=2)

    with pytest.raises(CourtModelIOError, match=message):
        _run_training_boundary(
            CourtKeypointModelIO(_spec("kp", 2), focal_gamma=2.0),
            model,
            batch,
        )

    assert model.calls == 0


@pytest.mark.parametrize("task", ["seg", "line"])
def test_dense_mask_semantic_violation_fails_before_forward(task: str) -> None:
    model = _CountingCourtModel(output_channels=3 if task == "seg" else 1)
    adapter: CourtSegmentationModelIO | CourtLineModelIO
    if task == "seg":
        adapter = CourtSegmentationModelIO(
            _spec("seg", 3),
            ce_weight=1.0,
            dice_weight=1.0,
        )
        batch = {
            "image": torch.zeros(1, 3, 8, 8),
            "mask": torch.full((1, 8, 8), 3, dtype=torch.long),
        }
        message = "invalid class index"
    else:
        adapter = CourtLineModelIO(
            _spec("line", 1),
            bce_weight=1.0,
            dice_weight=1.0,
            pos_weight=1.0,
        )
        batch = {
            "image": torch.zeros(1, 3, 8, 8),
            "mask": torch.full((1, 1, 8, 8), 1.5),
        }
        message = "Line mask values must be in"

    with pytest.raises(CourtModelIOError, match=message):
        _run_training_boundary(adapter, model, batch)

    assert model.calls == 0


def test_keypoint_decode_returns_original_pixel_coordinates() -> None:
    adapter = CourtKeypointModelIO(_spec("kp", 1), focal_gamma=2.0)
    logits = torch.full((1, 1, 4, 5), -10.0)
    logits[0, 0, 2, 3] = 10.0

    prediction = adapter.decode_prediction(
        logits,
        original_size_hw=(7, 9),
        subpixel_refine=False,
        max_peaks=1,
    )

    assert isinstance(prediction, CourtKeypointPrediction)
    torch.testing.assert_close(prediction.keypoints, torch.tensor([[[6.0, 4.0]]]))
    assert prediction.scores.shape == prediction.valid.shape == (1, 1)
    assert prediction.covariance.shape == (1, 1, 2, 2)
    assert prediction.valid.all()
    assert prediction.heatmaps.shape == (1, 4, 5)


def test_dense_adapters_return_typed_predictions() -> None:
    seg_adapter = CourtSegmentationModelIO(
        _spec("seg", 3),
        ce_weight=1.0,
        dice_weight=1.0,
    )
    line_adapter = CourtLineModelIO(
        _spec("line", 1),
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=1.0,
    )

    seg = seg_adapter.decode_prediction(
        torch.zeros(1, 3, 4, 5),
        original_size_hw=(4, 5),
        subpixel_refine=False,
    )
    line = line_adapter.decode_prediction(
        torch.zeros(1, 1, 4, 5),
        original_size_hw=(4, 5),
        subpixel_refine=False,
    )

    assert isinstance(seg, CourtSegmentationPrediction)
    assert seg.mask.shape == (4, 5)
    assert isinstance(line, CourtLinePrediction)
    torch.testing.assert_close(line.probability, torch.full((4, 5), 0.5))
