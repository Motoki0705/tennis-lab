"""Unit tests for ball detection Lightning test-prediction payloads."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.tasks.ball_detection.model_io.contracts import (
    BallModelCall,
    BallTrainingCall,
)
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.ball_detection.training.staged_lightning_module import (
    StagedBallDetectionLightningModule,
)
from src.tasks.base.training.metric_logging import WeightedMetricAccumulator

pytestmark = pytest.mark.unit


def test_metric_logging_contract_covers_all_ball_metrics_and_staged_inherits_it() -> (
    None
):
    contract = BallDetectionLightningModule.metric_logging_contract

    for stage in ("train", "val", "test"):
        assert contract.for_stage(stage).headline_keys == (
            "precision",
            "recall",
            "f1",
            "mean_distance_px",
        )
    assert StagedBallDetectionLightningModule.metric_logging_contract is contract


def test_test_artifact_path_uses_ball_metric_logging_contract() -> None:
    class _ArtifactBallModule(BallDetectionLightningModule):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.test_metrics = BallDetectionMetrics(
                peak_threshold=0.5,
                ball_distance_threshold=5.0,
                nms_kernel=3,
                max_predictions_per_frame=1,
                subpixel_refine=False,
            )
            self.test_metrics.tp += 3
            self.test_metrics.fp += 1
            self.test_metrics.fn += 2
            self.test_metrics.distance_sum += 10.5
            self.test_metrics.distance_count += 3
            self._test_metric_diagnostic_accumulator = WeightedMetricAccumulator()
            self.saved: dict[str, Any] = {}

        def save_test_predictions(
            self,
            metrics: dict[str, Any] | None = None,
            diagnostic_metrics: dict[str, Any] | None = None,
        ) -> None:
            self.saved = {
                "metrics": metrics,
                "diagnostic_metrics": diagnostic_metrics,
            }

        def _flush_stage_metrics(self, stage: str) -> None:
            assert stage == "test"
            self.test_metrics.reset()

    module = _ArtifactBallModule()

    module.on_test_epoch_end()

    assert module.saved["diagnostic_metrics"] == {}
    assert module.saved["metrics"] == pytest.approx(
        {
            "precision": 0.75,
            "recall": 0.6,
            "f1": 2 * 0.75 * 0.6 / (0.75 + 0.6),
            "mean_distance_px": 3.5,
        }
    )
    assert module.test_metrics.tp.item() == 0.0


def test_gt_trajectory_marks_frames_without_a_visible_ball_as_padding() -> None:
    module = object.__new__(BallDetectionLightningModule)
    coords = torch.tensor(
        [
            [
                [[1.0, 2.0], [5.0, 10.0]],
                [[3.0, 4.0], [7.0, 12.0]],
                [[9.0, 16.0], [2.0, 6.0]],
            ]
        ]
    )
    visibility = torch.tensor([[[0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]])
    images = torch.zeros(1, 3, 3, 2, 2)
    call = BallTrainingCall(
        model_call=BallModelCall(
            images=images,
            model_input=images,
            model_args=(images,),
            batch_size=1,
            frame_count=3,
        ),
        target_heatmaps=torch.zeros(1, 3, 2, 2),
        coords=coords,
        visibility=visibility,
        original_size=torch.tensor([[11.0, 21.0]]),
    )

    trajectory, padding_mask = module._extract_gt_trajectory(call)

    expected_trajectory = torch.tensor([[[0.5, 0.5], [0.3, 0.2], [0.9, 0.8]]])
    torch.testing.assert_close(trajectory, expected_trajectory)
    assert torch.equal(padding_mask, torch.tensor([[False, True, False]]))


def test_test_prediction_payload_persists_tracknet_predictions_and_targets() -> None:
    module = object.__new__(BallDetectionLightningModule)
    pred_heatmaps = torch.rand(2, 3, 4, 5)
    batch = {
        "coords": torch.rand(2, 3, 4, 2),
        "visibility": torch.ones(2, 3, 4),
        "original_size": torch.tensor([[1280.0, 720.0], [1280.0, 720.0]]),
        "heatmap_size": torch.tensor([[5.0, 4.0], [5.0, 4.0]]),
        "window_id": ["Game1/Clip1:0", "Game1/Clip1:4"],
    }

    payload = module.test_prediction_payload(
        batch,
        {"pred_heatmaps": pred_heatmaps},
    )

    assert payload["pred_heatmaps"] is pred_heatmaps
    assert payload["window_id"] == ["Game1/Clip1:0", "Game1/Clip1:4"]
    assert payload["target_coords"] is batch["coords"]
    assert payload["target_visibility"] is batch["visibility"]
    assert payload["original_size"] is batch["original_size"]
    assert payload["heatmap_size"] is batch["heatmap_size"]
