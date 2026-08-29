"""Unit coverage for the shared fixed-query Lightning lifecycle."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)


class _TrackingModule(TrackingLightningModule[Tensor]):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.metric_requests: list[bool] = []

    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[Tensor]:
        self.metric_requests.append(compute_metrics)
        prediction = batch["prediction"]
        return TrackingStepResult(
            losses={"total": prediction.sum(), "position": prediction.mean()},
            metrics={"quality": prediction.mean()} if compute_metrics else {},
            prediction=prediction,
        )

    def tracking_prediction_result(self, prediction: Tensor) -> dict[str, Any]:
        return {"prediction": prediction}


def test_shared_lifecycle_dispatches_metrics_logging_and_test_collection(
    make_training_config: Any,
    monkeypatch: Any,
) -> None:
    module = _TrackingModule(make_training_config())
    logged: list[tuple[str, dict[str, Any]]] = []
    collected: list[tuple[dict[str, Tensor], dict[str, Any]]] = []
    monkeypatch.setattr(
        module,
        "log",
        lambda name, value, **kwargs: logged.append((name, kwargs)),
    )
    monkeypatch.setattr(
        module,
        "collect_test_predictions",
        lambda batch, result: collected.append((batch, result)),
    )
    batch = {
        "prediction": torch.tensor([[1.0, 2.0]]),
        "target_position": torch.zeros(1, 1, 1, 3),
    }

    training_loss = module.training_step(batch, 0)
    validation_result = module.validation_step(batch, 0)
    test_result = module.test_step(batch, 0)

    torch.testing.assert_close(training_loss, torch.tensor(3.0))
    assert validation_result["prediction"] is batch["prediction"]
    assert test_result["prediction"] is batch["prediction"]
    assert module.metric_requests == [False, True, True]
    logged_names = {name for name, _ in logged}
    assert "train/quality" not in logged_names
    assert {"val/quality", "test/quality"}.issubset(logged_names)
    assert all(kwargs["batch_size"] == 1 for _, kwargs in logged)
    assert all(kwargs["on_epoch"] is True for _, kwargs in logged)
    assert collected == [(batch, test_result)]


@pytest.mark.parametrize(
    "batch",
    [
        {"prediction": torch.ones(1)},
        {
            "prediction": torch.ones(1),
            "target_position": torch.tensor(0.0),
        },
        {
            "prediction": torch.ones(0),
            "target_position": torch.empty(0, 1, 1, 3),
        },
    ],
    ids=["missing", "scalar", "empty"],
)
def test_shared_lifecycle_rejects_missing_or_invalid_batch_axis(
    make_training_config: Any,
    batch: dict[str, Tensor],
) -> None:
    module = _TrackingModule(make_training_config())

    with pytest.raises(ValueError, match="target_position"):
        module.validation_step(batch, 0)
    assert module.metric_requests == []
