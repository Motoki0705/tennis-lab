"""Unit coverage for the shared fixed-query Lightning lifecycle."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor

from src.tasks.base.training.metric_logging import (
    MetricContractError,
    ScalarMetricStatistic,
    evaluation_only_metric_logging_contract,
)
from src.tasks.base.training.tracking_lightning_module import (
    TrackingLightningModule,
    TrackingStepResult,
)


class _TrackingModule(TrackingLightningModule[Tensor]):
    metric_logging_contract = evaluation_only_metric_logging_contract(
        "tracking fixture",
        headline_keys=("quality",),
        progress_bar_keys=("quality",),
    )

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
        denominator = prediction.new_tensor(float(prediction.shape[0]))
        return TrackingStepResult(
            losses={"total": prediction.sum(), "position": prediction.mean()},
            metrics=(
                {"quality": prediction.mean(), "axis_detail": prediction.max()}
                if compute_metrics
                else {}
            ),
            prediction=prediction,
            statistics=(
                {
                    "quality": ScalarMetricStatistic(
                        prediction.mean() * denominator,
                        denominator,
                    ),
                    "axis_detail": ScalarMetricStatistic(
                        prediction.max() * denominator,
                        denominator,
                    ),
                }
                if compute_metrics
                else None
            ),
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
    monkeypatch.setattr(module, "_reset_test_prediction_buffer", lambda: None)
    monkeypatch.setattr(module, "save_test_predictions", lambda **kwargs: None)
    batch = {
        "prediction": torch.tensor([[1.0, 2.0]]),
        "target_position": torch.zeros(1, 1, 1, 3),
    }

    training_loss = module.training_step(batch, 0)
    module.on_validation_epoch_start()
    validation_result = module.validation_step(batch, 0)
    module.on_validation_epoch_end()
    module.on_test_epoch_start()
    test_result = module.test_step(batch, 0)
    module.on_test_epoch_end()

    torch.testing.assert_close(training_loss, torch.tensor(3.0))
    assert validation_result["prediction"] is batch["prediction"]
    assert test_result["prediction"] is batch["prediction"]
    assert module.metric_requests == [False, True, True]
    logged_names = {name for name, _ in logged}
    assert "train/quality" not in logged_names
    assert {"val/quality", "test/quality"}.issubset(logged_names)
    assert "val/axis_detail" not in logged_names
    assert "test/loss_position" not in logged_names
    assert all(
        kwargs.get("batch_size", 1) == 1
        for _, kwargs in logged
    )
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


def test_tracking_test_artifacts_split_headlines_and_weighted_diagnostics(
    make_training_config: Any,
    monkeypatch: Any,
) -> None:
    module = _TrackingModule(make_training_config())
    captured: dict[str, Any] = {}
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_reset_test_prediction_buffer", lambda: None)
    monkeypatch.setattr(module, "collect_test_predictions", lambda *args: None)
    monkeypatch.setattr(
        module,
        "save_test_predictions",
        lambda **kwargs: captured.update(kwargs),
    )
    module.on_test_epoch_start()

    module.test_step(
        {
            "prediction": torch.tensor([[1.0, 3.0], [1.0, 3.0]]),
            "target_position": torch.zeros(2, 1, 1, 3),
        },
        0,
    )
    module.test_step(
        {
            "prediction": torch.tensor([[7.0, 9.0]]),
            "target_position": torch.zeros(1, 1, 1, 3),
        },
        1,
    )
    module.on_test_epoch_end()

    assert captured["metrics"] == {"quality": pytest.approx(4.0)}
    assert captured["diagnostic_metrics"] == {
        "axis_detail": pytest.approx(5.0),
        "loss_position": pytest.approx(4.0),
    }


def test_tracking_eval_missing_headline_is_an_error(
    make_training_config: Any,
    monkeypatch: Any,
) -> None:
    module = _TrackingModule(make_training_config())
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "compute_tracking_step",
        lambda batch, *, compute_metrics: TrackingStepResult(
            losses={"total": torch.tensor(0.0)},
            metrics={},
            prediction=batch["prediction"],
            statistics={},
        ),
    )

    module.on_validation_epoch_start()
    module.validation_step(
        {
            "prediction": torch.ones(1, 1),
            "target_position": torch.zeros(1, 1, 1, 3),
        },
        0,
    )
    with pytest.raises(MetricContractError, match="missing required headline"):
        module.on_validation_epoch_end()


def test_tracking_required_headline_with_no_denominator_is_an_error(
    make_training_config: Any,
    monkeypatch: Any,
) -> None:
    module = _TrackingModule(make_training_config())
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "compute_tracking_step",
        lambda batch, *, compute_metrics: TrackingStepResult(
            losses={"total": torch.tensor(0.0)},
            metrics={"quality": torch.tensor(0.0)},
            prediction=batch["prediction"],
            statistics={
                "quality": ScalarMetricStatistic(
                    torch.tensor(0.0), torch.tensor(0.0)
                )
            },
        ),
    )
    module.on_validation_epoch_start()
    module.validation_step(
        {
            "prediction": torch.ones(1, 1),
            "target_position": torch.zeros(1, 1, 1, 3),
        },
        0,
    )

    with pytest.raises(MetricContractError, match="missing required headline"):
        module.on_validation_epoch_end()
