"""Unit coverage for the shared fixed-query Lightning lifecycle."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.tasks.base.training.tracking_benchmark import TrackingFusionBenchmarkResult
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

    def tracking_prediction_result(
        self, result: TrackingStepResult[Tensor]
    ) -> dict[str, Any]:
        return {"prediction": result.prediction}


def test_shared_lifecycle_dispatches_metrics_logging_and_test_collection(
    make_training_config: Any,
    monkeypatch: Any,
) -> None:
    module = _TrackingModule(make_training_config())
    logged: list[str] = []
    collected: list[tuple[dict[str, Tensor], dict[str, Any]]] = []
    monkeypatch.setattr(
        module,
        "log",
        lambda name, value, **kwargs: logged.append(name),
    )
    monkeypatch.setattr(
        module,
        "collect_test_predictions",
        lambda batch, result: collected.append((batch, result)),
    )
    batch = {"prediction": torch.tensor([[1.0, 2.0]])}

    training_loss = module.training_step(batch, 0)
    validation_result = module.validation_step(batch, 0)
    test_result = module.test_step(batch, 0)

    torch.testing.assert_close(training_loss, torch.tensor(3.0))
    assert validation_result["prediction"] is batch["prediction"]
    assert test_result["prediction"] is batch["prediction"]
    assert module.metric_requests == [False, True, True]
    assert "train/quality" not in logged
    assert {"val/quality", "test/quality"}.issubset(logged)
    assert collected == [(batch, test_result)]


def test_test_epoch_end_merges_fusion_benchmark_metrics(
    make_training_config: Any,
    monkeypatch: Any,
) -> None:
    module = _TrackingModule(make_training_config())
    saved: list[dict[str, float]] = []
    monkeypatch.setattr(
        type(module),
        "trainer",
        property(
            lambda self: type(
                "Trainer",
                (),
                {"callback_metrics": {"test/loss": torch.tensor(0.25)}},
            )()
        ),
    )
    monkeypatch.setattr(
        module,
        "benchmark_court_peak_fusion",
        lambda: TrackingFusionBenchmarkResult(1.5, 2.5),
    )
    monkeypatch.setattr(
        module,
        "save_test_predictions",
        lambda metrics: saved.append(metrics),
    )

    module.on_test_epoch_end()

    assert saved == [
        {
            "loss": 0.25,
            "court_peak_fusion_latency_ms": 1.5,
            "court_peak_fusion_peak_memory_mb": 2.5,
        }
    ]
