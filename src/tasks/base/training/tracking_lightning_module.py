"""Shared Lightning lifecycle for fixed-query tracking tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule

PredictionT = TypeVar("PredictionT")


@dataclass(frozen=True, slots=True)
class TrackingStepResult(Generic[PredictionT]):
    """Task-computed tensors consumed by the shared Lightning lifecycle."""

    losses: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    prediction: PredictionT


class TrackingLightningModule(BaseLightningModule, ABC, Generic[PredictionT]):
    """Own stage dispatch, logging, prediction collection, and test finalization.

    Task subclasses retain model-I/O preparation, matching, losses, metrics, and
    prediction payload schemas behind the two hooks below.
    """

    @abstractmethod
    def compute_tracking_step(
        self,
        batch: dict[str, Tensor],
        *,
        compute_metrics: bool,
    ) -> TrackingStepResult[PredictionT]:
        """Execute one task-specific validated model/loss step."""

    @abstractmethod
    def tracking_prediction_result(
        self, prediction: PredictionT
    ) -> dict[str, Any]:
        """Map a typed prediction to the task's Lightning result contract."""

    def _shared_tracking_step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> TrackingStepResult[PredictionT]:
        result = self.compute_tracking_step(
            batch,
            compute_metrics=stage != "train",
        )
        self.log(
            f"{stage}/loss",
            result.losses["total"],
            on_step=stage == "train",
            on_epoch=True,
        )
        for name, value in result.losses.items():
            if name != "total":
                self.log(
                    f"{stage}/loss_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                )
        for name, value in result.metrics.items():
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True)
        return result

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_tracking_step(batch, "train").losses["total"]

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Any]:
        del batch_idx
        result = self._shared_tracking_step(batch, "val")
        return self.tracking_prediction_result(result.prediction)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Any]:
        del batch_idx
        result = self._shared_tracking_step(batch, "test")
        output = self.tracking_prediction_result(result.prediction)
        self.collect_test_predictions(batch, output)
        return output

    def on_test_epoch_end(self) -> None:
        metrics = {
            key.removeprefix("test/"): float(value.detach().cpu())
            for key, value in self.trainer.callback_metrics.items()
            if key.startswith("test/") and isinstance(value, Tensor)
        }
        self.save_test_predictions(metrics)


__all__ = ["TrackingLightningModule", "TrackingStepResult"]
