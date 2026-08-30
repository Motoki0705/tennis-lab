"""Shared Lightning lifecycle for fixed-query tracking tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.metric_logging import (
    MetricContractError,
    MetricLoggingContract,
    MetricPartition,
    MetricStatisticsAccumulator,
    ScalarMetricStatistic,
)

PredictionT = TypeVar("PredictionT")


@dataclass(frozen=True, slots=True)
class TrackingStepResult(Generic[PredictionT]):
    """Task-computed tensors consumed by the shared Lightning lifecycle."""

    losses: Mapping[str, Tensor]
    metrics: Mapping[str, Tensor]
    prediction: PredictionT
    statistics: Mapping[str, ScalarMetricStatistic] | None = None


class TrackingLightningModule(BaseLightningModule, ABC, Generic[PredictionT]):
    """Own stage dispatch, logging, prediction collection, and test finalization.

    Task subclasses retain model-I/O preparation, matching, losses, metrics, and
    prediction payload schemas behind the two hooks below.
    """

    metric_logging_contract: MetricLoggingContract

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        contract = getattr(self, "metric_logging_contract", None)
        if not isinstance(contract, MetricLoggingContract):
            raise TypeError(
                f"{type(self).__name__} must define a MetricLoggingContract."
            )
        self._validation_metric_accumulator = MetricStatisticsAccumulator()
        self._test_metric_accumulator = MetricStatisticsAccumulator()

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
        target_position = batch.get("target_position")
        if target_position is None:
            raise ValueError(
                "tracking batch must contain target_position with a batch axis."
            )
        if not isinstance(target_position, Tensor):
            raise ValueError("tracking batch target_position must be a Tensor.")
        if target_position.ndim == 0 or target_position.shape[0] <= 0:
            raise ValueError(
                "tracking batch target_position must have a non-empty batch axis."
            )
        batch_size = int(target_position.shape[0])
        result = self.compute_tracking_step(
            batch,
            compute_metrics=stage != "train",
        )
        self.log(
            f"{stage}/loss",
            result.losses["total"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=stage != "test",
        )
        if stage == "train":
            if result.statistics:
                raise MetricContractError(
                    "Tracking training steps must not compute metric statistics."
                )
            return result
        if result.statistics is None:
            raise MetricContractError(
                f"Tracking {stage} step did not provide metric statistics."
            )
        accumulator = (
            self._validation_metric_accumulator
            if stage == "val"
            else self._test_metric_accumulator
        )
        accumulator.update(result.statistics)
        if stage == "test":
            accumulator.update(
                {
                    f"loss_{name}": ScalarMetricStatistic.from_mean(
                        value, weight=batch_size
                    )
                    for name, value in result.losses.items()
                    if name != "total"
                }
            )
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

    def on_test_epoch_start(self) -> None:
        self._reset_test_prediction_buffer()
        self._test_metric_accumulator.reset()

    def on_validation_epoch_start(self) -> None:
        self._validation_metric_accumulator.reset()

    def _finalize_tracking_metrics(self, stage: str) -> MetricPartition:
        accumulator = (
            self._validation_metric_accumulator
            if stage == "val"
            else self._test_metric_accumulator
        )
        partition = self.metric_logging_contract.partition(
            stage,
            accumulator.compute(),
        )
        stage_contract = self.metric_logging_contract.for_stage(stage)
        for name, value in partition.headline.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in stage_contract.progress_bar_keys,
            )
        return partition

    def on_validation_epoch_end(self) -> None:
        self._finalize_tracking_metrics("val")

    def on_test_epoch_end(self) -> None:
        partition = self._finalize_tracking_metrics("test")
        self.save_test_predictions(
            metrics={key: float(value) for key, value in partition.headline.items()},
            diagnostic_metrics={
                key: float(value) for key, value in partition.diagnostics.items()
            },
        )


__all__ = ["TrackingLightningModule", "TrackingStepResult"]
