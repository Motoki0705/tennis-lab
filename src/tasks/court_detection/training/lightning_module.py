"""PyTorch Lightning module for court detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.model_io.adapters import CourtModelIOAdapter
from src.tasks.court_detection.model_io.contracts import CourtTrainingResult
from src.tasks.court_detection.model_io.factory import build_court_detection_pair
from src.tasks.court_detection.training.metrics import CourtDetectionMetrics
from src.tasks.court_detection.visualization.adapters.render_inputs import (
    CourtQualitativeRenderer,
    build_court_qualitative_renderer,
)


class CourtDetectionLightningModule(BaseLightningModule):
    """Unified Lightning module for court detection tasks.

    Supports three tasks via ``config.data.task``:

    * ``seg`` — Court cell segmentation (CE + Dice, 7 classes).
    * ``kp``  — Court keypoint heatmap regression (Focal BCE, 14 channels).
    * ``line`` — Court white-line segmentation (BCE + Dice, 1 channel).

    Inherits optimizer/scheduler logic from
    :class:`~src.tasks.base.training.lightning_module.BaseLightningModule`.
    """

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.save_hyperparameters()
        runtime = CourtTrainingConfig.from_config(config)
        self.qualitative_fps = runtime.qualitative_fps
        self.qualitative_style = runtime.render_style

        model_pair = build_court_detection_pair(self.config)
        self.model = model_pair.model
        self.model_io = cast(CourtModelIOAdapter, model_pair.adapter)
        self.qualitative_renderer: CourtQualitativeRenderer = (
            build_court_qualitative_renderer(self.model_io)
        )

        # Metrics
        self.train_metrics = CourtDetectionMetrics(
            self.model_io.spec.task, runtime.data.output_channels
        )
        self.val_metrics = CourtDetectionMetrics(
            self.model_io.spec.task, runtime.data.output_channels
        )
        self.test_metrics = CourtDetectionMetrics(
            self.model_io.spec.task, runtime.data.output_channels
        )

    def forward(self, *model_args: Tensor) -> Tensor:
        """Compute over a model-I/O boundary-prepared argument tuple."""
        return cast(Tensor, self.model(*model_args))

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> CourtTrainingResult:
        """Shared computation for train/val/test steps."""
        call = self.model_io.prepare_training_batch(batch)
        logits = self.model(*call.model_call.model_args)
        result = self.model_io.training_result(logits, call)
        self.log(f"{stage}/loss", result.loss, prog_bar=True, sync_dist=True)
        return result

    def _metric_tracker_for_stage(self, stage: str) -> CourtDetectionMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _flush_stage_metrics(self, stage: str) -> None:
        metrics = self._metric_tracker_for_stage(stage).compute()
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=(stage == "val" and name in ("miou", "mean_dist", "dice")),
            )
        self._metric_tracker_for_stage(stage).reset()

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self._shared_step(batch, "train")
        self.train_metrics.update(outputs.logits, batch)
        return outputs.loss

    def on_train_epoch_end(self) -> None:
        self._flush_stage_metrics("train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        outputs = self._shared_step(batch, "val")
        self.val_metrics.update(outputs.logits, batch)

    def on_validation_epoch_end(self) -> None:
        self._flush_stage_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._reset_test_prediction_buffer()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        outputs = self._shared_step(batch, "test")
        self.test_metrics.update(outputs.logits, batch)
        self.collect_test_predictions(batch, {"logits": outputs.logits})

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        saved = self.save_test_predictions(metrics=metrics)
        if saved is not None:
            print(f"[test] saved test-split predictions -> {saved}")
        self._flush_stage_metrics("test")

    def test_prediction_payload(
        self, batch: dict[str, Any], result: dict[str, Tensor]
    ) -> dict[str, Any]:
        """Persist court predictions from ``data/court/data_val.json``."""
        return self.model_io.test_payload(batch, result["logits"])

    # ------------------------------------------------------------------
    # Qualitative validation logging
    # ------------------------------------------------------------------

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        """Render court detection panels via shared rendering/ layer (pred-only).

        The output adapter and renderer were selected once in ``__init__``.
        """
        device = next(self.parameters()).device
        style = self.qualitative_style.build()

        for batch_idx, batch in enumerate(batches):
            moved_batch = {
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            with torch.no_grad():
                call = self.model_io.prepare_training_batch(moved_batch)
                logits = self.model(*call.model_call.model_args).cpu()
            self.model_io.validate_logits(logits, call.model_call)
            frames_rgb = self.qualitative_renderer.render(
                batch=batch,
                logits=logits,
                style=style,
                clip_label=f"court_{self.model_io.spec.task}",
            )

            save_qualitative_clip(
                frames_rgb=frames_rgb,
                artifact_dir=artifact_dir,
                name=f"court_batch{batch_idx:02d}",
                tb_writer=tb_writer,
                tag=f"qualitative/court_detection/batch{batch_idx:02d}",
                global_step=global_step,
                fps=self.qualitative_fps,
            )
