"""Lightning fine-tuning of DINO for tennis-player detection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.player_detection.configuration import PlayerTrainingConfig
from src.tasks.player_detection.data.detection_dataset import DetectionBatch
from src.tasks.player_detection.evaluation.metrics import (
    HEADLINE_METRICS,
    PlayerDetectionMetrics,
)
from src.tasks.player_detection.models.dino_detector import (
    backbone_parameter_names,
    build_player_dino,
    decode_player_detections,
)

# Final-decoder-layer components logged besides the weighted total.
_LOGGED_LOSSES = ("loss_ce", "loss_bbox", "loss_giou", "loss_ce_dn", "loss_bbox_dn", "loss_giou_dn")


class PlayerDetectionLightningModule(BaseLightningModule):
    def __init__(self, config: Any, runtime: PlayerTrainingConfig) -> None:
        super().__init__(config)
        if not torch.cuda.is_available():
            raise RuntimeError("DINO fine-tuning requires CUDA (upstream denoising uses .cuda()).")
        self.runtime = runtime
        self.model, self.criterion = build_player_dino(runtime.model, device="cuda")
        self.weight_dict = dict(cast("dict[str, float]", self.criterion.weight_dict))
        self.val_metrics = PlayerDetectionMetrics(runtime.evaluation)
        self.test_metrics = PlayerDetectionMetrics(runtime.evaluation)

    def optimizer_param_groups(self) -> list[dict[str, Any]]:
        backbone = backbone_parameter_names(self.model)
        named = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        groups: list[dict[str, Any]] = [
            {"params": [p for name, p in named if name not in backbone]},
            {
                "params": [p for name, p in named if name in backbone],
                "lr": self.learning_rate * self.runtime.model.backbone_lr_scale,
            },
        ]
        return [group for group in groups if group["params"]]

    @staticmethod
    def _targets(batch: DetectionBatch) -> list[dict[str, torch.Tensor]]:
        return [
            {"labels": labels, "boxes": boxes}
            for labels, boxes in zip(batch.labels, batch.boxes_cxcywh, strict=True)
        ]

    def training_step(self, batch: DetectionBatch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        targets = self._targets(batch)
        outputs = self.model(batch.images, targets)
        losses: dict[str, torch.Tensor] = self.criterion(outputs, targets)
        loss = torch.stack(
            [losses[name] * weight for name, weight in self.weight_dict.items() if name in losses]
        ).sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite DINO loss: { {k: float(v) for k, v in losses.items()} }")
        size = len(batch)
        self.log("train/loss", loss, prog_bar=True, batch_size=size)
        for name in _LOGGED_LOSSES:
            self.log(f"train_components/{name}", losses[name], batch_size=size)
        return loss

    def _evaluate(self, batch: DetectionBatch, metrics: PlayerDetectionMetrics) -> list[Any]:
        outputs = self.model(batch.images, None)
        detections = decode_player_detections(
            outputs,
            batch.original_sizes,
            max_detections=self.runtime.evaluation.max_detections,
        )
        metrics.update(detections, [boxes.cpu() for boxes in batch.boxes_xyxy_px])
        return detections

    def validation_step(self, batch: DetectionBatch, batch_idx: int) -> None:
        del batch_idx
        self._evaluate(batch, self.val_metrics)

    def on_validation_epoch_end(self) -> None:
        headline, _ = self.val_metrics.compute()
        self.val_metrics.reset()
        for name in HEADLINE_METRICS:
            self.log(f"val/{name}", headline[name], prog_bar=name in {"map", "f1"})

    def on_test_epoch_start(self) -> None:
        self._reset_test_prediction_buffer()
        self.test_metrics.reset()

    def test_step(self, batch: DetectionBatch, batch_idx: int) -> None:
        del batch_idx
        detections = self._evaluate(batch, self.test_metrics)
        self.collect_test_predictions(
            batch,
            {
                "boxes_xyxy": torch.stack([d.boxes_xyxy for d in detections]),
                "scores": torch.stack([d.scores for d in detections]),
            },
        )

    def test_prediction_payload(self, batch: Any, result: dict[str, Any]) -> dict[str, np.ndarray]:
        del batch
        return {
            "boxes_xyxy": result["boxes_xyxy"].numpy(),
            "scores": result["scores"].numpy(),
        }

    def on_test_epoch_end(self) -> None:
        headline, diagnostics = self.test_metrics.compute()
        for name in HEADLINE_METRICS:
            self.log(f"test/{name}", headline[name])
        self.save_test_predictions(
            metrics={f"test/{name}": value for name, value in headline.items()},
            diagnostic_metrics={f"test/{name}": value for name, value in diagnostics.items()},
        )
