"""Lightning module for the multi-ball track-query baseline."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch

from src.tasks.ball_tracking.models import BallTrackQueryModel
from src.tasks.ball_tracking.training.losses import BallTrackingLoss
from src.tasks.ball_tracking.training.metrics import ball_tracking_metrics
from src.tasks.base.training.lightning_module import BaseLightningModule


class BallTrackingLightningModule(BaseLightningModule):
    """Train and evaluate multi-ball clip-local slots."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = BallTrackQueryModel(config.model)
        self.criterion = BallTrackingLoss(config.loss)

    @staticmethod
    def _model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys = (
            "ball_uv",
            "ball_score",
            "ball_candidate_mask",
            "ball_visible",
            "frame_mask",
            "view_mask",
        )
        return {key: batch[key] for key in keys}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return cast(dict[str, torch.Tensor], self.model(**self._model_inputs(batch)))

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self(batch)
        losses, assignments = self.criterion(prediction, batch)
        self.log(
            f"{stage}/loss", losses["total"], on_step=stage == "train", on_epoch=True
        )
        for name, value in losses.items():
            if name != "total":
                self.log(f"{stage}/loss_{name}", value, on_step=False, on_epoch=True)
        if stage != "train":
            for name, value in ball_tracking_metrics(
                prediction, batch, assignments
            ).items():
                self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True)
        return losses["total"], prediction

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        _, prediction = self._shared_step(batch, "val")
        return prediction

    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        _, prediction = self._shared_step(batch, "test")
        self.collect_test_predictions(batch, prediction)
        return prediction

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        return {
            "pred_position": self._to_numpy(result["position"]),
            "pred_presence_logits": self._to_numpy(result["presence_logits"]),
            "target_position": self._to_numpy(batch["position_3d"]),
            "target_presence": self._to_numpy(batch["ball_present"]),
            "frame_mask": self._to_numpy(batch["frame_mask"]),
        }

    def on_test_epoch_end(self) -> None:
        metrics = {
            key.removeprefix("test/"): float(value.detach().cpu())
            for key, value in self.trainer.callback_metrics.items()
            if key.startswith("test/") and isinstance(value, torch.Tensor)
        }
        self.save_test_predictions(metrics)
