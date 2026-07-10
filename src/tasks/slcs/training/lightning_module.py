"""Lightning module for SLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.slcs.models import build_slcs_model
from src.tasks.slcs.training.losses import SLCSLoss, SLCSLossConfig
from src.tasks.slcs.training.metrics import SLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig

_FORWARD_KEYS = (
    "player_kp",
    "player_kp_vis",
    "player_valid",
    "ball_uv",
    "ball_vis",
    "court_kp",
    "court_vis",
    "frame_mask",
    "dino_tokens",
    "dino_frame_idx",
    "dino_valid",
)


class SLCSLightningModule(BaseLightningModule):
    """Supervised training of :class:`SLCSFusionModel` on pseudo-labels."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)
        if config is None:
            raise ValueError("SLCSLightningModule requires a config.")
        self.model = build_slcs_model(config)
        loss_cfg = config.get("loss", {}) or {}
        self.loss_fn = SLCSLoss(SLCSLossConfig.from_dict(dict(loss_cfg)))
        self._metrics = {
            "train": SLCSMetrics(),
            "val": SLCSMetrics(),
            "test": SLCSMetrics(),
        }

    # ------------------------------------------------------------------

    def forward_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the model on an SLCSBatch dict."""
        missing = [k for k in _FORWARD_KEYS if k not in batch]
        if missing:
            raise KeyError(f"SLCS batch is missing forward keys: {missing}.")
        kwargs = {k: batch[k] for k in _FORWARD_KEYS}
        outputs: dict[str, Tensor] = self.model(**kwargs)
        return outputs

    def _step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        outputs = self.forward_batch(batch)
        losses = self.loss_fn(outputs, batch)
        batch_metrics = self._metrics[stage].update(outputs, batch)

        batch_size = int(batch["frame_mask"].shape[0])
        on_step = stage == "train"
        self.log(
            f"{stage}/loss",
            losses["total"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for name, value in losses.items():
            if name == "total":
                continue
            self.log(
                f"{stage}/loss_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        for name, value in batch_metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name.endswith("position_error_m"),
                batch_size=batch_size,
            )
        if stage == "test":
            self.collect_test_predictions(batch, {"outputs": outputs})
        return losses["total"]

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        return self._step(batch, "train")

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        return self._step(batch, "test")

    # ------------------------------------------------------------------
    # Epoch-end aggregation
    # ------------------------------------------------------------------

    def _log_epoch_metrics(self, stage: str) -> None:
        metrics = self._metrics[stage].compute()
        for name, value in metrics.items():
            self.log(f"{stage}/{name}_epoch", value, on_epoch=True)
        self._metrics[stage].reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        metrics = self._metrics["test"].compute()
        for name, value in metrics.items():
            self.log(f"test/{name}_epoch", value, on_epoch=True)
        self._metrics["test"].reset()
        self.save_test_predictions(metrics)

    # ------------------------------------------------------------------
    # Test prediction persistence (issue #533 repro bundles)
    # ------------------------------------------------------------------

    def test_prediction_payload(
        self, batch: Any, result: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        outputs = result["outputs"]
        payload_keys = {
            "pred_player_position": outputs["player_position"],
            "pred_player_rotation": outputs["player_rotation"],
            "pred_player_position_log_b": outputs["player_position_log_b"],
            "pred_player_rotation_log_b": outputs["player_rotation_log_b"],
            "pred_ball_position": outputs["ball_position"],
            "pred_ball_position_log_b": outputs["ball_position_log_b"],
            "target_player_position": batch["target_player_position"],
            "target_player_rotation": batch["target_player_rotation"],
            "target_player_valid": batch["target_player_valid"],
            "target_ball_position": batch["target_ball_position"],
            "target_ball_valid": batch["target_ball_valid"],
            "frame_mask": batch["frame_mask"],
        }
        return {k: self._to_numpy(v) for k, v in payload_keys.items()}

    # Keep val metrics deterministic when dino tokens are float16-derived.
    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        if "dino_tokens" in batch and batch["dino_tokens"].dtype != torch.float32:
            batch["dino_tokens"] = batch["dino_tokens"].float()
        return batch


__all__ = ["SLCSLightningModule"]
