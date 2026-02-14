"""Unified PyTorch Lightning module for PLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.base.training.lightning_module import BaseLightningModule
from src.plcs.models import build_plcs_model
from src.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.plcs.training.metrics import PLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSLightningModule(BaseLightningModule):
    """Lightning module for unified PLCS I/O training."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        self.model: nn.Module = build_plcs_model(self.config)

        loss_cfg_dict = self.config.get("loss", {})
        if loss_cfg_dict:
            loss_cfg = PLCSLossConfig.from_dict(dict(loss_cfg_dict))
        else:
            train_cfg = self.config.get("training", {})
            loss_cfg = PLCSLossConfig(
                position_weight=float(train_cfg.get("position_loss_weight", 1.0)),
                rotation_weight=float(train_cfg.get("rotation_loss_weight", 1.0)),
            )
        self.loss_fn = PLCSLoss(config=loss_cfg)

        metrics_cfg = self.config.get("metrics", {})
        self.train_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
            angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
        )
        self.val_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
            angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
        )
        self.test_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.get("position_threshold_m", 0.5)),
            angle_threshold_deg=float(metrics_cfg.get("angle_threshold_deg", 15.0)),
        )

    def _build_seq_mask(self, human_mask: Tensor | None) -> Tensor | None:
        if human_mask is None:
            return None
        if human_mask.dim() != 3:
            raise ValueError(
                f"human_mask must be (B,N,T), got shape {tuple(human_mask.shape)}"
            )
        return (human_mask > 0).any(dim=1)

    def _forward_from_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(
            human_kp=batch["human_kp"],
            court_kp=batch["court_kp"],
            human_vis=batch.get("human_vis"),
            human_mask=batch.get("human_mask"),
            court_vis=batch.get("court_vis"),
        )

    def _filter_valid_for_metrics(
        self,
        pred_position: Tensor,
        pred_rotation: Tensor,
        target_position: Tensor,
        target_rotation: Tensor,
        seq_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if pred_position.dim() == 2:
            return pred_position, pred_rotation, target_position, target_rotation

        if seq_mask is None:
            return (
                pred_position.reshape(-1, pred_position.shape[-1]),
                pred_rotation.reshape(-1, pred_rotation.shape[-1]),
                target_position.reshape(-1, target_position.shape[-1]),
                target_rotation.reshape(-1, target_rotation.shape[-1]),
            )

        valid = seq_mask.reshape(-1) > 0
        pred_pos_flat = pred_position.reshape(-1, pred_position.shape[-1])
        pred_rot_flat = pred_rotation.reshape(-1, pred_rotation.shape[-1])
        tgt_pos_flat = target_position.reshape(-1, target_position.shape[-1])
        tgt_rot_flat = target_rotation.reshape(-1, target_rotation.shape[-1])

        if valid.any():
            return (
                pred_pos_flat[valid],
                pred_rot_flat[valid],
                tgt_pos_flat[valid],
                tgt_rot_flat[valid],
            )

        # no valid tokens: return first entry to keep metric code safe
        return (
            pred_pos_flat[:1],
            pred_rot_flat[:1],
            tgt_pos_flat[:1],
            tgt_rot_flat[:1],
        )

    def _select_metrics(self, stage: str) -> PLCSMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        outputs = self._forward_from_batch(batch)
        seq_mask = self._build_seq_mask(batch.get("human_mask"))

        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            seq_mask=seq_mask,
        )

        pred_pos, pred_rot, tgt_pos, tgt_rot = self._filter_valid_for_metrics(
            outputs["position"],
            outputs["rotation"],
            batch["position"],
            batch["rotation"],
            seq_mask,
        )
        metrics = self._select_metrics(stage).update(
            pred_pos,
            pred_rot,
            tgt_pos,
            tgt_rot,
        )

        return losses["total"], {
            **metrics,
            **{f"loss_{k}": float(v.item()) for k, v in losses.items()},
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_error_m", metrics.get("position_error_m", 0.0), prog_bar=True)
        self.log("train/ang_error_deg", metrics.get("angular_error_deg", 0.0), prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos_error_m", metrics.get("position_error_m", 0.0), prog_bar=True)
        self.log("val/ang_error_deg", metrics.get("angular_error_deg", 0.0), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/pos_error_m", metrics.get("position_error_m", 0.0))
        self.log("test/ang_error_deg", metrics.get("angular_error_deg", 0.0))

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()
