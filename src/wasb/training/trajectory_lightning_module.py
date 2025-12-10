from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.wasb.models.trajectory_completer import BiLSTMCompleter

if TYPE_CHECKING:
    from omegaconf import DictConfig


class TrajectoryBiLSTMLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: DictConfig | dict | None = None,
        steps_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["steps_per_epoch"])
        self.config = config or {}
        self.steps_per_epoch = steps_per_epoch

        train_cfg = self.config.get("training", {})
        model_cfg = self.config.get("model", {})

        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.max_epochs = train_cfg.get("max_epochs", 50)
        self.min_lr = train_cfg.get("min_lr", 1e-6)

        self.lambda_block = float(train_cfg.get("lambda_block", 1.0))
        self.lambda_sparse = float(train_cfg.get("lambda_sparse", 1.0))
        self.lambda_noise = float(train_cfg.get("lambda_noise", 1.0))

        hidden_dim = int(model_cfg.get("hidden_dim", 64))
        num_layers = int(model_cfg.get("num_layers", 2))
        dropout = float(model_cfg.get("dropout", 0.1))
        score_threshold = float(model_cfg.get("score_threshold", 0.5))

        self.completer = BiLSTMCompleter(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            score_threshold=score_threshold,
            device="cpu",
        )
        self.completer._build_model()
        assert self.completer._model is not None
        self.model = self.completer._model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @staticmethod
    def _masked_mean(loss: Tensor, mask: Tensor) -> Tensor:
        mask = mask.to(dtype=loss.dtype, device=loss.device)
        denom = mask.sum()
        if denom <= 0:
            return torch.zeros((), dtype=loss.dtype, device=loss.device)
        return (loss * mask).sum() / (denom + 1e-8)

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        xy_input_px: Tensor = batch["xy_input_px"]
        target_xy_px: Tensor = batch["target_xy_px"]
        visibility_input: Tensor = batch["visibility_input"]
        scores_input: Tensor = batch["scores_input"]
        loss_mask_block: Tensor = batch["loss_mask_block"]
        loss_mask_sparse: Tensor = batch["loss_mask_sparse"]
        loss_mask_noise: Tensor = batch["loss_mask_noise"]

        device = self.device
        xy_input_px = xy_input_px.to(device)
        target_xy_px = target_xy_px.to(device)
        visibility_input = visibility_input.to(device)
        scores_input = scores_input.to(device)
        loss_mask_block = loss_mask_block.to(device)
        loss_mask_sparse = loss_mask_sparse.to(device)
        loss_mask_noise = loss_mask_noise.to(device)

        scale = torch.tensor([1920.0, 1080.0], dtype=torch.float32, device=device)

        xy_input_norm = xy_input_px / scale
        target_xy_norm = target_xy_px / scale

        x_components = [
            xy_input_norm[..., 0],
            xy_input_norm[..., 1],
            visibility_input.to(dtype=torch.float32),
            scores_input.to(dtype=torch.float32),
        ]
        model_in = torch.stack(x_components, dim=-1)

        pred_norm = self(model_in)
        diff = pred_norm - target_xy_norm
        mse_per_frame = (diff * diff).sum(dim=-1)

        loss_block = self._masked_mean(mse_per_frame, loss_mask_block)
        loss_sparse = self._masked_mean(mse_per_frame, loss_mask_sparse)
        loss_noise = self._masked_mean(mse_per_frame, loss_mask_noise)

        total_loss = (
            self.lambda_block * loss_block
            + self.lambda_sparse * loss_sparse
            + self.lambda_noise * loss_noise
        )

        total_mask = (loss_mask_block + loss_mask_sparse + loss_mask_noise) > 0
        rmse_px = torch.zeros((), dtype=torch.float32, device=device)
        if total_mask.any():
            pred_px = pred_norm * scale
            diff_px = pred_px - target_xy_px
            sq = (diff_px * diff_px).sum(dim=-1)
            rmse_px = torch.sqrt(
                self._masked_mean(sq, total_mask.to(dtype=torch.float32))
            )

        metrics = {
            "loss_total": total_loss.detach().item(),
            "loss_block": loss_block.detach().item(),
            "loss_sparse": loss_sparse.detach().item(),
            "loss_noise": loss_noise.detach().item(),
            "rmse_px": rmse_px.detach().item(),
        }
        return total_loss, metrics

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_block", metrics["loss_block"], prog_bar=False)
        self.log("train/loss_sparse", metrics["loss_sparse"], prog_bar=False)
        self.log("train/loss_noise", metrics["loss_noise"], prog_bar=False)
        self.log("train/rmse_px", metrics["rmse_px"], prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/loss_block", metrics["loss_block"], prog_bar=False)
        self.log("val/loss_sparse", metrics["loss_sparse"], prog_bar=False)
        self.log("val/loss_noise", metrics["loss_noise"], prog_bar=False)
        self.log("val/rmse_px", metrics["rmse_px"], prog_bar=True)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/loss_block", metrics["loss_block"])
        self.log("test/loss_sparse", metrics["loss_sparse"])
        self.log("test/loss_noise", metrics["loss_noise"])
        self.log("test/rmse_px", metrics["rmse_px"])

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = 1000

        total_steps = steps_per_epoch * max(self.max_epochs, 1)
        if total_steps <= self.warmup_steps + 1:
            return {"optimizer": optimizer}

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - self.warmup_steps, 1),
            eta_min=self.min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
