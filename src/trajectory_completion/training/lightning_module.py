"""LightningModule for UV trajectory completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import functional as F

from src.trajectory_completion.models.uv_completion_model import UVTrajectoryCompletionModel

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _masked_huber(pred: Tensor, target: Tensor, mask: Tensor, *, delta: float) -> Tensor:
    mask_f = mask.to(pred.dtype).unsqueeze(-1)
    denom = mask_f.sum().clamp_min(1.0)
    loss = F.huber_loss(pred, target, reduction="none", delta=float(delta))
    return (loss * mask_f).sum() / denom


def _smoothness_loss(pred: Tensor, mask: Tensor) -> Tensor:
    if pred.shape[1] < 3:
        return pred.new_tensor(0.0)
    m = mask.to(pred.dtype)
    a = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
    m2 = m[:, 2:] * m[:, 1:-1] * m[:, :-2]
    denom = m2.sum().clamp_min(1.0)
    return (a.abs().sum(dim=-1) * m2).sum() / denom


class TrajectoryCompletionLightningModule(pl.LightningModule):
    """Train a UV trajectory completion model."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.model = UVTrajectoryCompletionModel.from_config(config)

        train_cfg = config.get("training", {}) or {}
        loss_cfg = train_cfg.get("loss", {}) or {}
        metrics_cfg = config.get("metrics", {}) or {}
        self.learning_rate = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 200))
        self.max_epochs = int(train_cfg.get("max_epochs", 50))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))
        self.scheduler = str(train_cfg.get("scheduler", "cosine"))

        self.masked_weight = float(loss_cfg.get("masked_weight", 1.0))
        self.observed_weight = float(loss_cfg.get("observed_weight", 0.1))
        self.smoothness_weight = float(loss_cfg.get("smoothness_weight", 0.0))
        self.huber_delta = float(loss_cfg.get("huber_delta", 0.02))

        self.masked_accuracy_threshold = float(metrics_cfg.get("masked_accuracy_threshold_px", 2.0))
        self.observed_accuracy_threshold = float(metrics_cfg.get("observed_accuracy_threshold_px", 2.0))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.model(
            ball_uv_in=batch["ball_uv_in"],
            ball_vis=batch["ball_vis"],
            court_kp=batch["court_kp"],
            court_vis=batch.get("court_vis"),
            seq_len=batch.get("seq_len"),
            ball_mask=batch.get("ball_mask"),
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        pred = self.forward(batch)
        loss, logs = self._compute_losses(pred, batch)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        pred = self.forward(batch)
        loss, logs = self._compute_losses(pred, batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, on_epoch=True)

    def _compute_losses(self, pred: Tensor, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        ball_uv_gt = batch["ball_uv_gt"]
        ball_mask = batch["ball_mask"]
        ball_vis = batch["ball_vis"]
        seq_len = batch.get("seq_len")

        B, T, _ = pred.shape
        if ball_uv_gt.shape[1] != T:
            ball_uv_gt = ball_uv_gt[:, :T]
            ball_mask = ball_mask[:, :T]
            ball_vis = ball_vis[:, :T]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=T)

        if seq_len is None:
            valid = ball_mask > 0
        else:
            t = torch.arange(T, device=pred.device)[None, :]
            valid = (t < seq_len.to(torch.long).view(B, 1)) & (ball_mask > 0)

        masked = valid & (ball_vis <= 0)
        observed = valid & (ball_vis > 0)

        loss_masked = _masked_huber(pred, ball_uv_gt, masked, delta=self.huber_delta)
        loss_observed = _masked_huber(pred, ball_uv_gt, observed, delta=self.huber_delta)

        loss = self.masked_weight * loss_masked + self.observed_weight * loss_observed

        loss_smooth = pred.new_tensor(0.0)
        if self.smoothness_weight > 0:
            loss_smooth = _smoothness_loss(pred, valid)
            loss = loss + self.smoothness_weight * loss_smooth

        acc_masked = pred.new_tensor(0.0)
        acc_observed = pred.new_tensor(0.0)
        if self.masked_accuracy_threshold > 0 or self.observed_accuracy_threshold > 0:
            error = torch.linalg.vector_norm(pred - ball_uv_gt, dim=-1)
            if self.masked_accuracy_threshold > 0:
                masked_denom = masked.to(torch.float32).sum().clamp_min(1.0)
                acc_masked = (error <= self.masked_accuracy_threshold).to(torch.float32)
                acc_masked = (acc_masked * masked.to(torch.float32)).sum() / masked_denom
            if self.observed_accuracy_threshold > 0:
                observed_denom = observed.to(torch.float32).sum().clamp_min(1.0)
                acc_observed = (error <= self.observed_accuracy_threshold).to(torch.float32)
                acc_observed = (acc_observed * observed.to(torch.float32)).sum() / observed_denom

        denom = valid.to(torch.float32).sum().clamp_min(1.0)
        masked_ratio = masked.to(torch.float32).sum() / denom
        logs = {
            "loss_masked": loss_masked.detach(),
            "loss_observed": loss_observed.detach(),
            "loss_smooth": loss_smooth.detach(),
            "masked_ratio": masked_ratio.detach(),
            "accuracy_masked": acc_masked.detach(),
            "accuracy_observed": acc_observed.detach(),
        }
        return loss, logs

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "data": {"max_seq_len": 32},
            "model": {"name": "uv_transformer", "hidden_dim": 64, "num_layers": 2, "num_heads": 4, "max_seq_len": 32},
            "training": {"learning_rate": 1e-3, "weight_decay": 0.0, "loss": {"masked_weight": 1.0, "observed_weight": 0.1}},
        }
    )
    module = TrajectoryCompletionLightningModule(cfg)
    batch = {
        "ball_uv_in": torch.rand(2, 32, 2),
        "ball_vis": torch.randint(0, 2, (2, 32)).float(),
        "ball_uv_gt": torch.rand(2, 32, 2),
        "ball_mask": torch.ones(2, 32),
        "court_kp": torch.rand(2, 20, 2),
        "court_vis": torch.ones(2, 20),
        "seq_len": torch.tensor([32, 20]),
    }
    out = module.forward(batch)
    assert out.shape == (2, 32, 2)
    loss, _ = module._compute_losses(out, batch)
    assert torch.isfinite(loss)
    print("trajectory_completion.lightning smoke ok")
