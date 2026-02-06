"""LightningModule for UV trajectory completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from src.base.training.lightning_module import BaseLightningModule
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


def _boundary_jump(pred: Tensor, valid: Tensor, observed: Tensor) -> Tensor:
    if pred.shape[1] < 2:
        return pred.new_tensor(0.0)
    switch = valid[:, 1:] & valid[:, :-1] & (observed[:, 1:] != observed[:, :-1])
    jump = torch.linalg.vector_norm(pred[:, 1:] - pred[:, :-1], dim=-1)
    denom = switch.to(pred.dtype).sum().clamp_min(1.0)
    return (jump * switch.to(pred.dtype)).sum() / denom


class TrajectoryCompletionLightningModule(BaseLightningModule):
    """Train a UV trajectory completion model."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.model = UVTrajectoryCompletionModel.from_config(config)

        train_cfg = config.get("training", {}) or {}
        loss_cfg = train_cfg.get("loss", {}) or {}
        metrics_cfg = config.get("metrics", {}) or {}

        self.masked_weight = float(loss_cfg.get("masked_weight", 1.0))
        self.observed_weight = float(loss_cfg.get("observed_weight", 0.1))
        self.smoothness_weight = float(loss_cfg.get("smoothness_weight", 0.0))
        self.huber_delta = float(loss_cfg.get("huber_delta", 0.02))

        masked_schedule_cfg = loss_cfg.get("masked_schedule", {}) or {}
        self.masked_schedule_enabled = bool(masked_schedule_cfg.get("enabled", False))
        self.masked_schedule_start_epoch = int(masked_schedule_cfg.get("start_epoch", 0))
        self.masked_schedule_end_epoch = int(masked_schedule_cfg.get("end_epoch", 0))
        self.masked_weight_min = float(masked_schedule_cfg.get("weight_min", self.masked_weight))
        self.masked_weight_max = float(masked_schedule_cfg.get("weight_max", self.masked_weight))

        aux_cfg = loss_cfg.get("auxiliary_observed", {}) or {}
        self.auxiliary_observed_enabled = bool(aux_cfg.get("enabled", False))
        self.auxiliary_observed_weight = float(aux_cfg.get("weight", 0.0))
        self.auxiliary_depth_weighting = str(aux_cfg.get("depth_weighting", "linear"))
        self.auxiliary_exp_gamma = float(aux_cfg.get("exp_gamma", 1.5))
        self.auxiliary_predictor_hidden_dim = int(aux_cfg.get("predictor_hidden_dim", self.model.hidden_dim))
        self.auxiliary_layer_weights = self._build_auxiliary_layer_weights(len(self.model.blocks))

        self.auxiliary_observed_heads: nn.ModuleList | None = None
        if self.auxiliary_observed_enabled and len(self.model.blocks) > 0:
            self.auxiliary_observed_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.model.hidden_dim, self.auxiliary_predictor_hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.auxiliary_predictor_hidden_dim, 2),
                    )
                    for _ in range(len(self.model.blocks))
                ]
            )

        self.masked_accuracy_threshold = float(metrics_cfg.get("masked_accuracy_threshold_px", 2.0))
        self.observed_accuracy_threshold = float(metrics_cfg.get("observed_accuracy_threshold_px", 2.0))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.model(
            ball_uv_in=batch["ball_uv_in"],
            ball_obs_mask=batch["ball_obs_mask"],
            court_kp=batch["court_kp"],
            court_vis=batch.get("court_vis"),
            seq_len=batch.get("seq_len"),
            ball_mask=batch.get("ball_mask"),
        )

    def _build_auxiliary_layer_weights(self, num_layers: int) -> Tensor:
        if num_layers <= 0:
            return torch.empty(0)
        if self.auxiliary_depth_weighting == "exp":
            raw = torch.tensor(
                [self.auxiliary_exp_gamma ** i for i in range(num_layers)],
                dtype=torch.float32,
            )
        elif self.auxiliary_depth_weighting == "linear":
            raw = torch.arange(1, num_layers + 1, dtype=torch.float32)
        else:
            raise ValueError(
                f"Invalid training.loss.auxiliary_observed.depth_weighting: "
                f"{self.auxiliary_depth_weighting}. Use 'linear' or 'exp'."
            )
        return raw / raw.sum().clamp_min(1e-8)

    def _current_masked_weight(self) -> float:
        if not self.masked_schedule_enabled:
            return self.masked_weight
        if self.masked_schedule_end_epoch <= self.masked_schedule_start_epoch:
            return self.masked_weight_max
        epoch = int(self.current_epoch)
        if epoch <= self.masked_schedule_start_epoch:
            return self.masked_weight_min
        if epoch >= self.masked_schedule_end_epoch:
            return self.masked_weight_max
        ratio = (epoch - self.masked_schedule_start_epoch) / float(
            self.masked_schedule_end_epoch - self.masked_schedule_start_epoch
        )
        return self.masked_weight_min + ratio * (self.masked_weight_max - self.masked_weight_min)

    def _forward_with_auxiliary(self, batch: dict[str, Tensor]) -> tuple[Tensor, list[Tensor] | None]:
        if self.auxiliary_observed_enabled and self.auxiliary_observed_heads is not None:
            pred, intermediate = self.model(
                ball_uv_in=batch["ball_uv_in"],
                ball_obs_mask=batch["ball_obs_mask"],
                court_kp=batch["court_kp"],
                court_vis=batch.get("court_vis"),
                seq_len=batch.get("seq_len"),
                ball_mask=batch.get("ball_mask"),
                return_intermediate_ball_hidden=True,
            )
            return pred, intermediate
        return self.forward(batch), None

    def _compute_auxiliary_observed_loss(
        self,
        intermediate_ball_hidden: list[Tensor] | None,
        ball_uv_gt: Tensor,
        observed: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_aux_total = ball_uv_gt.new_tensor(0.0)
        logs: dict[str, Tensor] = {}
        if (
            not self.auxiliary_observed_enabled
            or self.auxiliary_observed_heads is None
            or not intermediate_ball_hidden
            or self.auxiliary_observed_weight <= 0
        ):
            return loss_aux_total, logs

        layer_weights = self.auxiliary_layer_weights.to(device=ball_uv_gt.device, dtype=ball_uv_gt.dtype)
        if len(intermediate_ball_hidden) != len(self.auxiliary_observed_heads):
            raise RuntimeError(
                "Mismatch between intermediate hidden states and auxiliary heads: "
                f"{len(intermediate_ball_hidden)} vs {len(self.auxiliary_observed_heads)}."
            )

        for idx, (hidden, head) in enumerate(zip(intermediate_ball_hidden, self.auxiliary_observed_heads, strict=True)):
            pred_aux = head(hidden)
            loss_aux_layer = _masked_huber(pred_aux, ball_uv_gt, observed, delta=self.huber_delta)
            weighted = layer_weights[idx] * loss_aux_layer
            loss_aux_total = loss_aux_total + weighted
            logs[f"loss_aux_layer{idx + 1}"] = loss_aux_layer.detach()
            logs[f"loss_aux_weight_layer{idx + 1}"] = layer_weights[idx].detach()
        return loss_aux_total, logs

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        pred, intermediate = self._forward_with_auxiliary(batch)
        loss, logs = self._compute_losses(pred, batch, intermediate_ball_hidden=intermediate)
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        pred, intermediate = self._forward_with_auxiliary(batch)
        loss, logs = self._compute_losses(pred, batch, intermediate_ball_hidden=intermediate)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, on_epoch=True)

    def _compute_losses(
        self,
        pred: Tensor,
        batch: dict[str, Tensor],
        *,
        intermediate_ball_hidden: list[Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        ball_uv_gt = batch["ball_uv_gt"]
        ball_mask = batch["ball_mask"]
        ball_vis = batch["ball_vis"]
        ball_obs_mask = batch["ball_obs_mask"]
        seq_len = batch.get("seq_len")

        B, T, _ = pred.shape
        if ball_uv_gt.shape[1] != T:
            ball_uv_gt = ball_uv_gt[:, :T]
            ball_mask = ball_mask[:, :T]
            ball_vis = ball_vis[:, :T]
            ball_obs_mask = ball_obs_mask[:, :T]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=T)

        if seq_len is None:
            valid = (ball_mask > 0) & (ball_vis > 0)
        else:
            t = torch.arange(T, device=pred.device)[None, :]
            valid = (t < seq_len.to(torch.long).view(B, 1)) & (ball_mask > 0) & (ball_vis > 0)

        masked = valid & (ball_obs_mask <= 0)
        observed = valid & (ball_obs_mask > 0)

        loss_masked = _masked_huber(pred, ball_uv_gt, masked, delta=self.huber_delta)
        loss_observed = _masked_huber(pred, ball_uv_gt, observed, delta=self.huber_delta)
        masked_weight_t = self._current_masked_weight()

        loss = masked_weight_t * loss_masked + self.observed_weight * loss_observed

        loss_smooth = pred.new_tensor(0.0)
        if self.smoothness_weight > 0:
            loss_smooth = _smoothness_loss(pred, valid)
            loss = loss + self.smoothness_weight * loss_smooth

        loss_aux, aux_logs = self._compute_auxiliary_observed_loss(intermediate_ball_hidden, ball_uv_gt, observed)
        if self.auxiliary_observed_enabled and self.auxiliary_observed_weight > 0:
            loss = loss + self.auxiliary_observed_weight * loss_aux

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
        boundary_jump_pred = _boundary_jump(pred, valid, observed)
        boundary_jump_gt = _boundary_jump(ball_uv_gt, valid, observed)
        boundary_jump_error = (boundary_jump_pred - boundary_jump_gt).abs()
        logs = {
            "loss_masked": loss_masked.detach(),
            "loss_observed": loss_observed.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_aux": loss_aux.detach(),
            "masked_weight_t": pred.new_tensor(float(masked_weight_t)).detach(),
            "masked_ratio": masked_ratio.detach(),
            "accuracy_masked": acc_masked.detach(),
            "accuracy_observed": acc_observed.detach(),
            "boundary_jump_pred": boundary_jump_pred.detach(),
            "boundary_jump_gt": boundary_jump_gt.detach(),
            "boundary_jump_error": boundary_jump_error.detach(),
        }
        logs.update(aux_logs)
        return loss, logs

    # configure_optimizers inherited from BaseLightningModule


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
        "ball_obs_mask": torch.randint(0, 2, (2, 32)).float(),
        "ball_uv_gt": torch.rand(2, 32, 2),
        "ball_vis": torch.randint(0, 2, (2, 32)).float(),
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
