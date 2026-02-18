"""LightningModule for UV trajectory completion."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from src.base.training.lightning_module import BaseLightningModule
from src.trajectory_completion.models import build_trajectory_completion_model

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


def _masked_bce_with_logits(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.to(logits.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none")
    return (loss * mask_f).sum() / denom


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
        model_cfg = config.get("model", {}) or {}
        model_name = str(model_cfg.get("name", "uv_transformer"))
        self.use_court_context = model_name != "uv_transformer_nocourt"
        self.model = build_trajectory_completion_model(self.config)
        self._maybe_initialize_nocourt_from_court_checkpoint(model_name=model_name)

        train_cfg = config.get("training", {}) or {}
        loss_cfg = train_cfg.get("loss", {}) or {}
        metrics_cfg = config.get("metrics", {}) or {}

        self.masked_weight = float(loss_cfg.get("masked_weight", 1.0))
        self.observed_weight = float(loss_cfg.get("observed_weight", 0.1))
        self.smoothness_weight = float(loss_cfg.get("smoothness_weight", 0.0))
        self.in_frame_weight = float(loss_cfg.get("in_frame_weight", 0.2))
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
        self.in_frame_threshold = float(metrics_cfg.get("in_frame_threshold", 0.5))

    def _maybe_initialize_nocourt_from_court_checkpoint(self, *, model_name: str) -> None:
        if model_name != "uv_transformer_nocourt":
            return
        model_cfg = self.config.get("model", {}) or {}
        init_ckpt = model_cfg.get("init_from_court_checkpoint")
        if init_ckpt is None or str(init_ckpt).strip() == "":
            return

        ckpt_path = Path(str(init_ckpt))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init_from_court_checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint format for init_from_court_checkpoint: {ckpt_path}")

        source_state: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if isinstance(key, str) and key.startswith("model.") and isinstance(value, Tensor):
                source_state[key[len("model.") :]] = value

        target_state = self.model.state_dict()
        mapped_state: dict[str, Tensor] = {}
        for target_key, target_value in target_state.items():
            direct = source_state.get(target_key)
            if isinstance(direct, Tensor) and tuple(direct.shape) == tuple(target_value.shape):
                mapped_state[target_key] = direct
                continue
            if target_key.startswith("blocks."):
                source_key = "ball_temporal_layers." + target_key[len("blocks.") :]
                temporal = source_state.get(source_key)
                if isinstance(temporal, Tensor) and tuple(temporal.shape) == tuple(target_value.shape):
                    mapped_state[target_key] = temporal

        if not mapped_state:
            raise RuntimeError(
                "No compatible parameters found when initializing uv_transformer_nocourt "
                f"from checkpoint: {ckpt_path}"
            )
        self.model.load_state_dict(mapped_state, strict=False)
        print(
            "Initialized uv_transformer_nocourt from court checkpoint "
            f"{ckpt_path} with {len(mapped_state)} tensor(s)."
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        if self.use_court_context:
            out = self.model(
                batch["ball_uv"],
                batch["court_kp"],
                batch.get("ball_vis"),
                batch.get("ball_mask"),
                batch.get("court_vis"),
                return_in_frame_logits=False,
            )
        else:
            out = self.model(
                batch["ball_uv"],
                batch.get("ball_vis"),
                batch.get("ball_mask"),
                return_in_frame_logits=False,
            )
        if isinstance(out, tuple):
            return out[0]
        return out

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

    def _forward_with_auxiliary(self, batch: dict[str, Tensor]) -> tuple[Tensor, list[Tensor] | None, Tensor]:
        if self.use_court_context:
            if self.auxiliary_observed_enabled and self.auxiliary_observed_heads is not None:
                pred, intermediate, in_frame_logits = self.model(
                    batch["ball_uv"],
                    batch["court_kp"],
                    batch.get("ball_vis"),
                    batch.get("ball_mask"),
                    batch.get("court_vis"),
                    return_intermediate_ball_hidden=True,
                    return_in_frame_logits=True,
                )
                return pred, intermediate, in_frame_logits
            pred, in_frame_logits = self.model(
                batch["ball_uv"],
                batch["court_kp"],
                batch.get("ball_vis"),
                batch.get("ball_mask"),
                batch.get("court_vis"),
                return_in_frame_logits=True,
            )
            return pred, None, in_frame_logits

        if self.auxiliary_observed_enabled and self.auxiliary_observed_heads is not None:
            pred, intermediate, in_frame_logits = self.model(
                batch["ball_uv"],
                batch.get("ball_vis"),
                batch.get("ball_mask"),
                return_intermediate_ball_hidden=True,
                return_in_frame_logits=True,
            )
            return pred, intermediate, in_frame_logits
        pred, in_frame_logits = self.model(
            batch["ball_uv"],
            batch.get("ball_vis"),
            batch.get("ball_mask"),
            return_in_frame_logits=True,
        )
        return pred, None, in_frame_logits

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
        pred, intermediate, in_frame_logits = self._forward_with_auxiliary(batch)
        loss, logs = self._compute_losses(
            pred,
            batch,
            intermediate_ball_hidden=intermediate,
            in_frame_logits=in_frame_logits,
        )
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:  # noqa: ARG002
        pred, intermediate, in_frame_logits = self._forward_with_auxiliary(batch)
        loss, logs = self._compute_losses(
            pred,
            batch,
            intermediate_ball_hidden=intermediate,
            in_frame_logits=in_frame_logits,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, on_epoch=True)

    def _compute_losses(
        self,
        pred: Tensor,
        batch: dict[str, Tensor],
        *,
        intermediate_ball_hidden: list[Tensor] | None = None,
        in_frame_logits: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        ball_uv_gt = batch["ball_uv_gt"]
        ball_mask = batch["ball_mask"]
        ball_gt_vis = batch["ball_gt_vis"]
        ball_vis = batch["ball_vis"]
        ball_in_frame_gt = batch.get("ball_in_frame_gt", ball_gt_vis)

        _, T, _ = pred.shape
        if ball_uv_gt.shape[1] != T:
            ball_uv_gt = ball_uv_gt[:, :T]
            ball_mask = ball_mask[:, :T]
            ball_gt_vis = ball_gt_vis[:, :T]
            ball_vis = ball_vis[:, :T]
            ball_in_frame_gt = ball_in_frame_gt[:, :T]
        valid = (ball_mask > 0) & (ball_gt_vis > 0)
        in_frame_valid = ball_mask > 0

        masked = valid & (ball_vis <= 0)
        observed = valid & (ball_vis > 0)

        loss_masked = _masked_huber(pred, ball_uv_gt, masked, delta=self.huber_delta)
        loss_observed = _masked_huber(pred, ball_uv_gt, observed, delta=self.huber_delta)
        masked_weight_t = self._current_masked_weight()

        loss = masked_weight_t * loss_masked + self.observed_weight * loss_observed

        loss_smooth = pred.new_tensor(0.0)
        if self.smoothness_weight > 0:
            loss_smooth = _smoothness_loss(pred, valid)
            loss = loss + self.smoothness_weight * loss_smooth

        loss_in_frame = _masked_bce_with_logits(in_frame_logits, ball_in_frame_gt, in_frame_valid)
        if self.in_frame_weight > 0:
            loss = loss + self.in_frame_weight * loss_in_frame

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
        in_frame_probs = torch.sigmoid(in_frame_logits)
        in_frame_pred = in_frame_probs >= float(self.in_frame_threshold)
        in_frame_target = ball_in_frame_gt > 0.5
        in_frame_valid_f = in_frame_valid.to(torch.float32)
        in_frame_denom = in_frame_valid_f.sum().clamp_min(1.0)
        in_frame_acc = (
            ((in_frame_pred == in_frame_target).to(torch.float32) * in_frame_valid_f).sum() / in_frame_denom
        )
        tp = ((in_frame_pred & in_frame_target) & in_frame_valid).to(torch.float32).sum()
        fp = ((in_frame_pred & (~in_frame_target)) & in_frame_valid).to(torch.float32).sum()
        fn = (((~in_frame_pred) & in_frame_target) & in_frame_valid).to(torch.float32).sum()
        in_frame_precision = tp / (tp + fp).clamp_min(1.0)
        in_frame_recall = tp / (tp + fn).clamp_min(1.0)
        logs = {
            "loss_masked": loss_masked.detach(),
            "loss_observed": loss_observed.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_in_frame": loss_in_frame.detach(),
            "loss_aux": loss_aux.detach(),
            "masked_weight_t": pred.new_tensor(float(masked_weight_t)).detach(),
            "masked_ratio": masked_ratio.detach(),
            "accuracy_masked": acc_masked.detach(),
            "accuracy_observed": acc_observed.detach(),
            "in_frame_acc": in_frame_acc.detach(),
            "in_frame_precision": in_frame_precision.detach(),
            "in_frame_recall": in_frame_recall.detach(),
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
            "model": {
                "name": "uv_transformer",
                "hidden_dim": 64,
                "num_ball_layers": 2,
                "num_query_layers": 2,
                "num_heads": 4,
                "max_seq_len": 32,
            },
            "training": {"learning_rate": 1e-3, "weight_decay": 0.0, "loss": {"masked_weight": 1.0, "observed_weight": 0.1}},
        }
    )
    module = TrajectoryCompletionLightningModule(cfg)
    batch = {
        "ball_uv": torch.rand(2, 32, 2),
        "ball_vis": torch.randint(0, 2, (2, 32)).float(),
        "ball_uv_gt": torch.rand(2, 32, 2),
        "ball_gt_vis": torch.randint(0, 2, (2, 32)).float(),
        "ball_in_frame_gt": torch.randint(0, 2, (2, 32)).float(),
        "ball_mask": torch.ones(2, 32),
        "court_kp": torch.rand(2, 20, 2),
        "court_vis": torch.ones(2, 20),
    }
    out = module.forward(batch)
    assert out.shape == (2, 32, 2)
    _, _, in_frame_logits = module._forward_with_auxiliary(batch)
    loss, _ = module._compute_losses(out, batch, in_frame_logits=in_frame_logits)
    assert torch.isfinite(loss)
    print("trajectory_completion.lightning smoke ok")
