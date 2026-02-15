"""PyTorch Lightning module for ball multi-task learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from src.base.training.lightning_module import BaseLightningModule
from src.ball_multitask.models import build_ball_multitask_model
from src.blcs.training.losses import BLCSLoss
from src.event_detection.utils.peaks import extract_event_peaks

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


def _make_time_mask(seq_len: Tensor, T: int) -> Tensor:
    B = seq_len.shape[0]
    t = torch.arange(T, device=seq_len.device)[None, :]
    return t < seq_len.to(torch.long).view(B, 1)


class BallMultitaskLightningModule(BaseLightningModule):
    """Lightning module for unified UV/3D/event training."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.model = build_ball_multitask_model(self.config)

        train_cfg = config.get("training", {}) or {}
        loss_cfg = train_cfg.get("loss", {}) or {}
        uv_cfg = loss_cfg.get("uv", {}) or {}
        traj_cfg = loss_cfg.get("traj3d", {}) or {}
        evt_cfg = loss_cfg.get("event", {}) or {}
        metrics_cfg = config.get("metrics", {}) or {}

        self.uv_loss_weight = float(uv_cfg.get("weight", 1.0))
        self.uv_masked_weight = float(uv_cfg.get("masked_weight", 1.0))
        self.uv_observed_weight = float(uv_cfg.get("observed_weight", 0.1))
        self.uv_smoothness_weight = float(uv_cfg.get("smoothness_weight", 0.0))
        self.uv_huber_delta = float(uv_cfg.get("huber_delta", 0.02))

        self.traj_loss_weight = float(traj_cfg.get("weight", 1.0))
        self.traj_loss = BLCSLoss(
            position_weight=float(traj_cfg.get("position_weight", 1.0)),
            velocity_weight=float(traj_cfg.get("velocity_weight", 0.1)),
            smoothness_weight=float(traj_cfg.get("smoothness_weight", 0.05)),
        )

        self.event_weight_uv = float(evt_cfg.get("weight_uv", 0.2))
        self.event_weight_3d = float(evt_cfg.get("weight_3d", 0.2))
        pos_weight_cfg = evt_cfg.get("pos_weight", [1.0, 1.0])
        pos_weight = torch.tensor(pos_weight_cfg, dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight, persistent=False)

        self.masked_accuracy_threshold = float(
            metrics_cfg.get("masked_accuracy_threshold_px", 0.0)
        )
        self.observed_accuracy_threshold = float(
            metrics_cfg.get("observed_accuracy_threshold_px", 0.0)
        )
        self.peak_threshold = float(metrics_cfg.get("peak_threshold", 0.5))
        self.match_tolerance_frames = int(metrics_cfg.get("match_tolerance_frames", 3))

        curriculum_cfg = train_cfg.get("curriculum", {}) or {}
        self.uv_steps_per_3d = int(curriculum_cfg.get("uv_steps_per_3d", 4))
        self.start_3d_epoch = int(curriculum_cfg.get("start_3d_epoch", 0))

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        if self._is_3d_step(batch_idx):
            outputs = self.model.forward_3d_event(
                batch["ball_pos_world"],
                seq_len=batch.get("seq_len"),
                ball_mask=batch.get("ball_mask"),
            )
            logits = outputs["event_logits"]
            event_targets, seq_len = self._slice_event_targets(batch, logits.shape[1])
            loss_event = self._event_loss(logits, event_targets, seq_len)
            total = self.event_weight_3d * loss_event
            self.log("train/loss", total, prog_bar=True)
            self.log("train/event_loss_3d", loss_event, prog_bar=True)
            return total

        outputs = self.model.forward_uv(
            batch["ball_uv_in"],
            batch["court_kp"],
            ball_vis=batch.get("ball_vis"),
            ball_mask=batch.get("ball_mask"),
            court_vis=batch.get("court_vis"),
            seq_len=batch.get("seq_len"),
        )
        total, logs = self._compute_uv_losses(outputs, batch)
        self.log("train/loss", total, prog_bar=True)
        for key, value in logs.items():
            self.log(f"train/{key}", value, prog_bar=False)
        return total

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        outputs = self.model.forward_uv(
            batch["ball_uv_in"],
            batch["court_kp"],
            ball_vis=batch.get("ball_vis"),
            ball_mask=batch.get("ball_mask"),
            court_vis=batch.get("court_vis"),
            seq_len=batch.get("seq_len"),
        )
        total, logs = self._compute_uv_losses(outputs, batch)
        self.log("val/loss", total, prog_bar=True)
        for key, value in logs.items():
            self.log(f"val/{key}", value, prog_bar=False)

        logits_3d = self.model.forward_3d_event(
            batch["ball_pos_world"],
            seq_len=batch.get("seq_len"),
            ball_mask=batch.get("ball_mask"),
        )["event_logits"]
        event_targets_3d, seq_len_3d = self._slice_event_targets(batch, logits_3d.shape[1])
        loss_event_3d = self._event_loss(logits_3d, event_targets_3d, seq_len_3d)
        self.log("val/event_loss_3d", loss_event_3d, prog_bar=False)

        event_targets_uv, seq_len_uv = self._slice_event_targets(
            batch, outputs["event_logits"].shape[1]
        )
        acc_uv = self._peak_match_accuracy(outputs["event_logits"], event_targets_uv, seq_len_uv)
        acc_3d = self._peak_match_accuracy(logits_3d, event_targets_3d, seq_len_3d)
        self.log("val/event_accuracy_uv", acc_uv, prog_bar=False)
        self.log("val/event_accuracy_3d", acc_3d, prog_bar=False)

    def _compute_uv_losses(
        self, outputs: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        uv_pred = outputs["uv_completed"]
        pos_pred = outputs["position_3d"]
        evt_logits = outputs["event_logits"]

        T = uv_pred.shape[1]
        ball_uv_gt = batch["ball_uv_gt"][:, :T]
        ball_vis = batch["ball_vis"][:, :T]
        ball_mask = batch["ball_mask"][:, :T]
        position_3d = batch["position_3d"][:, :T]
        event_targets = batch["event_targets"][:, :T]
        seq_len = batch.get("seq_len")
        if seq_len is not None:
            seq_len = torch.clamp(seq_len, max=T)

        B = uv_pred.shape[0]
        if seq_len is None:
            valid = ball_mask > 0
        else:
            t = torch.arange(T, device=uv_pred.device)[None, :]
            valid = (t < seq_len.to(torch.long).view(B, 1)) & (ball_mask > 0)

        masked = valid & (ball_vis <= 0)
        observed = valid & (ball_vis > 0)

        loss_masked = _masked_huber(uv_pred, ball_uv_gt, masked, delta=self.uv_huber_delta)
        loss_observed = _masked_huber(uv_pred, ball_uv_gt, observed, delta=self.uv_huber_delta)
        loss_uv = self.uv_masked_weight * loss_masked + self.uv_observed_weight * loss_observed

        loss_smooth = uv_pred.new_tensor(0.0)
        if self.uv_smoothness_weight > 0:
            loss_smooth = _smoothness_loss(uv_pred, valid)
            loss_uv = loss_uv + self.uv_smoothness_weight * loss_smooth

        loss_traj = self.traj_loss(pred_position=pos_pred, target_position=position_3d, mask=valid)["total"]
        loss_event = self._event_loss(evt_logits, event_targets, seq_len)

        total = (
            self.uv_loss_weight * loss_uv
            + self.traj_loss_weight * loss_traj
            + self.event_weight_uv * loss_event
        )

        logs: dict[str, Tensor] = {
            "uv_loss": loss_uv.detach(),
            "uv_loss_masked": loss_masked.detach(),
            "uv_loss_observed": loss_observed.detach(),
            "uv_loss_smooth": loss_smooth.detach(),
            "traj_loss": loss_traj.detach(),
            "event_loss_uv": loss_event.detach(),
        }

        if self.masked_accuracy_threshold > 0 or self.observed_accuracy_threshold > 0:
            error = torch.linalg.vector_norm(uv_pred - ball_uv_gt, dim=-1)
            if self.masked_accuracy_threshold > 0:
                masked_denom = masked.to(torch.float32).sum().clamp_min(1.0)
                acc_masked = (error <= self.masked_accuracy_threshold).to(torch.float32)
                acc_masked = (acc_masked * masked.to(torch.float32)).sum() / masked_denom
                logs["uv_accuracy_masked"] = acc_masked.detach()
            if self.observed_accuracy_threshold > 0:
                observed_denom = observed.to(torch.float32).sum().clamp_min(1.0)
                acc_observed = (error <= self.observed_accuracy_threshold).to(torch.float32)
                acc_observed = (acc_observed * observed.to(torch.float32)).sum() / observed_denom
                logs["uv_accuracy_observed"] = acc_observed.detach()

        return total, logs

    def _event_loss(self, logits: Tensor, targets: Tensor, seq_len: Tensor | None) -> Tensor:
        B, T, E = logits.shape
        pos_weight = self.pos_weight
        if pos_weight.numel() != E:
            raise ValueError(f"event pos_weight length {pos_weight.numel()} != num_events {E}")

        loss_per = F.binary_cross_entropy_with_logits(
            logits,
            targets.to(logits.dtype),
            reduction="none",
            pos_weight=pos_weight.to(device=logits.device, dtype=logits.dtype),
        )
        if seq_len is None:
            return loss_per.mean()
        time_mask = _make_time_mask(seq_len, T).to(loss_per.dtype)
        denom = time_mask.sum().clamp_min(1.0) * E
        return (loss_per * time_mask.unsqueeze(-1)).sum() / denom

    def _slice_event_targets(
        self, batch: dict[str, Tensor], T: int
    ) -> tuple[Tensor, Tensor | None]:
        targets = batch["event_targets"][:, :T]
        seq_len = batch.get("seq_len")
        if seq_len is not None:
            seq_len = torch.clamp(seq_len, max=T)
        return targets, seq_len

    def _is_3d_step(self, batch_idx: int) -> bool:
        _ = batch_idx
        if self.current_epoch < self.start_3d_epoch:
            return False
        if self.uv_steps_per_3d <= 0:
            return False
        cycle = self.uv_steps_per_3d + 1
        return ((int(self.global_step) + 1) % cycle) == 0

    def _peak_match_accuracy(self, logits: Tensor, targets: Tensor, seq_len: Tensor | None) -> Tensor:
        probs = torch.sigmoid(logits)
        pred_peaks, _ = extract_event_peaks(
            probs,
            seq_len,
            threshold=self.peak_threshold,
            min_distance=max(self.match_tolerance_frames, 1),
            top_k=None,
        )
        gt_peaks, _ = extract_event_peaks(
            targets,
            seq_len,
            threshold=self.peak_threshold,
            min_distance=1,
            top_k=None,
        )

        B, _, E = probs.shape
        correct = 0
        total = B * E
        tolerance = max(self.match_tolerance_frames, 0)
        for b in range(B):
            for e in range(E):
                pred = pred_peaks[b][e]
                gt = gt_peaks[b][e]
                if not gt and not pred:
                    correct += 1
                    continue
                if not gt or not pred:
                    continue
                matched = False
                for g in gt:
                    if any(abs(g - p) <= tolerance for p in pred):
                        matched = True
                        break
                if matched:
                    correct += 1
        return logits.new_tensor(correct / max(total, 1))


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = {
        "model": {"hidden_dim": 32, "num_layers": 2, "num_heads": 4, "max_seq_len": 16, "num_events": 2},
        "training": {"loss": {"event": {"pos_weight": [1.0, 1.0]}}},
    }
    module = BallMultitaskLightningModule(cfg)  # type: ignore[arg-type]
    batch = {
        "ball_uv_in": torch.rand(2, 16, 2),
        "ball_uv_gt": torch.rand(2, 16, 2),
        "ball_vis": torch.ones(2, 16),
        "ball_mask": torch.ones(2, 16),
        "court_kp": torch.rand(2, 20, 2),
        "court_vis": torch.ones(2, 20),
        "position_3d": torch.rand(2, 16, 3),
        "ball_pos_world": torch.rand(2, 16, 3),
        "event_targets": torch.zeros(2, 16, 2),
        "seq_len": torch.tensor([16, 12]),
    }
    loss = module.training_step(batch, 0)
    assert torch.isfinite(loss)
    print("ball_multitask.lightning smoke ok")
