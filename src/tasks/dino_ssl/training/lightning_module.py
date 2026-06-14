"""Lightning module implementing DINOv3 self-distillation with LoRA.

Loss orchestration uses DINOv3's own loss implementations (DINO class-token
loss, iBOT masked-patch loss, KoLeo regulariser) so the self-supervised learning
strategy matches the upstream recipe. Only the LoRA adapters and projection heads
are optimised; the teacher is an EMA of the student.
"""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch

from src.tasks.dino_ssl._dinov3 import load_dinov3_losses_and_head
from src.tasks.dino_ssl.models.backbone import count_trainable_parameters
from src.tasks.dino_ssl.models.ssl_network import DinoSSLNetwork


def _cosine(start: float, end: float, progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


class DinoSSLLightningModule(pl.LightningModule):
    """DINOv3 LoRA self-distillation training step."""

    def __init__(self, config: Any, steps_per_epoch: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.steps_per_epoch = steps_per_epoch

        _, dino_loss_cls, ibot_loss_cls, koleo_loss_cls = load_dinov3_losses_and_head()
        self.network = DinoSSLNetwork(config.model)

        loss_cfg = config.training.loss
        self.dino_loss = dino_loss_cls(
            out_dim=int(config.model.dino.out_dim),
            student_temp=float(loss_cfg.student_temp),
            center_momentum=float(loss_cfg.center_momentum),
        )
        self.dino_loss.init_weights()

        self.ibot_enabled = self.network.ibot_enabled
        if self.ibot_enabled:
            self.ibot_loss = ibot_loss_cls(
                patch_out_dim=int(config.model.ibot.out_dim),
                student_temp=float(loss_cfg.student_temp),
                center_momentum=float(loss_cfg.center_momentum),
            )
            self.ibot_loss.init_weights()
        self.koleo_loss = koleo_loss_cls()

        self.dino_weight = float(loss_cfg.dino_weight)
        self.ibot_weight = float(loss_cfg.ibot_weight)
        self.koleo_weight = float(loss_cfg.koleo_weight)

        sched_cfg = config.training.schedule
        self.teacher_temp = float(sched_cfg.teacher_temp)
        self.teacher_temp_warmup = float(sched_cfg.teacher_temp_warmup)
        self.teacher_temp_warmup_epochs = int(sched_cfg.teacher_temp_warmup_epochs)
        self.momentum_base = float(sched_cfg.momentum_base)
        self.momentum_final = float(sched_cfg.momentum_final)

        trainable, total = count_trainable_parameters(self.network.student)
        self.trainable_params = trainable
        print(
            f"[dino_ssl] trainable {trainable:,} / {total:,} student params "
            f"({100 * trainable / max(total, 1):.2f}%)."
        )

    # ---- schedules ----
    def _max_steps(self) -> int:
        try:
            estimated = int(self.trainer.estimated_stepping_batches)
            if estimated > 0:
                return estimated
        except Exception:  # noqa: BLE001 - trainer may be unavailable
            pass
        if self.steps_per_epoch:
            return max(
                self.steps_per_epoch * int(self.config.training.trainer.max_epochs), 1
            )
        return 1

    def _current_teacher_temp(self) -> float:
        warmup_epochs = max(self.teacher_temp_warmup_epochs, 1)
        progress = min(self.current_epoch / warmup_epochs, 1.0)
        return (
            self.teacher_temp_warmup
            + (self.teacher_temp - self.teacher_temp_warmup) * progress
        )

    def _current_momentum(self) -> float:
        progress = self.global_step / max(self._max_steps(), 1)
        return _cosine(self.momentum_base, self.momentum_final, 1.0 - progress)

    # ---- loss computation ----
    def _compute_losses(
        self, batch: dict[str, Any], *, update_center: bool
    ) -> dict[str, torch.Tensor]:
        global_crops = batch["global_crops"]
        local_crops = batch["local_crops"]
        masks = batch["masks"] if self.ibot_enabled else None

        teacher_temp = self._current_teacher_temp()
        teacher_out = self.network.forward_teacher(global_crops)
        student_out = self.network.forward_student(global_crops, local_crops, masks)

        teacher_cls = teacher_out["cls_logits"]
        teacher_probs = self.dino_loss.softmax_center_teacher(
            teacher_cls, teacher_temp, update_centers=update_center
        )
        dino_value = self.dino_loss(
            student_out["cls_logits"], teacher_probs, ignore_diagonal=True
        )
        if update_center:
            self.dino_loss.update_center(teacher_cls.reshape(-1, teacher_cls.shape[-1]))

        losses: dict[str, torch.Tensor] = {"dino": dino_value}

        if self.ibot_enabled:
            teacher_patch = teacher_out["patch_logits"]
            teacher_patch_probs = self.ibot_loss.softmax_center_teacher(
                teacher_patch, teacher_temp, update_centers=update_center
            )
            masks_flat = torch.cat(masks, dim=0)
            ibot_value = self.ibot_loss(
                student_out["patch_logits"], teacher_patch_probs, masks_flat
            )
            if update_center:
                self.ibot_loss.update_center(teacher_patch)
            losses["ibot"] = ibot_value

        koleo_terms = [
            self.koleo_loss(feat) for feat in student_out["global_cls_features"]
        ]
        losses["koleo"] = torch.stack(koleo_terms).mean()

        total = self.dino_weight * losses["dino"] + self.koleo_weight * losses["koleo"]
        if self.ibot_enabled:
            total = total + self.ibot_weight * losses["ibot"]
        losses["total"] = total
        return losses

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses = self._compute_losses(batch, update_center=True)
        batch_size = batch["global_crops"][0].shape[0]
        self.log("train/loss", losses["total"], prog_bar=True, batch_size=batch_size)
        self.log("train/dino", losses["dino"], batch_size=batch_size)
        self.log("train/koleo", losses["koleo"], batch_size=batch_size)
        if self.ibot_enabled:
            self.log("train/ibot", losses["ibot"], batch_size=batch_size)
        self.log("schedule/teacher_temp", self._current_teacher_temp())
        self.log("schedule/momentum", self._current_momentum())
        return losses["total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        losses = self._compute_losses(batch, update_center=False)
        batch_size = batch["global_crops"][0].shape[0]
        self.log("val/loss", losses["total"], prog_bar=True, batch_size=batch_size)
        self.log("val/dino", losses["dino"], batch_size=batch_size)
        if self.ibot_enabled:
            self.log("val/ibot", losses["ibot"], batch_size=batch_size)
        return losses["total"]

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.network.update_teacher(self._current_momentum())

    def configure_optimizers(self) -> Any:
        optim_cfg = self.config.training.optimizer
        params = [p for p in self.network.student.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(optim_cfg.lr),
            weight_decay=float(optim_cfg.weight_decay),
            betas=tuple(optim_cfg.get("betas", (0.9, 0.999))),
        )
        warmup_steps = int(optim_cfg.get("warmup_steps", 0))
        max_steps = self._max_steps()

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return _cosine(1.0, float(optim_cfg.get("min_lr_ratio", 0.01)), progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = ["DinoSSLLightningModule"]
