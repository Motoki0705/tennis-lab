"""Shared LightningModule utilities for TennisDETR variants."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.training.utils.tennis_debug_vis import (
    render_debug_images_naive,
    render_debug_images_with_cameras,
)
from src.training.utils.tennis_matching import match_queries_to_targets


class BaseTennisLightningModule(LightningModule):
    """Base module providing optimizer and scheduler helpers.

    Subclasses are expected to define the following attributes in __init__:

    - self._lr: float
    - self._weight_decay: float
    - self._max_steps: int
    - self._scheduler_cfg: dict[str, Any]
    """

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and LR scheduler shared across TennisDETR modules."""
        optimizer = AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _build_scheduler(
        self,
        optimizer: AdamW,
    ) -> CosineAnnealingLR | LambdaLR | None:
        """Return the configured LR scheduler or ``None`` if disabled."""
        if self._max_steps <= 0:
            return None
        scheduler_name = str(self._scheduler_cfg.get("name") or "").lower()
        if scheduler_name == "cosine_with_warmup":
            warmup_steps = int(self._scheduler_cfg.get("warmup_steps", 0))
            min_lr_ratio = float(self._scheduler_cfg.get("min_lr_ratio", 0.0))
            lr_lambda = self._build_warmup_cosine_lambda(warmup_steps, min_lr_ratio)
            return LambdaLR(optimizer, lr_lambda=lr_lambda)
        return CosineAnnealingLR(optimizer, T_max=self._max_steps)

    def _build_warmup_cosine_lambda(
        self,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> Callable[[int], float]:
        """Construct a lambda function implementing warmup + cosine decay."""
        warmup = max(0, int(warmup_steps))
        base_min_ratio = float(min_lr_ratio)
        max_steps = max(1, self._max_steps)

        def _lr_lambda(step: int) -> float:
            step_f = float(step)
            if warmup > 0 and step_f < warmup:
                return step_f / float(max(1, warmup))
            progress_steps = max(1, max_steps - warmup)
            progress = min(max((step_f - warmup) / progress_steps, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_min_ratio + (1.0 - base_min_ratio) * cos

        return _lr_lambda

    def training_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Run a forward pass, compute losses, and log train scalars."""
        outputs = self.forward(batch)
        loss_dict = self._compute_loss(outputs, batch)
        self._log_losses(loss_dict, stage="train")
        return loss_dict["total"]

    def validation_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Evaluate the model, log losses, and optionally render debug images."""
        outputs = self.forward(batch)
        loss_dict = self._compute_loss(outputs, batch)
        self._log_losses(loss_dict, stage="val")
        viz_max_batches = int(getattr(self, "_viz_max_batches", 0))
        if batch_idx < viz_max_batches:
            image_gt, image_pred = self._render_debug_images(batch, outputs)
            step = int(self.global_step)
            if image_gt is not None:
                self._log_tensorboard_image("val/pose2d_gt", image_gt, step)
            if image_pred is not None:
                self._log_tensorboard_image("val/pose2d_pred_reproj", image_pred, step)

    def _match_queries_to_targets(
        self,
        pose_pred: Tensor,
        pose_gt: Tensor,
        exist_gt: Tensor,
        exist_logit: Tensor,
    ) -> list[tuple[Tensor, Tensor]]:
        lambda_pose = float(getattr(self, "_lambda_pose_match", 1.0))
        lambda_exist = float(getattr(self, "_lambda_exist_match", 1.0))
        return match_queries_to_targets(
            pose_pred=pose_pred,
            pose_gt=pose_gt,
            exist_gt=exist_gt,
            exist_logit=exist_logit,
            lambda_pose_match=lambda_pose,
            lambda_exist_match=lambda_exist,
        )

    def _render_debug_images(
        self,
        batch: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
    ) -> tuple[Tensor | None, Tensor | None]:
        try:
            from src.visualize.tennis_render import render_pose2d_frame
        except Exception:
            return None, None

        pose_pred = outputs.get("pose_3d")
        exist_conf = outputs.get("exist_conf")
        exist_threshold = float(getattr(self, "_exist_threshold", 0.5))

        images = render_debug_images_with_cameras(
            batch=batch,
            pose_pred=pose_pred,
            exist_conf=exist_conf,
            exist_threshold=exist_threshold,
            render_pose2d_frame=render_pose2d_frame,
        )
        if images != (None, None):
            return images
        return render_debug_images_naive(
            batch=batch,
            pose_pred=pose_pred,
            render_pose2d_frame=render_pose2d_frame,
        )

    def _log_losses(self, loss_dict: Mapping[str, Tensor], stage: str) -> None:
        for key, value in loss_dict.items():
            tag = f"{stage}/{key}"
            self.log(tag, value, prog_bar=(key == "total"), sync_dist=False)

    def _log_tensorboard_image(self, tag: str, image: Tensor, step: int) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        for writer in self._iter_tensorboard_writers(logger):
            writer.add_image(tag, image, step)

    def _iter_tensorboard_writers(self, logger: Any) -> list[Any]:
        experiments: list[Any] = []
        experiment = getattr(logger, "experiment", None)
        if experiment is not None and hasattr(experiment, "add_image"):
            experiments.append(experiment)
        child_loggers = getattr(logger, "loggers", None)
        if child_loggers:
            for child in child_loggers:
                exp = getattr(child, "experiment", None)
                if exp is not None and hasattr(exp, "add_image"):
                    experiments.append(exp)
        return experiments
