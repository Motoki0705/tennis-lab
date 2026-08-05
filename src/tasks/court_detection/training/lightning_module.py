"""PyTorch Lightning module for court detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.models import build_court_detection_model
from src.tasks.court_detection.training.losses import (
    BinaryDiceLoss,
    DiceLoss,
    FocalBCEWithLogitsLoss,
)
from src.tasks.court_detection.training.metrics import CourtDetectionMetrics
from src.utils.data.heatmaps import heatmaps_to_pixel_coords


class CourtDetectionLightningModule(BaseLightningModule):
    """Unified Lightning module for court detection tasks.

    Supports three tasks via ``config.data.task``:

    * ``seg`` — Court cell segmentation (CE + Dice, 7 classes).
    * ``kp``  — Court keypoint heatmap regression (Focal BCE, 14 channels).
    * ``line`` — Court white-line segmentation (BCE + Dice, 1 channel).

    Inherits optimizer/scheduler logic from
    :class:`~src.tasks.base.training.lightning_module.BaseLightningModule`.
    """

    def __init__(self, config: object) -> None:
        super().__init__(config)
        self.save_hyperparameters()
        runtime = CourtTrainingConfig.from_config(config)
        self.task = runtime.data.task
        self.qualitative_fps = runtime.qualitative_fps
        self.qualitative_style = runtime.render_style

        # Model
        self.model = build_court_detection_model(self.config)

        # Task-specific loss
        if self.task == "seg":
            self.ce_weight = runtime.loss.ce_weight
            self.dice_weight = runtime.loss.dice_weight
            self.ce_loss_fn = nn.CrossEntropyLoss()
            self.dice_loss_fn: DiceLoss | BinaryDiceLoss = DiceLoss(
                num_classes=runtime.data.output_channels
            )
        elif self.task == "kp":
            self.loss_fn = FocalBCEWithLogitsLoss(
                gamma=runtime.loss.focal_gamma,
            )
        elif self.task == "line":
            self.bce_weight = runtime.loss.bce_weight
            self.dice_weight = runtime.loss.dice_weight
            pos_weight = torch.tensor([runtime.loss.pos_weight])
            self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.dice_loss_fn = BinaryDiceLoss()

        # Metrics
        self.train_metrics = CourtDetectionMetrics(
            self.task, runtime.data.output_channels
        )
        self.val_metrics = CourtDetectionMetrics(
            self.task, runtime.data.output_channels
        )
        self.test_metrics = CourtDetectionMetrics(
            self.task, runtime.data.output_channels
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.model.forward(images)

    def _compute_loss(self, logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Compute task-specific loss."""
        if self.task == "seg":
            masks = batch["mask"]
            loss_ce = self.ce_loss_fn.forward(logits, masks)
            loss_dice = self.dice_loss_fn.forward(logits, masks)
            return self.ce_weight * loss_ce + self.dice_weight * loss_dice
        elif self.task == "kp":
            heatmaps = batch["heatmap"]
            return self.loss_fn.forward(logits, heatmaps)
        else:  # line
            masks = batch["mask"]
            loss_bce = self.bce_loss_fn.forward(logits, masks)
            loss_dice = self.dice_loss_fn.forward(logits, masks)
            return self.bce_weight * loss_bce + self.dice_weight * loss_dice

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        """Shared computation for train/val/test steps."""
        images = batch["image"]
        logits = self.model(images)
        loss = self._compute_loss(logits, batch)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss, "logits": logits}

    def _metric_tracker_for_stage(self, stage: str) -> CourtDetectionMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _flush_stage_metrics(self, stage: str) -> None:
        metrics = self._metric_tracker_for_stage(stage).compute()
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=(stage == "val" and name in ("miou", "mean_dist", "dice")),
            )
        self._metric_tracker_for_stage(stage).reset()

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self._shared_step(batch, "train")
        self.train_metrics.update(outputs["logits"], batch)
        return outputs["loss"]

    def on_train_epoch_end(self) -> None:
        self._flush_stage_metrics("train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        outputs = self._shared_step(batch, "val")
        self.val_metrics.update(outputs["logits"], batch)

    def on_validation_epoch_end(self) -> None:
        self._flush_stage_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._reset_test_prediction_buffer()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        outputs = self._shared_step(batch, "test")
        self.test_metrics.update(outputs["logits"], batch)
        self.collect_test_predictions(batch, outputs)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        saved = self.save_test_predictions(metrics=metrics)
        if saved is not None:
            print(f"[test] saved test-split predictions -> {saved}")
        self._flush_stage_metrics("test")

    def test_prediction_payload(
        self, batch: dict[str, Any], result: dict[str, Tensor]
    ) -> dict[str, Any]:
        """Persist court predictions from ``data/court/data_val.json``."""
        logits = result["logits"]
        payload: dict[str, Any] = {
            "image_id": batch["image_id"],
            "image_size": batch["image_size"],
        }
        if self.task == "kp":
            payload.update(
                {
                    "pred_keypoints": heatmaps_to_pixel_coords(logits),
                    "target_keypoints": batch["keypoints"],
                }
            )
            return payload

        if self.task == "seg":
            payload.update(
                {
                    "pred_mask_flat": logits.argmax(dim=1).reshape(logits.shape[0], -1),
                    "target_mask_flat": batch["mask"].reshape(
                        batch["mask"].shape[0], -1
                    ),
                    "padded_size": logits.new_tensor(
                        [logits.shape[-2], logits.shape[-1]],
                        dtype=torch.int64,
                    ).repeat(logits.shape[0], 1),
                }
            )
            return payload

        pred_line = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        target_line = batch["mask"].reshape(batch["mask"].shape[0], -1)
        payload.update(
            {
                "pred_line_prob_flat": pred_line,
                "target_line_mask_flat": target_line,
                "padded_size": logits.new_tensor(
                    [logits.shape[-2], logits.shape[-1]],
                    dtype=torch.int64,
                ).repeat(logits.shape[0], 1),
            }
        )
        return payload

    # ------------------------------------------------------------------
    # Qualitative validation logging
    # ------------------------------------------------------------------

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        """Render court detection panels via shared rendering/ layer (pred-only).

        Imports are deferred to avoid a circular import through the visualization
        package (visualization → api → inference → training → visualization).
        """
        # Deferred imports to break circular dependency:
        # training.lightning_module → visualization.__init__ → orchestrator
        # → api.predict → inference → training.lightning_module
        from src.tasks.court_detection.visualization.adapters.render_inputs import (  # noqa: PLC0415
            batch_to_court_frame,
            logits_to_kp_prediction,
            logits_to_line_prob,
            logits_to_seg_mask,
        )
        from src.tasks.court_detection.visualization.rendering import (  # noqa: PLC0415
            render_kp_frames,
            render_line_frames,
            render_seg_frames,
        )

        device = next(self.parameters()).device
        style = self.qualitative_style.build()

        for batch_idx, batch in enumerate(batches):
            images = batch["image"].to(device)

            with torch.no_grad():
                logits = self.model(images).cpu()  # (B, C, H, W)

            court_frame = batch_to_court_frame(batch, sample_idx=0)
            pred_logits_sample = logits[0]  # (C, H, W)
            clip_label = f"court_{self.task}"

            if self.task == "seg":
                mask = logits_to_seg_mask(pred_logits_sample)
                frames_rgb = render_seg_frames(
                    frames=[court_frame],
                    masks=[mask],
                    style=style,
                    clip_label=clip_label,
                )
            elif self.task == "kp":
                kp_pred = logits_to_kp_prediction(pred_logits_sample)
                frames_rgb = render_kp_frames(
                    frames=[court_frame],
                    predictions=[kp_pred],
                    style=style,
                    clip_label=clip_label,
                )
            else:  # line
                line_prob = logits_to_line_prob(pred_logits_sample)
                frames_rgb = render_line_frames(
                    frames=[court_frame],
                    probs=[line_prob],
                    style=style,
                    clip_label=clip_label,
                )

            save_qualitative_clip(
                frames_rgb=frames_rgb,
                artifact_dir=artifact_dir,
                name=f"court_batch{batch_idx:02d}",
                tb_writer=tb_writer,
                tag=f"qualitative/court_detection/batch{batch_idx:02d}",
                global_step=global_step,
                fps=self.qualitative_fps,
            )
