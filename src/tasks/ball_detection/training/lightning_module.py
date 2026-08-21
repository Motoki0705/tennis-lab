"""PyTorch Lightning module for ball detection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import BallTrainingCall
from src.tasks.ball_detection.model_io.factory import build_ball_detection_pair
from src.tasks.ball_detection.models import build_ball_detection_discriminator
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.losses import (
    FocalBCEWithLogitsLoss,
    validate_focal_bce_inputs,
)
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.utils.data.heatmaps import heatmaps_to_soft_argmax

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _build_metrics(metrics_cfg: Mapping[str, Any]) -> BallDetectionMetrics:
    """Construct a :class:`BallDetectionMetrics` from a metrics config dict."""
    return BallDetectionMetrics(
        peak_threshold=float(metrics_cfg["peak_threshold"]),
        ball_distance_threshold=float(metrics_cfg["ball_distance_threshold"]),
        nms_kernel=int(metrics_cfg["nms_kernel"]),
        max_predictions_per_frame=int(metrics_cfg["max_predictions_per_frame"]),
        subpixel_refine=bool(metrics_cfg["subpixel_refine"]),
    )


def _rgb_triplet(values: Any, *, name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    return int(values[0]), int(values[1]), int(values[2])


class BallStepResult(TypedDict):
    """Typed supervised outputs consumed by shared GAN/staged lifecycles."""

    loss: Tensor
    metrics: dict[str, Any]
    pred_heatmaps: Tensor
    gan_fake: Tensor
    gan_real: Tensor
    gan_padding_mask: Tensor


class BallDetectionLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for training ball detection.

    Supervised training optimizes heatmap losses. When ``training.gan.enabled``
    is set, ball coordinate sequences extracted from predicted heatmaps via
    differentiable soft-argmax are scored by a trajectory discriminator against
    ground-truth coordinate sequences, and the binary real/fake signal is fed
    back to the heatmap model as an adversarial loss.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        loss_cfg = self.config.loss
        metrics_cfg = self.config.metrics

        model_pair = build_ball_detection_pair(self.config)
        self.model = model_pair.model
        self.model_io = cast(BallModelIOAdapter, model_pair.adapter)

        loss_gamma = float(loss_cfg.gamma)
        self.loss_fn = FocalBCEWithLogitsLoss(gamma=loss_gamma)

        gan_cfg = self.config.training.gan
        gan_enabled = bool(gan_cfg.enabled)
        self.gan_soft_argmax_temperature = float(gan_cfg.soft_argmax_temperature)
        self._initialize_manual_gan(
            discriminator=(
                build_ball_detection_discriminator(self.config) if gan_enabled else None
            ),
        )

        self.train_metrics, self.val_metrics, self.test_metrics = (
            _build_metrics(metrics_cfg) for _ in range(3)
        )

    def forward(self, *model_args: Tensor) -> Tensor:
        """Compute over a model-I/O boundary-prepared argument tuple."""
        return cast(Tensor, self.model(*model_args))

    def _predict_heatmap_logits(
        self, images: Tensor, target_size_hw: tuple[int, int]
    ) -> Tensor:
        """Predict per-frame heatmap logits resized to the target heatmap size."""
        call = self.model_io.prepare_model_call(images)
        logits = self.model(*call.model_args)
        return self.model_io.resized_logits(
            logits,
            call,
            target_size_hw=target_size_hw,
        )

    def _extract_gt_trajectory(self, call: BallTrainingCall) -> tuple[Tensor, Tensor]:
        """Extract the primary ball trajectory and its padding mask.

        Returns:
            Tuple of:
                - ball_xy: Normalized ``(B, T, 2)`` coordinates in ``(x, y)``.
                - padding_mask: Boolean ``(B, T)`` mask where ``True`` marks a
                  frame without a visible ball.
        """
        coords = call.coords
        visibility = call.visibility
        original_size = call.original_size

        # Pick the first visible instance per frame (argmax of a 0/1 mask
        # returns the first maximal entry).
        first_visible = visibility.argmax(dim=-1)  # (B, T)
        gather_index = first_visible[..., None, None].expand(-1, -1, 1, 2)
        ball_xy = coords.gather(2, gather_index).squeeze(2)  # (B, T, 2)

        scale = (original_size - 1.0).clamp(min=1.0)  # (B, 2)
        ball_xy = ball_xy / scale[:, None, :]
        padding_mask = ~(visibility.amax(dim=-1) > 0.5)
        return ball_xy, padding_mask

    def _compute_supervised_result(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> BallStepResult:
        """Compute forward pass, supervised loss, metrics, and GAN sequences."""
        call = self.model_io.prepare_training_batch(batch)
        raw_logits = self.model(*call.model_call.model_args)
        logits = self.model_io.training_logits(raw_logits, call)
        target_heatmaps = call.target_heatmaps

        validate_focal_bce_inputs(logits, target_heatmaps)
        loss = self.loss_fn(logits, target_heatmaps)
        pred_heatmaps = torch.sigmoid(logits)

        self._metric_tracker_for_stage(stage).update(
            pred_heatmaps,
            call.coords,
            call.visibility,
            call.original_size,
        )

        gan_real, gan_padding_mask = self._extract_gt_trajectory(call)
        gan_fake = heatmaps_to_soft_argmax(
            logits,
            temperature=self.gan_soft_argmax_temperature,
        )

        return {
            "loss": loss,
            "metrics": {},
            "pred_heatmaps": pred_heatmaps,
            "gan_fake": gan_fake,
            "gan_real": gan_real,
            "gan_padding_mask": gan_padding_mask,
        }

    def _metric_tracker_for_stage(self, stage: str) -> BallDetectionMetrics:
        """Return the metrics object for the current stage."""
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _flush_stage_metrics(self, stage: str) -> None:
        tracker = self._metric_tracker_for_stage(stage)
        metrics = tracker.compute()
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                value,
                prog_bar=(stage == "val" and name == "f1"),
            )
        tracker.reset()

    def _log_stage_metrics(
        self, stage: str, loss: Tensor, metrics: dict[str, Any]
    ) -> None:
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self._log_gan_metrics(stage, metrics)

    def test_prediction_payload(
        self, batch: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist TrackNet test-split heatmap predictions and targets."""
        return {
            "window_id": batch["window_id"],
            "pred_heatmaps": result["pred_heatmaps"],
            "target_coords": batch["coords"],
            "target_visibility": batch["visibility"],
            "original_size": batch["original_size"],
            "heatmap_size": batch["heatmap_size"],
        }

    def configure_optimizers(self) -> Any:
        """Configure generator/discriminator optimizers through the shared GAN helper."""
        return self.configure_gan_optimizers(self.model.parameters())

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
        """Render ball detection visualisations using the shared clip renderer.

        For each collected batch one sample (index 0) is rendered as a 2×2 grid
        animation (RGB / MDD / RGB+pred / heatmap) and persisted via
        ``save_qualitative_clip`` (→ GIF when T>1, PNG when T=1).
        """
        # Deferred imports break the import cycle through the visualization
        # package (visualization -> api.predict -> inference -> training).
        from src.tasks.ball_detection.visualization.adapters.render_inputs import (  # noqa: PLC0415
            build_render_animation_inputs,
        )
        from src.tasks.ball_detection.visualization.rendering.clip_renderer import (  # noqa: PLC0415
            DrawStyle,
            LayoutStyle,
            render_animation_frames,
        )

        qualitative_cfg = self.config.training.qualitative_rendering
        draw_cfg = qualitative_cfg.draw
        layout_cfg = qualitative_cfg.layout
        draw_style = DrawStyle(
            gt_radius=int(draw_cfg.gt_radius),
            pred_radius=int(draw_cfg.pred_radius),
            thickness=int(draw_cfg.thickness),
            gt_color_rgb=_rgb_triplet(
                draw_cfg.gt_color_rgb, name="qualitative_rendering.draw.gt_color_rgb"
            ),
            pred_color_rgb=_rgb_triplet(
                draw_cfg.pred_color_rgb,
                name="qualitative_rendering.draw.pred_color_rgb",
            ),
            text_color_rgb=_rgb_triplet(
                draw_cfg.text_color_rgb,
                name="qualitative_rendering.draw.text_color_rgb",
            ),
            muted_text_color_rgb=_rgb_triplet(
                draw_cfg.muted_text_color_rgb,
                name="qualitative_rendering.draw.muted_text_color_rgb",
            ),
        )
        layout_style = LayoutStyle(
            header_height=int(layout_cfg.header_height),
            tile_gap=int(layout_cfg.tile_gap),
            text_scale=float(layout_cfg.text_scale),
            text_thickness=int(layout_cfg.text_thickness),
            background_rgb=_rgb_triplet(
                layout_cfg.background_rgb,
                name="qualitative_rendering.layout.background_rgb",
            ),
            panel_label_height=int(layout_cfg.panel_label_height),
        )

        device = next(self.parameters()).device

        normalize_cfg = dict(self.config.data.augmentation.normalize_imagenet)
        peak_threshold = float(self.config.metrics.peak_threshold)

        for batch_idx, batch in enumerate(batches):
            images = batch["images"].to(device)  # (B, T, C, H, W)
            heatmaps_gt = batch["heatmaps"]  # (B, T, Hh, Wh) – for sizing only

            with torch.no_grad():
                logits = self._predict_heatmap_logits(images, heatmaps_gt.shape[-2:])
                pred_heatmaps = torch.sigmoid(logits).cpu()  # (B, T, Hh, Wh) in [0,1]

            render_kwargs = build_render_animation_inputs(
                images_btchw=images.cpu(),
                pred_heatmaps_bthw=pred_heatmaps,
                peak_threshold=peak_threshold,
                normalize_cfg=normalize_cfg,
                model_io=self.model_io,
                sample_idx=0,
                clip_label=f"val epoch={epoch} batch={batch_idx}",
            )

            frames = render_animation_frames(
                **render_kwargs,
                draw=draw_style,
                layout=layout_style,
            )

            save_qualitative_clip(
                frames_rgb=frames,
                artifact_dir=artifact_dir,
                name=f"ball_batch{batch_idx:02d}",
                tb_writer=tb_writer,
                tag=f"qualitative/ball_detection/batch{batch_idx:02d}",
                global_step=global_step,
                fps=float(qualitative_cfg.fps),
            )
