"""PyTorch Lightning module for ball detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from src.tasks.ball_detection.models import (
    build_ball_detection_discriminator,
    build_ball_detection_model,
    to_model_input,
)
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.base.training.gan_training import ManualGANSupportMixin
from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.losses import FocalBCEWithLogitsLoss
from src.tasks.base.training.qualitative_saving import save_qualitative_clip
from src.utils.data.heatmaps import (
    heatmaps_to_soft_argmax,
    resize_heatmap_sequence,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _build_metrics(metrics_cfg: Any) -> BallDetectionMetrics:
    """Construct a :class:`BallDetectionMetrics` from a metrics config dict."""
    return BallDetectionMetrics(
        peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
        ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
        nms_kernel=int(metrics_cfg.get("nms_kernel", 9)),
        max_predictions_per_frame=int(metrics_cfg.get("max_predictions_per_frame", 8)),
    )


class BallDetectionLightningModule(ManualGANSupportMixin, BaseLightningModule):
    """Lightning module for training ball detection.

    Supervised training optimizes heatmap losses. When ``training.gan.enabled``
    is set, ball coordinate sequences extracted from predicted heatmaps via
    differentiable soft-argmax are scored by a trajectory discriminator against
    ground-truth coordinate sequences, and the binary real/fake signal is fed
    back to the heatmap model as an adversarial loss.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        loss_cfg = self.config.get("loss", {})
        metrics_cfg = self.config.get("metrics", {})

        self.model = build_ball_detection_model(self.config)

        loss_gamma = float(loss_cfg.get("gamma", 2.0))
        self.loss_fn = FocalBCEWithLogitsLoss(gamma=loss_gamma, validate_shape=True)

        train_cfg = self.config.get("training", {})
        gan_cfg = train_cfg.get("gan", {}) or {}
        gan_enabled = bool(gan_cfg.get("enabled", False))
        self.gan_soft_argmax_temperature = float(
            gan_cfg.get("soft_argmax_temperature", 1.0)
        )
        self._initialize_manual_gan(
            discriminator=(
                build_ball_detection_discriminator(self.config) if gan_enabled else None
            ),
        )

        self.train_metrics, self.val_metrics, self.test_metrics = (
            _build_metrics(metrics_cfg) for _ in range(3)
        )

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            images: Input tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Logits of shape ``(B, 1, T, Hh, Wh)``.
        """
        return self.model(images)

    def _predict_heatmap_logits(self, images: Tensor, target_size_hw: tuple[int, int]) -> Tensor:
        """Predict per-frame heatmap logits resized to the target heatmap size."""
        model_cfg = self.config.get("model", {})
        model_input = to_model_input(images, model_cfg)

        logits = self.model(model_input)

        # Squeeze channel dim: (B, 1, T, Hh, Wh) -> (B, T, Hh, Wh)
        logits = logits.squeeze(1)

        # Interpolate if model output size != target heatmap size
        return resize_heatmap_sequence(logits, target_size_hw)

    def _extract_gt_trajectory(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Extract the primary ball trajectory and its visibility mask.

        Returns:
            Tuple of:
                - ball_xy: Normalized ``(B, T, 2)`` coordinates in ``(x, y)``.
                - mask: Boolean ``(B, T)`` mask of frames with a visible ball.
        """
        coords = batch["coords"]  # (B, T, K, 2) in original image pixels
        visibility = batch["visibility"]  # (B, T, K)
        original_size = batch["original_size"]  # (B, 2) as (width, height)

        # Pick the first visible instance per frame (argmax of a 0/1 mask
        # returns the first maximal entry).
        first_visible = visibility.argmax(dim=-1)  # (B, T)
        gather_index = first_visible[..., None, None].expand(-1, -1, 1, 2)
        ball_xy = coords.gather(2, gather_index).squeeze(2)  # (B, T, 2)

        scale = (original_size - 1.0).clamp(min=1.0)  # (B, 2)
        ball_xy = ball_xy / scale[:, None, :]
        mask = visibility.amax(dim=-1) > 0.5
        return ball_xy, mask

    def _compute_supervised_result(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Any]:
        """Compute forward pass, supervised loss, metrics, and GAN sequences."""
        target_heatmaps = batch["heatmaps"]
        logits = self._predict_heatmap_logits(
            batch["images"],
            target_heatmaps.shape[-2:],
        )

        loss = self.loss_fn(logits, target_heatmaps)
        pred_heatmaps = torch.sigmoid(logits)

        self._metric_tracker_for_stage(stage).update(
            pred_heatmaps,
            batch["coords"],
            batch["visibility"],
            batch["original_size"],
        )

        gan_real, gan_mask = self._extract_gt_trajectory(batch)
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
            "gan_mask": gan_mask,
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

    def _log_stage_metrics(self, stage: str, loss: Tensor, metrics: dict[str, Any]) -> None:
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

    # Style parameters are stored as plain kwargs so the module never imports
    # the visualization package at load time (which would create an import
    # cycle: visualization -> api -> inference -> training.lightning_module).
    # ``DrawStyle`` / ``LayoutStyle`` are constructed lazily in
    # ``render_qualitative_samples`` after the deferred import.
    _PEAK_THRESHOLD_DEFAULT: float = 0.5
    _QUALITATIVE_FPS: float = 5.0

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

        # Styles are built here (not as class attributes) so the module never
        # imports the visualization package at load time, which would create an
        # import cycle (visualization -> api -> inference -> training).
        draw_style = DrawStyle(
            gt_radius=6,
            pred_radius=6,
            thickness=2,
            gt_color_rgb=(0, 255, 0),
            pred_color_rgb=(255, 80, 80),
            text_color_rgb=(255, 255, 255),
            muted_text_color_rgb=(160, 160, 160),
        )
        layout_style = LayoutStyle(
            header_height=48,
            tile_gap=4,
            text_scale=0.55,
            text_thickness=1,
            background_rgb=(30, 30, 30),
            panel_label_height=22,
        )

        device = next(self.parameters()).device

        normalize_cfg = dict(
            self.config.get("data", {})
            .get("augmentation", {})
            .get("normalize_imagenet", {})
            or {}
        )
        model_cfg = dict(self.config.get("model", {}) or {})
        metrics_cfg = self.config.get("metrics", {})
        peak_threshold = float(
            metrics_cfg.get("peak_threshold", self._PEAK_THRESHOLD_DEFAULT)
        )

        for batch_idx, batch in enumerate(batches):
            images = batch["images"].to(device)  # (B, T, C, H, W)
            heatmaps_gt = batch["heatmaps"]       # (B, T, Hh, Wh) – for sizing only

            with torch.no_grad():
                logits = self._predict_heatmap_logits(images, heatmaps_gt.shape[-2:])
                pred_heatmaps = torch.sigmoid(logits).cpu()  # (B, T, Hh, Wh) in [0,1]

            render_kwargs = build_render_animation_inputs(
                images_btchw=images.cpu(),
                pred_heatmaps_bthw=pred_heatmaps,
                peak_threshold=peak_threshold,
                normalize_cfg=normalize_cfg,
                model_cfg=model_cfg,
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
                fps=self._QUALITATIVE_FPS,
            )
