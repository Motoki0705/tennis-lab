"""LightningModule for DanceTrack SceneModel training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import draw_bounding_boxes, make_grid

from src.datasets.collate_tracking import SceneBatch
from src.models.scene_model.build import build_scene_model

from .dino_denoiser import DinoDenoiser
from .head_adapter import HeadAdapter


@dataclass(slots=True)
class _MetricState:
    matched: float = 0.0
    total: float = 0.0

    def update(self, loss_dict: Mapping[str, Tensor]) -> None:
        matched = loss_dict.get("num_matches")
        total = loss_dict.get("num_targets")
        if matched is not None:
            self.matched += float(matched.detach().cpu().item())
        if total is not None:
            self.total += float(total.detach().cpu().item())

    def compute(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.matched / max(self.total, 1e-6)

    def reset(self) -> None:
        self.matched = 0.0
        self.total = 0.0


_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


class SceneModelLightningModule(LightningModule):
    """End-to-end Lightning wrapper that wires the SceneModel, head, and denoiser."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_scene_model(cfg.get("model"), cfg.get("debug"))
        self.head_adapter = HeadAdapter(cfg.get("model"))
        self.denoiser = DinoDenoiser(cfg.get("training", {}).get("denoiser"))
        self.metric_state = _MetricState()
        backbone_cfg = _to_dict(cfg.get("model", {})).get("backbone", {})
        self._freeze_backbone = bool(backbone_cfg.get("freeze", False))
        unfreeze_after = backbone_cfg.get("unfreeze_after")
        self._unfreeze_after = (
            int(unfreeze_after) if unfreeze_after is not None else None
        )
        self._backbone_frozen = False
        dataset_cfg = _to_dict(cfg.get("dataset", {}))
        image_cfg = dataset_cfg.get("image", {})
        norm_cfg = image_cfg.get("normalize", {})
        mean = torch.tensor(norm_cfg.get("mean", _DEFAULT_MEAN), dtype=torch.float32)
        std = torch.tensor(norm_cfg.get("std", _DEFAULT_STD), dtype=torch.float32)
        self.register_buffer("_image_mean", mean.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_image_std", std.view(1, 3, 1, 1), persistent=False)
        viz_cfg = _to_dict(cfg.get("logging", {})).get("visualizer", {})
        self._viz_max_frames = int(viz_cfg.get("max_frames", 4))
        self._viz_max_boxes = int(viz_cfg.get("max_boxes", 5))
        self._viz_exist_threshold = float(viz_cfg.get("exist_threshold", 0.5))
        self._apply_initial_backbone_freeze()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(
        self, frames: Tensor
    ) -> Mapping[str, Tensor]:  # pragma: no cover - thin wrapper
        """Proxy the PyTorch-style forward pass to the underlying SceneModel."""
        return cast(Mapping[str, Tensor], self.model(frames))

    def training_step(self, batch: SceneBatch, batch_idx: int) -> Tensor:
        """Execute one optimization step and log individual loss components."""
        outputs = self.model(batch.frames)
        dn_state = self.denoiser.make_noise(batch.targets)
        loss_dict = self.head_adapter.compute_loss(
            outputs, batch.targets, batch.padding_mask, dn_state
        )
        self._log_losses(loss_dict, stage="train")
        return loss_dict["total"]

    def validation_step(self, batch: SceneBatch, batch_idx: int) -> None:
        """Compute validation losses/metrics for logging."""
        outputs = self.model(batch.frames)
        dn_state = self.denoiser.make_noise(batch.targets)
        loss_dict = self.head_adapter.compute_loss(
            outputs, batch.targets, batch.padding_mask, dn_state
        )
        self._log_losses(loss_dict, stage="val")
        self.metric_state.update(loss_dict)
        if self._should_log_images(batch_idx):
            grid = self._build_visual_grid(batch, outputs)
            if grid is not None:
                self._log_tensorboard_image(
                    "val/bbox_predictions", grid, int(self.global_step)
                )

    def on_validation_epoch_end(self) -> None:
        """Publish aggregate IDF1-style metric at the end of validation."""
        score = self.metric_state.compute()
        self.log("val/idf1", score, prog_bar=True, sync_dist=False)
        self.metric_state.reset()

    def on_train_epoch_start(self) -> None:
        """Update backbone freeze policy based on the current epoch."""
        super().on_train_epoch_start()
        self._maybe_update_backbone_freeze(int(self.current_epoch))

    def configure_optimizers(self) -> Any:
        """Create the optimizer and optional cosine scheduler."""
        opt_cfg = _to_dict(self.cfg.get("training", {}).get("optimizer"))
        lr = float(opt_cfg.get("lr", 2e-4))
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        )
        sched_cfg = _to_dict(self.cfg.get("training", {}).get("scheduler"))
        if not sched_cfg:
            return optimizer
        trainer_cfg = _to_dict(self.cfg.get("training", {}).get("trainer"))
        max_epochs = int(trainer_cfg.get("max_epochs", 1))
        min_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * min_ratio
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _log_losses(self, loss_dict: Mapping[str, Tensor], stage: str) -> None:
        """Log every tracked loss/metric entry."""
        for key, value in loss_dict.items():
            if not isinstance(value, Tensor):
                continue
            tag = f"{stage}/{key}"
            self.log(tag, value, prog_bar=key == "total", sync_dist=False)
            if stage == "val" and key == "total":
                self.log("val/total_loss", value, prog_bar=False, sync_dist=False)

    def _apply_initial_backbone_freeze(self) -> None:
        if not self._freeze_backbone:
            return
        self._set_backbone_trainable(False)
        self._backbone_frozen = True

    def _set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.model.backbone.parameters():
            param.requires_grad = trainable

    def _maybe_update_backbone_freeze(self, epoch: int) -> None:
        if (
            self._freeze_backbone
            and self._backbone_frozen
            and self._unfreeze_after is not None
            and epoch >= self._unfreeze_after
        ):
            self._set_backbone_trainable(True)
            self._backbone_frozen = False

    def _should_log_images(self, batch_idx: int) -> bool:
        if self._viz_max_frames <= 0 or batch_idx > 0:
            return False
        trainer = getattr(self, "trainer", None)
        return trainer is None or getattr(trainer, "global_rank", 0) == 0

    def _build_visual_grid(
        self, batch: SceneBatch, outputs: Mapping[str, Tensor]
    ) -> Tensor | None:
        centers = outputs.get("bbox_center")
        sizes = outputs.get("bbox_size")
        exist = outputs.get("exist_conf")
        if centers is None or sizes is None or exist is None:
            return None
        if centers.ndim != 4 or sizes.ndim != 4:
            return None
        frames = batch.frames
        if frames.size(0) == 0:
            return None
        frames_0 = frames[0]
        mask = batch.padding_mask[0]
        valid = int((~mask).sum().item())
        if valid <= 0:
            return None
        centers = centers[0, :valid]
        sizes = sizes[0, :valid]
        exist = exist[0, :valid].squeeze(-1)
        boxes = torch.cat(
            [centers - 0.5 * sizes, centers + 0.5 * sizes],
            dim=-1,
        )
        denorm = self._denormalize_frames(frames_0[:valid]).clamp(0.0, 1.0)
        denorm = denorm.detach().cpu()
        boxes = boxes.detach().cpu()
        exist = exist.detach().cpu()
        num_frames = min(valid, self._viz_max_frames)
        annotated: list[Tensor] = []
        for idx in range(num_frames):
            annotated.append(
                self._draw_boxes_on_frame(denorm[idx], boxes[idx], exist[idx])
            )
        if not annotated:
            return None
        grid = make_grid(torch.stack(annotated), nrow=num_frames)
        return grid.float() / 255.0

    def _draw_boxes_on_frame(
        self, frame: Tensor, boxes: Tensor, scores: Tensor
    ) -> Tensor:
        image = (frame * 255).to(torch.uint8)
        scores = scores.view(-1)
        num_boxes = boxes.shape[0]
        if num_boxes == 0:
            return image
        keep = scores > self._viz_exist_threshold
        if not torch.any(keep):
            if scores.numel() == 0:
                return image
            top_k = min(self._viz_max_boxes, scores.numel())
            top_idx = torch.topk(scores, k=top_k).indices
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep[top_idx] = True
        selected = boxes[keep]
        selected_scores = scores[keep]
        if selected.numel() == 0:
            return image
        h, w = frame.shape[1], frame.shape[2]
        clipped = selected.clone()
        clipped[:, 0].clamp_(0, w - 1)
        clipped[:, 2].clamp_(0, w - 1)
        clipped[:, 1].clamp_(0, h - 1)
        clipped[:, 3].clamp_(0, h - 1)
        labels = [f"{float(score):.2f}" for score in selected_scores]
        return draw_bounding_boxes(
            image, clipped, labels=labels, colors="lime", width=2
        )

    def _denormalize_frames(self, frames: Tensor) -> Tensor:
        mean = self._image_mean.to(frames.device)
        std = self._image_std.to(frames.device)
        return frames * std + mean

    def _log_tensorboard_image(self, tag: str, image: Tensor, step: int) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        for writer in self._iter_tensorboard_writers(logger):
            writer.add_image(tag, image, step)

    def _iter_tensorboard_writers(self, logger: Any) -> Iterable[Any]:
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


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
