"""LightningModule for DanceTrack SceneModel training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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


class SceneModelLightningModule(LightningModule):
    """End-to-end Lightning wrapper that wires the SceneModel, head, and denoiser."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_scene_model(cfg.get("model"), cfg.get("debug"))
        self.head_adapter = HeadAdapter(cfg.get("model"))
        self.denoiser = DinoDenoiser(cfg.get("training", {}).get("denoiser"))
        self.metric_state = _MetricState()
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

    def on_validation_epoch_end(self) -> None:
        """Publish aggregate IDF1-style metric at the end of validation."""
        score = self.metric_state.compute()
        self.log("val/idf1", score, prog_bar=True, sync_dist=False)
        self.metric_state.reset()

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


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)
