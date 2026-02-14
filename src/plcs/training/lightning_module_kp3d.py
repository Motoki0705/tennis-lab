"""PyTorch Lightning module for PLCS keypoint-3D training."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.plcs.models.plcs_kp3d_model import PLCSKeypoint3DModel
from src.plcs.training.losses_kp3d import PLCSKeypoint3DLoss
from src.plcs.training.metrics_kp3d import PLCSKeypoint3DMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSKeypoint3DLightningModule(BaseLightningModule):
    """Lightning module for keypoint-3D variant of PLCS frame model."""

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)

        self.model: PLCSKeypoint3DModel = PLCSKeypoint3DModel.from_config(self.config)

        loss_cfg = self.config.get("loss", {})
        self.loss_fn = PLCSKeypoint3DLoss(
            keypoint_weight=float(loss_cfg.get("keypoint_3d_weight", 1.0))
        )

        self.train_metrics = PLCSKeypoint3DMetrics()
        self.val_metrics = PLCSKeypoint3DMetrics()
        self.test_metrics = PLCSKeypoint3DMetrics()

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if human_kp.dim() != 3:
            raise ValueError(
                "PLCSKeypoint3DLightningModule expects human_kp as (B,17,2), "
                f"got shape {tuple(human_kp.shape)}"
            )
        if court_kp.dim() != 3:
            raise ValueError(
                "PLCSKeypoint3DLightningModule expects court_kp as (B,20,2), "
                f"got shape {tuple(court_kp.shape)}"
            )
        return cast(dict[str, Tensor], self.model(human_kp, court_kp, human_vis, court_vis))

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        human_kp = batch["human_kp"]
        court_kp = batch["court_kp"]
        human_vis = batch.get("human_vis")
        court_vis = batch.get("court_vis")
        human_kp_3d = batch["human_kp_3d"]

        if human_kp.dim() != 3:
            raise ValueError(
                "Expected human_kp as (B,17,2) for kp3d training, "
                f"got shape {tuple(human_kp.shape)}"
            )
        if court_kp.dim() != 3:
            raise ValueError(
                "Expected court_kp as (B,20,2) for kp3d training, "
                f"got shape {tuple(court_kp.shape)}"
            )
        if human_kp_3d.dim() != 3:
            raise ValueError(
                "Expected human_kp_3d as (B,17,3) for kp3d training, "
                f"got shape {tuple(human_kp_3d.shape)}"
            )

        outputs = self.model(
            human_kp=human_kp,
            court_kp=court_kp,
            human_vis=human_vis,
            court_vis=court_vis,
        )

        losses = self.loss_fn(
            pred_player_kp_3d=outputs["player_kp_3d"],
            target_player_kp_3d=human_kp_3d,
            human_vis=human_vis,
        )

        if stage == "train":
            metrics = self.train_metrics.update(
                outputs["player_kp_3d"],
                human_kp_3d,
                human_vis=human_vis,
            )
        elif stage == "val":
            metrics = self.val_metrics.update(
                outputs["player_kp_3d"],
                human_kp_3d,
                human_vis=human_vis,
            )
        else:
            metrics = self.test_metrics.update(
                outputs["player_kp_3d"],
                human_kp_3d,
                human_vis=human_vis,
            )

        return losses["total"], {
            **metrics,
            **{f"loss_{k}": v.item() for k, v in losses.items()},
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mpjpe_m", metrics["mpjpe_m"], prog_bar=True)
        self.log("train/pelvis_error_m", metrics["pelvis_error_m"])
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mpjpe_m", metrics["mpjpe_m"], prog_bar=True)
        self.log("val/pelvis_error_m", metrics["pelvis_error_m"])

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/mpjpe_m", metrics["mpjpe_m"])
        self.log("test/pelvis_error_m", metrics["pelvis_error_m"])

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()
