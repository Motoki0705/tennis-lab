"""LightningModule for Tennis-DETR training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.tennis import TennisDETR, TennisDetrConfig


class TennisDetrModule(LightningModule):
    """Lightning wrapper that wires TennisDETR with loss and logging.

    Initialize the module from the merged experiment config. The
    ``cfg.model`` section is used to build :class:`TennisDetrConfig`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = _to_dict(cfg.get("model"))
        self.model = TennisDETR(_build_model_cfg(model_cfg))
        training_cfg = _to_dict(cfg.get("training", {}))
        optim_cfg = training_cfg.get("optimizer", {})
        self._lr = float(optim_cfg.get("lr", 1e-4))
        self._weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        self._max_steps = int(training_cfg.get("max_steps", 0))
        loss_cfg = training_cfg.get("loss", {})
        self._lambda_pose = float(loss_cfg.get("lambda_pose", 1.0))
        self._lambda_exist = float(loss_cfg.get("lambda_exist", 1.0))
        self._lambda_vel = float(loss_cfg.get("lambda_vel", 0.0))
        viz_cfg = _to_dict(cfg.get("logging", {})).get("visualizer", {})
        self._viz_max_batches = int(viz_cfg.get("max_batches", 2))
        self._exist_threshold = float(viz_cfg.get("exist_threshold", 0.5))
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(
        self,
        batch: Mapping[str, Tensor],
    ) -> Mapping[str, Tensor]:  # pragma: no cover - thin wrapper
        """Run a forward pass through the underlying TennisDETR model."""
        return cast(
            Mapping[str, Tensor],
            self.model(
                player_kpts_2d=batch["keypoints_2d"],
                player_mask=batch["player_mask"],
                court_kpts_2d=batch["court_2d"],
            ),
        )

    def training_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Execute one optimization step and log training losses."""
        outputs = self.forward(batch)
        loss_dict = self._compute_loss(outputs, batch)
        self._log_losses(loss_dict, stage="train")
        return loss_dict["total"]

    def validation_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Compute validation losses and optionally log qualitative outputs."""
        outputs = self.forward(batch)
        loss_dict = self._compute_loss(outputs, batch)
        self._log_losses(loss_dict, stage="val")
        if batch_idx < self._viz_max_batches:
            image_gt, image_pred = self._render_debug_images(batch, outputs)
            step = int(self.global_step)
            if image_gt is not None:
                self._log_tensorboard_image("val/pose2d_gt", image_gt, step)
            if image_pred is not None:
                self._log_tensorboard_image("val/pose2d_pred", image_pred, step)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and LR scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        if self._max_steps <= 0:
            return {"optimizer": optimizer}
        scheduler = CosineAnnealingLR(optimizer, T_max=self._max_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _compute_loss(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute masked L1 loss for 3D pose and BCE for existence."""
        pose_pred = cast(Tensor, outputs["pose_3d"])  # [B, Q, T, J, 3]
        exist_logit = cast(Tensor, outputs["exist_logit"])  # [B, Q, 1]
        pose_gt = batch["pose_3d_gt"]  # [B, T, M, J, 3]
        exist_gt = batch["exist_3d_gt"]  # [B, T, M]

        B, Q, T, J, _ = pose_pred.shape
        _, T_gt, M, J_gt, _ = pose_gt.shape
        if T_gt != T or J_gt != J:
            msg = "Ground-truth pose shape does not match predictions"
            raise ValueError(msg)

        # Align GT to queries: first M players map to first M queries.
        max_assign = min(Q, M)
        pose_gt_expanded = torch.zeros_like(pose_pred)
        exist_gt_query = torch.zeros(
            (B, Q, 1), dtype=torch.float32, device=pose_pred.device
        )

        pose_gt_perm = pose_gt.permute(0, 2, 1, 3, 4)  # [B, M, T, J, 3]
        pose_gt_expanded[:, :max_assign, :, :, :] = pose_gt_perm[
            :, :max_assign, :, :, :
        ]

        exist_gt_any = exist_gt.any(dim=1)  # [B, M]
        exist_gt_query[:, :max_assign, 0] = exist_gt_any[:, :max_assign].to(
            dtype=torch.float32
        )

        pose_mask = exist_gt_query > 0.0  # [B, Q, 1]
        pose_mask = pose_mask.unsqueeze(-1).unsqueeze(-1)  # [B, Q, 1, 1, 1]

        pose_l1 = torch.abs(pose_pred - pose_gt_expanded)
        pose_l1 = pose_l1 * pose_mask
        denom = pose_mask.sum().clamp_min(1.0)
        pose_loss = pose_l1.sum() / denom

        exist_loss = F.binary_cross_entropy_with_logits(exist_logit, exist_gt_query)

        vel_loss = torch.tensor(0.0, device=pose_pred.device)
        if self._lambda_vel > 0.0:
            vel = pose_pred[:, :, 1:, :, :] - pose_pred[:, :, :-1, :, :]
            vel_loss = (vel.pow(2.0)).mean()

        total = (
            self._lambda_pose * pose_loss
            + self._lambda_exist * exist_loss
            + self._lambda_vel * vel_loss
        )
        return {
            "total": total,
            "pose_l1": pose_loss.detach(),
            "exist_bce": exist_loss.detach(),
            "vel_l2": vel_loss.detach(),
        }

    def _render_debug_images(
        self,
        batch: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
    ) -> tuple[Tensor | None, Tensor | None]:
        """Render GT and Pred 2D overlays (with court) for TensorBoard."""
        try:
            from src.visualize.tennis_render import render_pose2d_frame
        except Exception:  # pragma: no cover - visualization is best-effort
            return None, None

        keypoints_2d = batch["keypoints_2d"]  # [B, T, V, M, J, 2]
        player_mask = batch["player_mask"]  # [B, T, V, M]
        court_2d = batch["court_2d"]  # [B, V, 20, 2]
        pose_pred = outputs["pose_3d"]  # [B, Q, T, J, 3]

        if keypoints_2d.ndim != 6 or pose_pred.ndim != 5 or court_2d.ndim != 4:
            return None, None
        B, T, V, M, J, _ = keypoints_2d.shape
        if B == 0 or T == 0 or V == 0 or M == 0:
            return None, None

        # First batch, first frame, first camera.
        kp = keypoints_2d[0, 0, 0]  # [M, J, 2], in [-1, 1]
        mask = player_mask[0, 0, 0]  # [M]
        court = court_2d[0, 0]  # [20, 2], in [-1, 1]
        Q = pose_pred.shape[1]

        H, W = 288, 512

        def _denorm_to_px(arr_2d: Tensor) -> np.ndarray:
            coords = arr_2d.detach().float().cpu().numpy().astype("float32")
            coords_px = np.empty_like(coords)
            coords_px[..., 0] = (coords[..., 0] + 1.0) * 0.5 * float(W - 1)
            coords_px[..., 1] = (coords[..., 1] + 1.0) * 0.5 * float(H - 1)
            return coords_px

        court_px = _denorm_to_px(court)  # [20,2]

        # GT image.
        player_pose_list_gt: list[np.ndarray] = []
        racket_list_gt: list[np.ndarray] = []
        for m in range(M):
            if not bool(mask[m].item()):
                continue
            pts = kp[m]  # [J,2]
            pts_px = _denorm_to_px(pts)  # [J,2]
            pose_px = pts_px[:17]
            racket_px = pts_px[17:]
            player_pose_list_gt.append(pose_px)
            racket_list_gt.append(racket_px)

        court_vis = [1] * int(court_px.shape[0])
        img_gt_np = render_pose2d_frame(
            width=W,
            height=H,
            court_points=court_px,
            court_visibility=court_vis,
            player_poses=player_pose_list_gt,
            player_pose_visibility=None,
            racket_points=racket_list_gt,
            racket_visibility=None,
        )
        img_gt = (
            torch.from_numpy(img_gt_np)
            .permute(2, 0, 1)
            .to(device=keypoints_2d.device, dtype=torch.float32)
            / 255.0
        )

        # Pred image: project 3D XY to 2D plane (debug-only).
        pose_slice = pose_pred[0, :, 0]  # [Q, J, 3]
        player_pose_list_pred: list[np.ndarray] = []
        racket_list_pred: list[np.ndarray] = []
        for q in range(Q):
            pts3d = pose_slice[q]  # [J,3]
            coords = pts3d[:, :2].detach().float().cpu().numpy().astype("float32")
            coords_px = np.empty_like(coords)
            coords_px[..., 0] = (coords[..., 0] * 0.1 + 0.5) * float(W - 1)
            coords_px[..., 1] = (coords[..., 1] * 0.1 + 0.5) * float(H - 1)
            pose_px = coords_px[:17]
            racket_px = coords_px[17:]
            player_pose_list_pred.append(pose_px)
            racket_list_pred.append(racket_px)

        img_pred_np = render_pose2d_frame(
            width=W,
            height=H,
            court_points=court_px,
            court_visibility=court_vis,
            player_poses=player_pose_list_pred,
            player_pose_visibility=None,
            racket_points=racket_list_pred,
            racket_visibility=None,
        )
        img_pred = (
            torch.from_numpy(img_pred_np)
            .permute(2, 0, 1)
            .to(device=keypoints_2d.device, dtype=torch.float32)
            / 255.0
        )

        return img_gt, img_pred

    def _log_losses(self, loss_dict: Mapping[str, Tensor], stage: str) -> None:
        """Log loss components with appropriate prefixes."""
        for key, value in loss_dict.items():
            tag = f"{stage}/{key}"
            self.log(tag, value, prog_bar=(key == "total"), sync_dist=False)

    def _log_tensorboard_image(self, tag: str, image: Tensor, step: int) -> None:
        """Log an image tensor to all TensorBoard-compatible loggers."""
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        for writer in self._iter_tensorboard_writers(logger):
            writer.add_image(tag, image, step)

    def _iter_tensorboard_writers(self, logger: Any) -> list[Any]:
        """Return all child loggers that expose a TensorBoard-like API."""
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


def _build_model_cfg(model_cfg: Mapping[str, Any]) -> TennisDetrConfig:
    """Construct TennisDetrConfig from a nested mapping."""
    cfg = TennisDetrConfig()
    if "D_model" in model_cfg:
        cfg.D_model = int(model_cfg["D_model"])
    if "dim_feedforward" in model_cfg:
        cfg.dim_feedforward = int(model_cfg["dim_feedforward"])
    if "nheads" in model_cfg:
        cfg.nheads = int(model_cfg["nheads"])
    if "encoder_layers" in model_cfg:
        cfg.encoder_layers = int(model_cfg["encoder_layers"])
    if "decoder_layers" in model_cfg:
        cfg.decoder_layers = int(model_cfg["decoder_layers"])
    if "dropout" in model_cfg:
        cfg.dropout = float(model_cfg["dropout"])
    if "num_joints" in model_cfg:
        cfg.num_joints = int(model_cfg["num_joints"])
    if "num_court_points" in model_cfg:
        cfg.num_court_points = int(model_cfg["num_court_points"])
    if "num_queries" in model_cfg:
        cfg.num_queries = int(model_cfg["num_queries"])
    if "max_cameras" in model_cfg:
        cfg.max_cameras = int(model_cfg["max_cameras"])
    if "max_frames" in model_cfg:
        cfg.max_frames = int(model_cfg["max_frames"])
    return cfg
