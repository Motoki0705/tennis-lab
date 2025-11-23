"""LightningModule for Tennis-DETR training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.models.tennis_multi_cam_3d_pose import TennisDETR, TennisDetrConfig
from src.training.base.tennis_multi_cam_3d_pose import BaseTennisLightningModule


class TennisDetrModule(BaseTennisLightningModule):
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
        self._scheduler_cfg = _to_dict(training_cfg.get("scheduler"))
        loss_cfg = training_cfg.get("loss", {})
        self._lambda_pose = float(loss_cfg.get("lambda_pose", 1.0))
        self._lambda_exist = float(loss_cfg.get("lambda_exist", 1.0))
        self._lambda_vel = float(loss_cfg.get("lambda_vel", 0.0))
        self._lambda_pose_match = float(loss_cfg.get("lambda_pose_match", 1.0))
        self._lambda_exist_match = float(loss_cfg.get("lambda_exist_match", 1.0))
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
        return super().training_step(batch, batch_idx)

    def validation_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> None:
        """Compute validation losses and optionally log qualitative outputs."""
        return super().validation_step(batch, batch_idx)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and LR scheduler."""
        return super().configure_optimizers()

    def _build_scheduler(self, optimizer: AdamW) -> CosineAnnealingLR | LambdaLR | None:
        """Return the configured LR scheduler or ``None`` if disabled."""
        return super()._build_scheduler(optimizer)

    def _build_warmup_cosine_lambda(
        self,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> Callable[[int], float]:
        """Construct a lambda function implementing warmup + cosine decay."""
        return super()._build_warmup_cosine_lambda(warmup_steps, min_lr_ratio)

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

        matches = self._match_queries_to_targets(
            pose_pred, pose_gt, exist_gt, exist_logit
        )

        exist_target = torch.zeros_like(exist_logit)
        pose_loss_num = pose_pred.new_tensor(0.0)
        pose_loss_den = pose_pred.new_tensor(0.0)

        for b, (matched_q, matched_t) in enumerate(matches):
            if matched_q.numel() == 0:
                continue
            exist_target[b, matched_q, 0] = 1.0
            pred_sel = pose_pred[b, matched_q]  # [K, T, J, 3]
            gt_sel = pose_gt[b][:, matched_t, :, :].permute(1, 0, 2, 3)  # [K, T, J, 3]
            mask_sel = exist_gt[b][:, matched_t].permute(1, 0)  # [K, T]
            mask = mask_sel.unsqueeze(-1).unsqueeze(-1).to(dtype=pred_sel.dtype)
            diff = torch.abs(pred_sel - gt_sel) * mask
            pose_loss_num = pose_loss_num + diff.sum()
            pose_loss_den = pose_loss_den + mask.sum() * float(J * 3)

        if pose_loss_den.item() > 0:
            pose_loss = pose_loss_num / pose_loss_den
        else:
            pose_loss = pose_pred.new_tensor(0.0)

        exist_loss = F.binary_cross_entropy_with_logits(exist_logit, exist_target)

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

    def _match_queries_to_targets(
        self,
        pose_pred: Tensor,
        pose_gt: Tensor,
        exist_gt: Tensor,
        exist_logit: Tensor,
    ) -> list[tuple[Tensor, Tensor]]:
        """Run Hungarian matching between predicted queries and GT players."""
        return super()._match_queries_to_targets(
            pose_pred=pose_pred,
            pose_gt=pose_gt,
            exist_gt=exist_gt,
            exist_logit=exist_logit,
        )

    def _render_debug_images(
        self,
        batch: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
    ) -> tuple[Tensor | None, Tensor | None]:
        """Render GT and Pred 2D overlays (with court) for TensorBoard."""
        return super()._render_debug_images(batch, outputs)

    def _log_losses(self, loss_dict: Mapping[str, Tensor], stage: str) -> None:
        """Log loss components with appropriate prefixes."""
        return super()._log_losses(loss_dict, stage)

    def _log_tensorboard_image(self, tag: str, image: Tensor, step: int) -> None:
        """Log an image tensor to all TensorBoard-compatible loggers."""
        return super()._log_tensorboard_image(tag, image, step)

    def _iter_tensorboard_writers(self, logger: Any) -> list[Any]:
        """Return all child loggers that expose a TensorBoard-like API."""
        return super()._iter_tensorboard_writers(logger)


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
