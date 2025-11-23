"""LightningModule for Tennis-DETR v2 training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.models.tennis_multi_cam_3d_pose import TennisDetrV2Config
from src.models.tennis_multi_cam_3d_pose.factory import validate_config_for_version
from src.models.tennis_multi_cam_3d_pose.model_v2 import TennisDETR_v2
from src.training.base.tennis_multi_cam_3d_pose import BaseTennisLightningModule


class TennisDetrV2Module(BaseTennisLightningModule):
    """Lightning wrapper that wires TennisDETR_v2 with loss and logging.

    Initialize the module from the merged experiment config. The
    ``cfg.model`` section is used to build :class:`TennisDetrConfig`.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = _to_dict(cfg.get("model"))
        self.model_cfg = _build_model_cfg(model_cfg)

        # v2設定の検証
        validate_config_for_version(self.model_cfg, "v2")

        self.model = TennisDETR_v2(self.model_cfg)

        training_cfg = _to_dict(cfg.get("training", {}))
        optim_cfg = training_cfg.get("optimizer", {})
        self._lr = float(optim_cfg.get("lr", 1e-4))
        self._weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        self._max_steps = int(training_cfg.get("max_steps", 0))
        self._scheduler_cfg = _to_dict(training_cfg.get("scheduler"))

        # v2用損失重み
        loss_cfg = training_cfg.get("loss", {})
        self._lambda_canonical = float(loss_cfg.get("lambda_canonical", 1.0))
        self._lambda_root_trans = float(loss_cfg.get("lambda_root_trans", 1.0))
        self._lambda_root_rot = float(loss_cfg.get("lambda_root_rot", 0.5))
        self._lambda_global = float(loss_cfg.get("lambda_global", 1.0))
        self._lambda_exist = float(loss_cfg.get("lambda_exist", 1.0))
        self._lambda_vel = float(loss_cfg.get("lambda_vel", 0.0))

        # マッチング用重み
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
        """Run a forward pass through the underlying TennisDETR_v2 model."""
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
        """Compute v2-specific losses for decomposed pose components."""
        # 予測値
        canonical_pred = cast(Tensor, outputs["canonical_pose"])  # [B, Q, T, J, 3]
        root_trans_pred = cast(Tensor, outputs["root_trans"])  # [B, Q, T, 3]
        root_rot_pred = cast(Tensor, outputs["root_rot"])  # [B, Q, T, 2]
        global_pred = cast(Tensor, outputs["pose_3d"])  # [B, Q, T, J, 3]
        exist_logit = cast(Tensor, outputs["exist_logit"])  # [B, Q, 1]

        # GT値（v2用データ）
        canonical_gt = batch["canonical_pose_gt"]  # [B, T, M, J, 3]
        root_trans_gt = batch["root_trans_gt"]  # [B, T, M, 3]
        root_rot_gt = batch["root_rot_gt"]  # [B, T, M, 2]
        global_gt = batch["global_pose_gt"]  # [B, T, M, J, 3]
        exist_gt = batch["exist_3d_gt"]  # [B, T, M]

        B, Q, T, J, _ = canonical_pred.shape
        _, T_gt, M, J_gt, _ = canonical_gt.shape
        if T_gt != T or J_gt != J:
            msg = "Ground-truth pose shape does not match predictions"
            raise ValueError(msg)

        # ハンガリアンマッチング（global_poseを使用）
        matches = self._match_queries_to_targets(
            global_pred, global_gt, exist_gt, exist_logit
        )

        # 損失計算用の変数初期化
        exist_target = torch.zeros_like(exist_logit)

        # 各コンポーネントの損失を計算
        canonical_loss_num = canonical_pred.new_tensor(0.0)
        canonical_loss_den = canonical_pred.new_tensor(0.0)

        root_trans_loss_num = root_trans_pred.new_tensor(0.0)
        root_trans_loss_den = root_trans_pred.new_tensor(0.0)

        root_rot_loss_num = root_rot_pred.new_tensor(0.0)
        root_rot_loss_den = root_rot_pred.new_tensor(0.0)

        global_loss_num = global_pred.new_tensor(0.0)
        global_loss_den = global_pred.new_tensor(0.0)

        for b, (matched_q, matched_t) in enumerate(matches):
            if matched_q.numel() == 0:
                continue
            exist_target[b, matched_q, 0] = 1.0

            # 各コンポーネントの損失を計算
            mask_sel = exist_gt[b][:, matched_t].permute(1, 0)  # [K, T]
            mask = mask_sel.unsqueeze(-1).unsqueeze(-1).to(dtype=canonical_pred.dtype)

            # Canonical pose損失
            canonical_pred_sel = canonical_pred[b, matched_q]  # [K, T, J, 3]
            canonical_gt_sel = canonical_gt[b][:, matched_t, :, :].permute(1, 0, 2, 3)
            canonical_diff = torch.abs(canonical_pred_sel - canonical_gt_sel) * mask
            canonical_loss_num = canonical_loss_num + canonical_diff.sum()
            canonical_loss_den = canonical_loss_den + mask.sum() * float(J * 3)

            # Root translation損失
            root_trans_pred_sel = root_trans_pred[b, matched_q]  # [K, T, 3]
            root_trans_gt_sel = root_trans_gt[b][:, matched_t, :].permute(1, 0, 2)
            root_trans_mask = mask_sel.unsqueeze(-1).to(dtype=root_trans_pred.dtype)
            root_trans_diff = (
                torch.abs(root_trans_pred_sel - root_trans_gt_sel) * root_trans_mask
            )
            root_trans_loss_num = root_trans_loss_num + root_trans_diff.sum()
            root_trans_loss_den = root_trans_loss_den + root_trans_mask.sum() * 3

            # Root rotation損失
            root_rot_pred_sel = root_rot_pred[b, matched_q]  # [K, T, 2]
            root_rot_gt_sel = root_rot_gt[b][:, matched_t, :].permute(1, 0, 2)
            root_rot_mask = mask_sel.unsqueeze(-1).to(dtype=root_rot_pred.dtype)
            root_rot_diff = (
                torch.abs(root_rot_pred_sel - root_rot_gt_sel) * root_rot_mask
            )
            root_rot_loss_num = root_rot_loss_num + root_rot_diff.sum()
            root_rot_loss_den = root_rot_loss_den + root_rot_mask.sum() * 2

            # Global pose損失
            global_pred_sel = global_pred[b, matched_q]  # [K, T, J, 3]
            global_gt_sel = global_gt[b][:, matched_t, :, :].permute(1, 0, 2, 3)
            global_diff = torch.abs(global_pred_sel - global_gt_sel) * mask
            global_loss_num = global_loss_num + global_diff.sum()
            global_loss_den = global_loss_den + mask.sum() * float(J * 3)

        # 最終損失を計算
        canonical_loss = (
            (canonical_loss_num / canonical_loss_den)
            if canonical_loss_den.item() > 0
            else canonical_pred.new_tensor(0.0)
        )
        root_trans_loss = (
            (root_trans_loss_num / root_trans_loss_den)
            if root_trans_loss_den.item() > 0
            else root_trans_pred.new_tensor(0.0)
        )
        root_rot_loss = (
            (root_rot_loss_num / root_rot_loss_den)
            if root_rot_loss_den.item() > 0
            else root_rot_pred.new_tensor(0.0)
        )
        global_loss = (
            (global_loss_num / global_loss_den)
            if global_loss_den.item() > 0
            else global_pred.new_tensor(0.0)
        )

        exist_loss = F.binary_cross_entropy_with_logits(exist_logit, exist_target)

        # Velocity損失（global poseで計算）
        vel_loss = torch.tensor(0.0, device=global_pred.device)
        if self._lambda_vel > 0.0:
            vel = global_pred[:, :, 1:, :, :] - global_pred[:, :, :-1, :, :]
            vel_loss = (vel.pow(2.0)).mean()

        # 総合損失
        total = (
            self._lambda_canonical * canonical_loss
            + self._lambda_root_trans * root_trans_loss
            + self._lambda_root_rot * root_rot_loss
            + self._lambda_global * global_loss
            + self._lambda_exist * exist_loss
            + self._lambda_vel * vel_loss
        )

        return {
            "total": total,
            "canonical_l1": canonical_loss.detach(),
            "root_trans_l1": root_trans_loss.detach(),
            "root_rot_l1": root_rot_loss.detach(),
            "global_l1": global_loss.detach(),
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


def _build_model_cfg(model_cfg: Mapping[str, Any]) -> TennisDetrV2Config:
    """Construct TennisDetrV2Config from a nested mapping."""
    cfg = TennisDetrV2Config()

    # 共通パラメータ
    if "D_model" in model_cfg:
        cfg.D_model = int(model_cfg["D_model"])
    if "dim_feedforward" in model_cfg:
        cfg.dim_feedforward = int(model_cfg["dim_feedforward"])
    if "nheads" in model_cfg:
        cfg.nheads = int(model_cfg["nheads"])
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

    # v2用パラメータ
    if "intra_layers" in model_cfg:
        cfg.intra_layers = int(model_cfg["intra_layers"])
    if "inter_layers" in model_cfg:
        cfg.inter_layers = int(model_cfg["inter_layers"])
    if "temporal_layers" in model_cfg:
        cfg.temporal_layers = int(model_cfg["temporal_layers"])

    return cfg
