"""TennisDETR v3用のLightningモジュール."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.models.tennis_multi_cam_3d_pose import TennisDetrV3Config
from src.models.tennis_multi_cam_3d_pose.factory import validate_config_for_version
from src.models.tennis_multi_cam_3d_pose.model_v3 import TennisDETR_v3
from src.training.base.tennis_multi_cam_3d_pose import BaseTennisLightningModule


class TennisDetrV3Module(BaseTennisLightningModule):
    """TennisDETR v3モデル用のLightningモジュール.

    Args:
        cfg (DictConfig): 設定オブジェクト

    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = _to_dict(cfg.get("model"))
        self.model_cfg = _build_model_cfg(model_cfg)
        validate_config_for_version(self.model_cfg, "v3")
        self.model = TennisDETR_v3(self.model_cfg)

        training_cfg = _to_dict(cfg.get("training", {}))
        optim_cfg = training_cfg.get("optimizer", {})
        self._lr = float(optim_cfg.get("lr", 1e-4))
        self._weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
        self._max_steps = int(training_cfg.get("max_steps", 0))
        self._scheduler_cfg = _to_dict(training_cfg.get("scheduler"))

        loss_cfg = training_cfg.get("loss", {})
        self._lambda_canonical = float(loss_cfg.get("lambda_canonical", 1.0))
        self._lambda_root_trans = float(loss_cfg.get("lambda_root_trans", 1.0))
        self._lambda_root_rot = float(loss_cfg.get("lambda_root_rot", 0.5))
        self._lambda_global = float(loss_cfg.get("lambda_global", 1.0))
        self._lambda_exist = float(loss_cfg.get("lambda_exist", 1.0))
        self._lambda_vel = float(loss_cfg.get("lambda_vel", 0.0))
        self._lambda_pose_match = float(loss_cfg.get("lambda_pose_match", 1.0))
        self._lambda_exist_match = float(loss_cfg.get("lambda_exist_match", 1.0))
        self._lambda_denoise_canonical = float(
            loss_cfg.get("lambda_denoise_canonical", 0.0)
        )
        self._lambda_denoise_root_trans = float(
            loss_cfg.get("lambda_denoise_root_trans", 0.0)
        )
        self._lambda_denoise_root_rot = float(
            loss_cfg.get("lambda_denoise_root_rot", 0.0)
        )

        denoise_cfg = _to_dict(training_cfg.get("denoise3d", {}))
        self._denoise_canonical_noise_std = float(
            denoise_cfg.get("canonical_noise_std", 0.0)
        )
        self._denoise_root_trans_noise_std = float(
            denoise_cfg.get("trans_noise_std", 0.0)
        )
        self._denoise_root_rot_noise_std = float(denoise_cfg.get("rot_noise_std", 0.0))

        viz_cfg = _to_dict(cfg.get("logging", {})).get("visualizer", {})
        self._viz_max_batches = int(viz_cfg.get("max_batches", 2))
        self._exist_threshold = float(viz_cfg.get("exist_threshold", 0.5))
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """フォワードパスを実行する.

        Args:
            batch (Mapping[str, Tensor]): 入力バッチ

        Returns:
            Mapping[str, Tensor]: 出力テンソルのマッピング

        """
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
        """トレーニングステップを実行する."""
        return super().training_step(batch, batch_idx)

    def validation_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> None:
        """検証ステップを実行する."""
        return super().validation_step(batch, batch_idx)

    def configure_optimizers(self) -> dict[str, Any]:
        """オプティマイザとスケジューラを設定する.

        Returns:
            dict[str, Any]: オプティマイザとスケジューラの辞書

        """
        return super().configure_optimizers()

    def _build_scheduler(self, optimizer: AdamW) -> CosineAnnealingLR | LambdaLR | None:
        if self._max_steps <= 0:
            return None
        return super()._build_scheduler(optimizer)

    def _build_warmup_cosine_lambda(
        self,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> Callable[[int], float]:
        return super()._build_warmup_cosine_lambda(warmup_steps, min_lr_ratio)

    def _compute_loss(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        canonical_pred = cast(Tensor, outputs["canonical_pose"])
        root_trans_pred = cast(Tensor, outputs["root_trans"])
        root_rot_pred = cast(Tensor, outputs["root_rot"])
        global_pred = cast(Tensor, outputs["pose_3d"])
        exist_logit = cast(Tensor, outputs["exist_logit"])
        tracks_enc_all = cast(Tensor, outputs["tracks_enc"])

        canonical_gt = batch["canonical_pose_gt"]
        root_trans_gt = batch["root_trans_gt"]
        root_rot_gt = batch["root_rot_gt"]
        global_gt = batch["global_pose_gt"]
        exist_gt = batch["exist_3d_gt"]

        B, Q, T, J, _ = canonical_pred.shape
        _, T_gt, M, J_gt, _ = canonical_gt.shape
        if T_gt != T or J_gt != J:
            raise ValueError("Ground-truth pose shape does not match predictions")

        matches = self._match_queries_to_targets(
            global_pred, global_gt, exist_gt, exist_logit
        )

        exist_target = torch.zeros_like(exist_logit)
        canonical_loss_num = canonical_pred.new_tensor(0.0)
        canonical_loss_den = canonical_pred.new_tensor(0.0)
        root_trans_loss_num = root_trans_pred.new_tensor(0.0)
        root_trans_loss_den = root_trans_pred.new_tensor(0.0)
        root_rot_loss_num = root_rot_pred.new_tensor(0.0)
        root_rot_loss_den = root_rot_pred.new_tensor(0.0)
        global_loss_num = global_pred.new_tensor(0.0)
        global_loss_den = global_pred.new_tensor(0.0)
        denoise_canonical_num = canonical_pred.new_tensor(0.0)
        denoise_canonical_den = canonical_pred.new_tensor(0.0)
        denoise_root_trans_num = root_trans_pred.new_tensor(0.0)
        denoise_root_trans_den = root_trans_pred.new_tensor(0.0)
        denoise_root_rot_num = root_rot_pred.new_tensor(0.0)
        denoise_root_rot_den = root_rot_pred.new_tensor(0.0)

        for b, (matched_q, matched_t) in enumerate(matches):
            if matched_q.numel() == 0:
                continue
            exist_target[b, matched_q, 0] = 1.0
            mask_sel = exist_gt[b][:, matched_t].permute(1, 0)
            mask = mask_sel.unsqueeze(-1).unsqueeze(-1).to(dtype=canonical_pred.dtype)

            canonical_pred_sel = canonical_pred[b, matched_q]
            canonical_gt_sel = canonical_gt[b][:, matched_t, :, :].permute(1, 0, 2, 3)
            canonical_diff = torch.abs(canonical_pred_sel - canonical_gt_sel) * mask
            canonical_loss_num = canonical_loss_num + canonical_diff.sum()
            canonical_loss_den = canonical_loss_den + mask.sum() * float(J * 3)

            root_trans_pred_sel = root_trans_pred[b, matched_q]
            root_trans_gt_sel = root_trans_gt[b][:, matched_t, :].permute(1, 0, 2)
            root_trans_mask = mask_sel.unsqueeze(-1).to(dtype=root_trans_pred.dtype)
            root_trans_diff = (
                torch.abs(root_trans_pred_sel - root_trans_gt_sel) * root_trans_mask
            )
            root_trans_loss_num = root_trans_loss_num + root_trans_diff.sum()
            root_trans_loss_den = root_trans_loss_den + root_trans_mask.sum() * 3

            root_rot_pred_sel = root_rot_pred[b, matched_q]
            root_rot_gt_sel = root_rot_gt[b][:, matched_t, :].permute(1, 0, 2)
            root_rot_mask = mask_sel.unsqueeze(-1).to(dtype=root_rot_pred.dtype)
            root_rot_diff = (
                torch.abs(root_rot_pred_sel - root_rot_gt_sel) * root_rot_mask
            )
            root_rot_loss_num = root_rot_loss_num + root_rot_diff.sum()
            root_rot_loss_den = root_rot_loss_den + root_rot_mask.sum() * 2

            global_pred_sel = global_pred[b, matched_q]
            global_gt_sel = global_gt[b][:, matched_t, :, :].permute(1, 0, 2, 3)
            global_diff = torch.abs(global_pred_sel - global_gt_sel) * mask
            global_loss_num = global_loss_num + global_diff.sum()
            global_loss_den = global_loss_den + mask.sum() * float(J * 3)

            use_denoise_canonical = (
                self._lambda_denoise_canonical > 0.0
                and self._denoise_canonical_noise_std > 0.0
            )
            use_denoise_root_trans = (
                self._lambda_denoise_root_trans > 0.0
                and self._denoise_root_trans_noise_std > 0.0
            )
            use_denoise_root_rot = (
                self._lambda_denoise_root_rot > 0.0
                and self._denoise_root_rot_noise_std > 0.0
            )

            if not (
                use_denoise_canonical or use_denoise_root_trans or use_denoise_root_rot
            ):
                continue

            tracks_enc_sel = tracks_enc_all[b, matched_q]

            if use_denoise_canonical:
                canonical_noise = torch.randn_like(canonical_gt_sel).mul(
                    self._denoise_canonical_noise_std
                )
                canonical_noisy = canonical_gt_sel + canonical_noise
                N_sel, T_sel, J_sel, _ = canonical_gt_sel.shape
                canon_flat = canonical_noisy.reshape(N_sel, T_sel, J_sel * 3)
                canon_embed = self.model.canonical_in_proj(canon_flat)
                trans_embed = self.model.root_trans_in_proj(root_trans_gt_sel)
                rot_embed = self.model.root_rot_in_proj(root_rot_gt_sel)
                tokens_canonical = (
                    tracks_enc_sel + canon_embed + trans_embed + rot_embed
                )
                canon_out_flat = self.model.denoise_canonical_head(tokens_canonical)
                canonical_denoised = canon_out_flat.reshape(N_sel, T_sel, J_sel, 3)
                canonical_denoise_diff = (
                    torch.abs(canonical_denoised - canonical_gt_sel) * mask
                )
                denoise_canonical_num = (
                    denoise_canonical_num + canonical_denoise_diff.sum()
                )
                denoise_canonical_den = denoise_canonical_den + mask.sum() * float(
                    J * 3
                )

            if use_denoise_root_trans:
                root_trans_noise = torch.randn_like(root_trans_gt_sel).mul(
                    self._denoise_root_trans_noise_std
                )
                root_trans_noisy = root_trans_gt_sel + root_trans_noise
                N_sel, T_sel, _ = root_trans_gt_sel.shape
                canon_flat_gt = canonical_gt_sel.reshape(N_sel, T_sel, J * 3)
                canon_embed_gt = self.model.canonical_in_proj(canon_flat_gt)
                trans_embed_noisy = self.model.root_trans_in_proj(root_trans_noisy)
                rot_embed_gt = self.model.root_rot_in_proj(root_rot_gt_sel)
                tokens_root_trans = (
                    tracks_enc_sel + canon_embed_gt + trans_embed_noisy + rot_embed_gt
                )
                root_trans_denoised = self.model.denoise_root_trans_head(
                    tokens_root_trans
                )
                root_trans_denoise_diff = (
                    torch.abs(root_trans_denoised - root_trans_gt_sel) * root_trans_mask
                )
                denoise_root_trans_num = (
                    denoise_root_trans_num + root_trans_denoise_diff.sum()
                )
                denoise_root_trans_den = denoise_root_trans_den + (
                    root_trans_mask.sum() * 3
                )

            if use_denoise_root_rot:
                root_rot_noise = torch.randn_like(root_rot_gt_sel).mul(
                    self._denoise_root_rot_noise_std
                )
                root_rot_noisy = root_rot_gt_sel + root_rot_noise
                N_sel, T_sel, _ = root_rot_gt_sel.shape
                canon_flat_gt = canonical_gt_sel.reshape(N_sel, T_sel, J * 3)
                canon_embed_gt = self.model.canonical_in_proj(canon_flat_gt)
                trans_embed_gt = self.model.root_trans_in_proj(root_trans_gt_sel)
                rot_embed_noisy = self.model.root_rot_in_proj(root_rot_noisy)
                tokens_root_rot = (
                    tracks_enc_sel + canon_embed_gt + trans_embed_gt + rot_embed_noisy
                )
                root_rot_denoised = self.model.denoise_root_rot_head(tokens_root_rot)
                root_rot_denoise_diff = (
                    torch.abs(root_rot_denoised - root_rot_gt_sel) * root_rot_mask
                )
                denoise_root_rot_num = (
                    denoise_root_rot_num + root_rot_denoise_diff.sum()
                )
                denoise_root_rot_den = denoise_root_rot_den + root_rot_mask.sum() * 2

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
        vel_loss = torch.tensor(0.0, device=global_pred.device)
        if self._lambda_vel > 0.0:
            vel = global_pred[:, :, 1:, :, :] - global_pred[:, :, :-1, :, :]
            vel_loss = (vel.pow(2.0)).mean()

        denoise_canonical_loss = (
            denoise_canonical_num / denoise_canonical_den
            if denoise_canonical_den.item() > 0
            else canonical_pred.new_tensor(0.0)
        )
        denoise_root_trans_loss = (
            denoise_root_trans_num / denoise_root_trans_den
            if denoise_root_trans_den.item() > 0
            else root_trans_pred.new_tensor(0.0)
        )
        denoise_root_rot_loss = (
            denoise_root_rot_num / denoise_root_rot_den
            if denoise_root_rot_den.item() > 0
            else root_rot_pred.new_tensor(0.0)
        )

        total = (
            self._lambda_canonical * canonical_loss
            + self._lambda_root_trans * root_trans_loss
            + self._lambda_root_rot * root_rot_loss
            + self._lambda_global * global_loss
            + self._lambda_exist * exist_loss
            + self._lambda_vel * vel_loss
            + self._lambda_denoise_canonical * denoise_canonical_loss
            + self._lambda_denoise_root_trans * denoise_root_trans_loss
            + self._lambda_denoise_root_rot * denoise_root_rot_loss
        )

        return {
            "total": total,
            "canonical_l1": canonical_loss.detach(),
            "root_trans_l1": root_trans_loss.detach(),
            "root_rot_l1": root_rot_loss.detach(),
            "global_l1": global_loss.detach(),
            "exist_bce": exist_loss.detach(),
            "vel_l2": vel_loss.detach(),
            "denoise_canonical_l1": denoise_canonical_loss.detach(),
            "denoise_root_trans_l1": denoise_root_trans_loss.detach(),
            "denoise_root_rot_l1": denoise_root_rot_loss.detach(),
        }

    def _match_queries_to_targets(
        self,
        pose_pred: Tensor,
        pose_gt: Tensor,
        exist_gt: Tensor,
        exist_logit: Tensor,
    ) -> list[tuple[Tensor, Tensor]]:
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
        return super()._render_debug_images(batch, outputs)

    def _log_losses(self, loss_dict: Mapping[str, Tensor], stage: str) -> None:
        return super()._log_losses(loss_dict, stage)

    def _log_tensorboard_image(self, tag: str, image: Tensor, step: int) -> None:
        return super()._log_tensorboard_image(tag, image, step)

    def _iter_tensorboard_writers(self, logger: Any) -> list[Any]:
        return super()._iter_tensorboard_writers(logger)


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def _build_model_cfg(model_cfg: Mapping[str, Any]) -> TennisDetrV3Config:
    cfg = TennisDetrV3Config()
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
    if "intra_layers" in model_cfg:
        cfg.intra_layers = int(model_cfg["intra_layers"])
    if "inter_layers" in model_cfg:
        cfg.inter_layers = int(model_cfg["inter_layers"])
    if "temporal_layers" in model_cfg:
        cfg.temporal_layers = int(model_cfg["temporal_layers"])
    return cfg
