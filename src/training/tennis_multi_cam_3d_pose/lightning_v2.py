"""LightningModule for Tennis-DETR v2 training."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.models.tennis_multi_cam_3d_pose import TennisDetrV2Config
from src.models.tennis_multi_cam_3d_pose.factory import validate_config_for_version
from src.models.tennis_multi_cam_3d_pose.model_v2 import TennisDETR_v2
from src.tennis.geometry.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    NET_HEIGHT_POST,
)


class TennisDetrV2Module(LightningModule):
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
                self._log_tensorboard_image("val/pose2d_pred_reproj", image_pred, step)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and LR scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def _build_scheduler(self, optimizer: AdamW) -> CosineAnnealingLR | LambdaLR | None:
        """Return the configured LR scheduler or ``None`` if disabled."""
        if self._max_steps <= 0:
            return None
        scheduler_name = str(self._scheduler_cfg.get("name") or "").lower()
        if scheduler_name == "cosine_with_warmup":
            warmup_steps = int(self._scheduler_cfg.get("warmup_steps", 0))
            min_lr_ratio = float(self._scheduler_cfg.get("min_lr_ratio", 0.0))
            lr_lambda = self._build_warmup_cosine_lambda(warmup_steps, min_lr_ratio)
            return LambdaLR(optimizer, lr_lambda=lr_lambda)
        return CosineAnnealingLR(optimizer, T_max=self._max_steps)

    def _build_warmup_cosine_lambda(
        self,
        warmup_steps: int,
        min_lr_ratio: float,
    ) -> Callable[[int], float]:
        """Construct a lambda function implementing warmup + cosine decay."""
        warmup = max(0, int(warmup_steps))
        base_min_ratio = float(min_lr_ratio)
        max_steps = max(1, self._max_steps)

        def _lr_lambda(step: int) -> float:
            step_f = float(step)
            if warmup > 0 and step_f < warmup:
                return step_f / float(max(1, warmup))
            progress_steps = max(1, max_steps - warmup)
            progress = min(max((step_f - warmup) / progress_steps, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_min_ratio + (1.0 - base_min_ratio) * cos

        return _lr_lambda

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
        B, Q, _, _, _ = pose_pred.shape
        _, _, M, J, _ = pose_gt.shape
        exist_any = exist_gt.any(dim=1)  # [B, M]
        matches: list[tuple[Tensor, Tensor]] = []

        for b in range(B):
            valid_mask = exist_any[b]
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
            if valid_indices.numel() == 0 or Q == 0:
                empty = (
                    pose_pred.new_zeros((0,), dtype=torch.long),
                    pose_pred.new_zeros((0,), dtype=torch.long),
                )
                matches.append(empty)
                continue

            pose_gt_b = pose_gt[b][:, valid_indices, :, :].permute(1, 0, 2, 3)
            exist_mask = exist_gt[b][:, valid_indices].permute(1, 0)  # [M_valid, T]
            pose_pred_b = pose_pred[b]  # [Q, T, J, 3]

            diff = torch.abs(pose_pred_b.unsqueeze(1) - pose_gt_b.unsqueeze(0))
            mask = exist_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(diff.dtype)
            diff = diff * mask
            counts = exist_mask.sum(dim=1).clamp_min(1).to(diff.dtype) * float(
                J * 3
            )  # [M_valid]
            pose_cost = diff.sum(dim=(2, 3, 4)) / counts.unsqueeze(0)

            exist_cost_q = F.binary_cross_entropy_with_logits(
                exist_logit[b, :, 0],
                torch.ones_like(exist_logit[b, :, 0]),
                reduction="none",
            )
            exist_cost = exist_cost_q[:, None].expand(-1, pose_cost.shape[1])

            total_cost = (
                self._lambda_pose_match * pose_cost
                + self._lambda_exist_match * exist_cost
            )
            cost_np = total_cost.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

            matched_queries = torch.as_tensor(
                row_ind, dtype=torch.long, device=pose_pred.device
            )
            matched_targets = valid_indices[
                torch.as_tensor(col_ind, dtype=torch.long, device=pose_pred.device)
            ]
            matches.append((matched_queries, matched_targets))

        return matches

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

        images = self._render_debug_images_with_cameras(
            batch, outputs, render_pose2d_frame
        )
        if images != (None, None):
            return images
        return self._render_debug_images_naive(batch, outputs, render_pose2d_frame)

    def _render_debug_images_with_cameras(
        self,
        batch: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
        render_pose2d_frame: Any,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Render overlays by reprojecting 3D predictions with camera metadata."""
        # v3と同様の実装（outputs["pose_3d"]を使用）
        required = {"camera_C", "camera_R", "camera_intr", "image_size"}
        if not required.issubset(batch.keys()):
            return None, None
        keypoints_2d = batch.get("keypoints_2d")
        player_mask = batch.get("player_mask")
        court_2d = batch.get("court_2d")
        pose_pred = outputs.get("pose_3d")  # v2ではglobal_poseがpose_3dキーで出力
        camera_C = batch.get("camera_C")
        camera_R = batch.get("camera_R")
        camera_intr = batch.get("camera_intr")
        image_size = batch.get("image_size")
        if (
            keypoints_2d is None
            or player_mask is None
            or court_2d is None
            or pose_pred is None
            or camera_C is None
            or camera_R is None
            or camera_intr is None
            or image_size is None
        ):
            return None, None
        if keypoints_2d.ndim != 6 or player_mask.ndim != 4 or pose_pred.ndim != 5:
            return None, None
        B, T, V, M, J, _ = keypoints_2d.shape
        if B == 0 or T == 0 or V == 0:
            return None, None

        b_idx = 0
        t_idx = 0
        v_idx = 0
        size_tensor = image_size[b_idx, v_idx]
        width = int(size_tensor[0].item())
        height = int(size_tensor[1].item())
        if width <= 0 or height <= 0:
            return None, None

        def _select_cam(tensor: Tensor) -> Tensor:
            if tensor.ndim == 3:
                return tensor[b_idx, v_idx]
            if tensor.ndim == 4:
                return tensor[b_idx, v_idx]
            return tensor[v_idx]

        cam_C = _select_cam(camera_C).to(device=pose_pred.device, dtype=pose_pred.dtype)
        cam_R = _select_cam(camera_R).to(device=pose_pred.device, dtype=pose_pred.dtype)
        cam_intr = _select_cam(camera_intr).to(
            device=pose_pred.device, dtype=pose_pred.dtype
        )

        court = court_2d[b_idx, v_idx] if court_2d.ndim == 4 else court_2d[v_idx]
        court_px = self._norm_to_px(court, width, height)
        court_vis = [1] * int(court_px.shape[0])

        kp = keypoints_2d[b_idx, t_idx, v_idx]
        mask = player_mask[b_idx, t_idx, v_idx]
        player_pose_list_gt: list[np.ndarray] = []
        racket_list_gt: list[np.ndarray] = []
        for m in range(M):
            if not bool(mask[m].item()):
                continue
            pts_px = self._norm_to_px(kp[m], width, height)
            player_pose_list_gt.append(pts_px[:17])
            racket_list_gt.append(pts_px[17:])

        img_gt_np = render_pose2d_frame(
            width=width,
            height=height,
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

        pose_slice = pose_pred[b_idx, :, t_idx]  # [Q, J, 3]
        pose_world = self._denorm_pose3d(pose_slice).detach()
        exist_conf = outputs.get("exist_conf")
        if exist_conf is not None and exist_conf.shape[0] > b_idx:
            exist_mask = exist_conf[b_idx, :, 0] >= self._exist_threshold
        else:
            exist_mask = torch.ones(
                pose_world.shape[0], dtype=torch.bool, device=pose_pred.device
            )

        player_pose_list_pred: list[np.ndarray] = []
        pose_vis_list: list[list[int]] = []
        racket_list_pred: list[np.ndarray] = []
        racket_vis_list: list[list[int]] = []
        for q in range(pose_world.shape[0]):
            if not bool(exist_mask[q].item()):
                continue
            uv, vis = self._project_world_points(cam_C, cam_R, cam_intr, pose_world[q])
            uv_np = uv.detach().float().cpu().numpy().astype("float32")
            vis_np = vis.detach().cpu().numpy().astype("uint8")
            player_pose_list_pred.append(uv_np[:17])
            racket_list_pred.append(uv_np[17:])
            pose_vis_list.append(vis_np[:17].tolist())
            racket_vis_list.append(vis_np[17:].tolist())

        img_pred_np = render_pose2d_frame(
            width=width,
            height=height,
            court_points=court_px,
            court_visibility=court_vis,
            player_poses=player_pose_list_pred,
            player_pose_visibility=pose_vis_list if pose_vis_list else None,
            racket_points=racket_list_pred,
            racket_visibility=racket_vis_list if racket_vis_list else None,
        )
        img_pred = (
            torch.from_numpy(img_pred_np)
            .permute(2, 0, 1)
            .to(device=keypoints_2d.device, dtype=torch.float32)
            / 255.0
        )

        return img_gt, img_pred

    def _render_debug_images_naive(
        self,
        batch: Mapping[str, Tensor],
        outputs: Mapping[str, Tensor],
        render_pose2d_frame: Any,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Fallback visualization when camera parameters are unavailable."""
        keypoints_2d = batch.get("keypoints_2d")
        player_mask = batch.get("player_mask")
        court_2d = batch.get("court_2d")
        pose_pred = outputs.get("pose_3d")  # v2ではglobal_pose
        if (
            keypoints_2d is None
            or player_mask is None
            or court_2d is None
            or pose_pred is None
        ):
            return None, None
        if keypoints_2d.ndim != 6 or pose_pred.ndim != 5 or court_2d.ndim != 4:
            return None, None
        B, T, V, M, _, _ = keypoints_2d.shape
        if B == 0 or T == 0 or V == 0 or M == 0:
            return None, None

        H, W = 288, 512
        kp = keypoints_2d[0, 0, 0]
        mask = player_mask[0, 0, 0]
        court = court_2d[0, 0]

        court_px = self._norm_to_px(court, W, H)
        player_pose_list_gt: list[np.ndarray] = []
        racket_list_gt: list[np.ndarray] = []
        for m in range(M):
            if not bool(mask[m].item()):
                continue
            pts_px = self._norm_to_px(kp[m], W, H)
            player_pose_list_gt.append(pts_px[:17])
            racket_list_gt.append(pts_px[17:])

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

        pose_slice = pose_pred[0, :, 0]
        player_pose_list_pred: list[np.ndarray] = []
        racket_list_pred: list[np.ndarray] = []
        for pts3d in pose_slice:
            coords = pts3d[:, :2].detach().float().cpu().numpy().astype("float32")
            coords_px = np.empty_like(coords)
            coords_px[..., 0] = (coords[..., 0] * 0.1 + 0.5) * float(W - 1)
            coords_px[..., 1] = (coords[..., 1] * 0.1 + 0.5) * float(H - 1)
            player_pose_list_pred.append(coords_px[:17])
            racket_list_pred.append(coords_px[17:])

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

    @staticmethod
    def _norm_to_px(coords: Tensor, width: int, height: int) -> np.ndarray:
        """Convert normalized [-1,1] coordinates to pixel space."""
        coords_arr = coords.detach().float().cpu().numpy().astype("float32")
        out = np.empty_like(coords_arr)
        w_span = max(width - 1, 1)
        h_span = max(height - 1, 1)
        out[..., 0] = (coords_arr[..., 0] + 1.0) * 0.5 * float(w_span)
        out[..., 1] = (coords_arr[..., 1] + 1.0) * 0.5 * float(h_span)
        return out

    @staticmethod
    def _denorm_pose3d(pose_norm: Tensor) -> Tensor:
        """Convert normalized court coordinates back to world meters."""
        scales = pose_norm.new_tensor(
            [HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST],
            dtype=pose_norm.dtype,
        )
        return pose_norm * scales

    @staticmethod
    def _project_world_points(
        cam_C: Tensor,
        cam_R: Tensor,
        cam_intr: Tensor,
        xyz_world: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Project world coordinates into the image plane."""
        rel = xyz_world - cam_C.view(1, 3)
        Xc = rel @ cam_R.t()
        z = Xc[:, 2]
        mask = z > 1e-6
        z_safe = torch.where(mask, z, torch.ones_like(z))
        f = cam_intr[0]
        cx = cam_intr[1]
        cy = cam_intr[2]
        u = f * (Xc[:, 0] / z_safe) + cx
        v = f * (-Xc[:, 1] / z_safe) + cy
        uv = torch.stack([u, v], dim=-1)
        return uv, mask

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
