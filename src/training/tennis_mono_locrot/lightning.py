"""LightningModule wrapper for the monocular location+rotation task."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict
from typing import Any, cast

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.models.tennis_mono_locrot import TennisMonoLocRotConfig, TennisMonoLocRotModel
from src.training.base.tennis_multi_cam_3d_pose import BaseTennisLightningModule
from src.training.utils.tennis_projection import denorm_pose3d, project_world_points


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def _build_model_cfg(model_cfg: Mapping[str, Any] | None) -> TennisMonoLocRotConfig:
    base = TennisMonoLocRotConfig()
    if model_cfg is None:
        return base
    merged = {**asdict(base), **dict(model_cfg)}
    return TennisMonoLocRotConfig(**merged)


class TennisMonoLocRotModule(BaseTennisLightningModule):
    """Configure training for the tennis_mono_locrot task.

    Args:
        cfg (DictConfig): Hydra/OmegaConf configuration tree that defines the
            model, optimizer, logging, and denoising settings.

    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        model_cfg = _to_dict(cfg.get("model"))
        self.model_cfg = _build_model_cfg(model_cfg)
        self.model = TennisMonoLocRotModel(self.model_cfg)

        training_cfg = _to_dict(cfg.get("training", {}))
        optim_cfg = training_cfg.get("optimizer", {})
        self._lr = float(optim_cfg.get("lr", 1.0e-4))
        self._weight_decay = float(optim_cfg.get("weight_decay", 1.0e-4))
        self._max_steps = int(training_cfg.get("max_steps", 0))
        self._scheduler_cfg = _to_dict(training_cfg.get("scheduler"))

        loss_cfg = training_cfg.get("loss", {})
        self._lambda_pose2d = float(loss_cfg.get("lambda_pose2d", 1.0))
        self._lambda_root_trans = float(loss_cfg.get("lambda_root_trans", 1.0))
        self._lambda_root_rot = float(loss_cfg.get("lambda_root_rot", 1.0))
        self._lambda_denoise_root_trans = float(
            loss_cfg.get("lambda_denoise_root_trans", 0.0)
        )
        self._lambda_denoise_root_rot = float(
            loss_cfg.get("lambda_denoise_root_rot", 0.0)
        )
        self._lambda_exist = float(loss_cfg.get("lambda_exist", 1.0))
        # Matching coefficients used by BaseTennisLightningModule._match_queries_to_targets.
        self._lambda_pose_match = float(loss_cfg.get("lambda_pose_match", 1.0))
        self._lambda_exist_match = float(loss_cfg.get("lambda_exist_match", 1.0))

        denoise_cfg = _to_dict(training_cfg.get("denoise", {}))
        self._denoise_root_trans_noise_std = float(
            denoise_cfg.get("root_trans_noise_std", 0.0)
        )
        self._denoise_root_rot_noise_std = math.radians(
            float(denoise_cfg.get("root_rot_noise_deg", 0.0))
        )
        self._denoise_num_samples = max(1, int(denoise_cfg.get("num_samples", 1)))

        input_mask_cfg = _to_dict(training_cfg.get("input_mask", {}))
        self._drop_joint_prob = float(input_mask_cfg.get("drop_joint_prob", 0.0))
        self._drop_court_prob = float(input_mask_cfg.get("drop_court_prob", 0.0))

        camera_cfg = _to_dict(training_cfg.get("camera_schedule"))
        self._camera_schedule: dict[str, int] | None = None
        if camera_cfg:
            max_cam = int(camera_cfg.get("max_cameras", self.model_cfg.max_cameras))
            min_cam = int(camera_cfg.get("min_cameras", 1))
            transition = max(1, int(camera_cfg.get("transition_steps", 1)))
            self._camera_schedule = {
                "start": max_cam,
                "target": min_cam,
                "transition": transition,
            }

        viz_cfg = _to_dict(cfg.get("logging", {})).get("visualizer", {})
        self._viz_max_batches = int(viz_cfg.get("max_batches", 0))
        self._exist_threshold = float(viz_cfg.get("exist_threshold", 0.5))

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def forward(self, batch: Mapping[str, Tensor]) -> MutableMapping[str, Tensor]:
        """Run the underlying model on the middle timestep of the sequence.

        Args:
            batch (Mapping[str, Tensor]): Mini-batch produced by the datamodule.

        Returns:
            MutableMapping[str, Tensor]: Model predictions with auxiliary tensors
            when denoising is enabled.

        Raises:
            ValueError: If any required tensor has an unexpected shape.

        """
        keypoints_2d = batch["keypoints_2d"]
        player_mask = batch["player_mask"]
        court_2d = batch["court_2d"]

        if keypoints_2d.ndim != 6:
            msg = "keypoints_2d must have shape [B, T, V, M, J, 2]"
            raise ValueError(msg)
        if player_mask.ndim != 4:
            msg = "player_mask must have shape [B, T, V, M]"
            raise ValueError(msg)

        B, T, V, M, J, _ = keypoints_2d.shape
        if T <= 0:
            msg = "sequence length T must be positive"
            raise ValueError(msg)
        if self.model_cfg.num_joints != J:
            msg = "unexpected num_joints in keypoints_2d"
            raise ValueError(msg)

        t_idx = T // 2
        player_kpts_2d = keypoints_2d[:, t_idx]
        player_mask_frame = player_mask[:, t_idx]

        if court_2d.ndim == 4:
            court_kpts_2d = court_2d
        elif court_2d.ndim == 3:
            court_kpts_2d = court_2d.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            msg = "court_2d must have shape [B, V, C, 2] or [V, C, 2]"
            raise ValueError(msg)

        if court_kpts_2d.shape[0] != B or court_kpts_2d.shape[1] != V:
            msg = "court_2d batch or view dimension mismatch"
            raise ValueError(msg)

        if self.training:
            (
                player_kpts_2d,
                player_mask_frame,
                court_kpts_2d,
            ) = self._apply_input_mask(player_kpts_2d, player_mask_frame, court_kpts_2d)

        denoise_inputs = None
        denoise_targets: Mapping[str, Tensor] | None = None
        if self.training and self._should_denoise():
            payload = self._build_denoise_payload(batch, t_idx)
            if payload is not None:
                denoise_inputs, denoise_targets = payload

        outputs = cast(
            dict[str, Tensor],
            self.model(
                player_kpts_2d=player_kpts_2d,
                player_mask=player_mask_frame,
                court_kpts_2d=court_kpts_2d,
                denoise_inputs=denoise_inputs,
            ),
        )

        if denoise_targets is not None:
            outputs["denoise_target_root_trans"] = denoise_targets["root_trans"]
            outputs["denoise_target_root_rot"] = denoise_targets["root_rot"]

        return outputs

    def _apply_input_mask(
        self,
        player_kpts_2d: Tensor,
        player_mask_frame: Tensor,
        court_kpts_2d: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self._drop_joint_prob > 0.0:
            B, V, M, J, _ = player_kpts_2d.shape
            device = player_kpts_2d.device
            drop_joints = torch.rand(B, V, M, J, device=device) < self._drop_joint_prob
            drop_joints = drop_joints & player_mask_frame.unsqueeze(-1)
            if drop_joints.any():
                drop_joints_exp = drop_joints.unsqueeze(-1)
                player_kpts_2d = player_kpts_2d.masked_fill(drop_joints_exp, 0.0)

        if self._drop_court_prob > 0.0:
            B_c, V_c, C_c, _ = court_kpts_2d.shape
            device = court_kpts_2d.device
            drop_court = (
                torch.rand(B_c, V_c, C_c, device=device) < self._drop_court_prob
            )
            if drop_court.any():
                drop_court_exp = drop_court.unsqueeze(-1)
                court_kpts_2d = court_kpts_2d.masked_fill(drop_court_exp, 0.0)

        return player_kpts_2d, player_mask_frame, court_kpts_2d

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        """Execute one optimization step.

        Args:
            batch (Mapping[str, Tensor]): Training mini-batch.
            batch_idx (int): Index of the batch within the epoch.

        Returns:
            Tensor: Total loss used for backpropagation.

        """
        outputs = self.forward(batch)
        loss_dict = self._compute_loss(outputs, batch)
        self._attach_logging_pose(outputs, batch)
        self._log_losses(loss_dict, stage="train")
        return loss_dict["total"]

    def validation_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> None:
        """Evaluate the model on a validation batch and render debug images.

        Args:
            batch (Mapping[str, Tensor]): Validation mini-batch.
            batch_idx (int): Index of the batch within the validation loop.

        Returns:
            None: PyTorch Lightning handles the logged metrics internally.

        """
        outputs = self.forward(batch)
        loss_dict = self._compute_loss(outputs, batch)
        self._attach_logging_pose(outputs, batch)
        self._log_losses(loss_dict, stage="val")
        viz_max_batches = int(getattr(self, "_viz_max_batches", 0))
        if batch_idx < viz_max_batches:
            image_gt, image_pred = self._render_debug_images(batch, outputs)
            step = int(self.global_step)
            if image_gt is not None:
                self._log_tensorboard_image("val/pose2d_gt", image_gt, step)
            if image_pred is not None:
                self._log_tensorboard_image("val/pose2d_pred_reproj", image_pred, step)

    def _compute_loss(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        (
            root_trans_pred,
            root_rot_pred,
            exist_logit,
            root_trans_frame,
            root_rot_frame,
            exist_frame,
            keypoints_frame,
            camera_C,
            camera_R,
            camera_intr,
        ) = self._extract_loss_inputs(outputs, batch)

        # ---- Project 3D roots (pred/GT) into each camera to obtain 2D poses ----
        # Treat camera views as the temporal dimension T for the matching utility.
        (
            matches,
            pose2d_loss,
            root_trans_loss,
            root_rot_loss,
            exist_loss,
        ) = self._compute_pose_root_exist_losses(
            root_trans_pred=root_trans_pred,
            root_rot_pred=root_rot_pred,
            exist_logit=exist_logit,
            root_trans_frame=root_trans_frame,
            root_rot_frame=root_rot_frame,
            exist_frame=exist_frame,
            keypoints_frame=keypoints_frame,
            camera_C=camera_C,
            camera_R=camera_R,
            camera_intr=camera_intr,
        )

        if isinstance(outputs, MutableMapping):
            outputs["_root_matches"] = matches  # type: ignore[index]

        (
            denoise_root_trans_loss,
            denoise_root_rot_loss,
        ) = self._compute_denoise_losses(outputs, root_trans_pred, root_rot_pred)

        total = (
            self._lambda_pose2d * pose2d_loss
            + self._lambda_root_trans * root_trans_loss
            + self._lambda_root_rot * root_rot_loss
            + self._lambda_exist * exist_loss
            + self._lambda_denoise_root_trans * denoise_root_trans_loss
            + self._lambda_denoise_root_rot * denoise_root_rot_loss
        )

        return {
            "total": total,
            "pose2d_l1": pose2d_loss.detach(),
            "root_trans_l1": root_trans_loss.detach(),
            "root_rot_l1": root_rot_loss.detach(),
            "exist_bce": exist_loss.detach(),
            "denoise_root_trans_l1": denoise_root_trans_loss.detach(),
            "denoise_root_rot_l1": denoise_root_rot_loss.detach(),
        }

    def _extract_loss_inputs(
        self,
        outputs: Mapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor | None,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        root_trans_pred = outputs["root_trans"]
        root_rot_pred = outputs["root_rot"]
        exist_logit = outputs.get("exist_logit")

        root_trans_gt = batch["root_trans_gt"]
        root_rot_gt = batch["root_rot_gt"]
        exist_gt = batch["exist_3d_gt"]
        keypoints_2d = batch["keypoints_2d"]

        camera_C = batch.get("camera_C")
        camera_R = batch.get("camera_R")
        camera_intr = batch.get("camera_intr")

        if camera_C is None or camera_R is None or camera_intr is None:
            msg = "camera_C, camera_R, and camera_intr must be present in the batch for pose2d loss"
            raise ValueError(msg)

        if (
            root_trans_gt.ndim != 4
            or root_rot_gt.ndim != 4
            or exist_gt.ndim != 3
            or keypoints_2d.ndim != 6
        ):
            msg = "unexpected tensor shapes for ground-truth or keypoints_2d"
            raise ValueError(msg)

        B, T_gt, M_gt, _ = root_trans_gt.shape
        B_kp, T_kp, V, M_kp, _, _ = keypoints_2d.shape
        if B_kp != B or M_kp != M_gt or T_kp != T_gt:
            msg = "batch/keypoints/root GT shapes are inconsistent"
            raise ValueError(msg)
        if T_gt <= 0:
            msg = "T_gt must be positive"
            raise ValueError(msg)

        t_idx = T_gt // 2

        root_trans_frame = root_trans_gt[:, t_idx]  # [B, M, 3]
        root_rot_frame = root_rot_gt[:, t_idx]  # [B, M, 2]
        exist_frame = exist_gt[:, t_idx]  # [B, M]

        if root_trans_pred.ndim != 3 or root_rot_pred.ndim != 3:
            msg = "unexpected prediction tensor shapes"
            raise ValueError(msg)
        if root_trans_pred.shape[0] != B or root_rot_pred.shape[0] != B:
            msg = "batch size of predictions and ground truth do not match"
            raise ValueError(msg)

        keypoints_frame = keypoints_2d[:, t_idx]  # [B, V, M, J, 2]

        return (
            root_trans_pred,
            root_rot_pred,
            exist_logit,
            root_trans_frame,
            root_rot_frame,
            exist_frame,
            keypoints_frame,
            camera_C,
            camera_R,
            camera_intr,
        )

    def _compute_pose_root_exist_losses(
        self,
        root_trans_pred: Tensor,
        root_rot_pred: Tensor,
        exist_logit: Tensor | None,
        root_trans_frame: Tensor,
        root_rot_frame: Tensor,
        exist_frame: Tensor,
        keypoints_frame: Tensor,
        camera_C: Tensor,
        camera_R: Tensor,
        camera_intr: Tensor,
    ) -> tuple[
        list[tuple[Tensor, Tensor]],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        B, T_v, M_gt, _, _ = keypoints_frame.shape
        B_pred, Q, _ = root_trans_pred.shape

        root_trans_pred_world = denorm_pose3d(root_trans_pred)  # [B, Q, 3]
        root_trans_gt_world = denorm_pose3d(root_trans_frame)  # [B, M, 3]

        pose2d_pred = root_trans_pred.new_zeros(B, Q, T_v, 1, 2)
        pose2d_gt = root_trans_pred.new_zeros(B, T_v, M_gt, 1, 2)

        def _select_cam_param(tensor: Tensor, b: int, v: int) -> Tensor:
            if tensor.ndim == 4 or tensor.ndim == 3:
                return tensor[b, v]
            return tensor[v]

        for b in range(B):
            xyz_pred = root_trans_pred_world[b]  # [Q, 3]
            xyz_gt = root_trans_gt_world[b]  # [M, 3]
            for v in range(T_v):
                cam_C_bv = _select_cam_param(camera_C, b, v).to(
                    device=xyz_pred.device,
                    dtype=xyz_pred.dtype,
                )
                cam_R_bv = _select_cam_param(camera_R, b, v).to(
                    device=xyz_pred.device,
                    dtype=xyz_pred.dtype,
                )
                cam_intr_bv = _select_cam_param(camera_intr, b, v).to(
                    device=xyz_pred.device,
                    dtype=xyz_pred.dtype,
                )

                uv_pred, _ = project_world_points(cam_C_bv, cam_R_bv, cam_intr_bv, xyz_pred)
                uv_gt, _ = project_world_points(cam_C_bv, cam_R_bv, cam_intr_bv, xyz_gt)

                pose2d_pred[b, :, v, 0, :] = uv_pred
                pose2d_gt[b, v, :, 0, :] = uv_gt

        exist_gt_match = exist_frame.unsqueeze(1).expand(-1, T_v, -1)

        if exist_logit is None:
            exist_logit_match = root_trans_pred.new_zeros(B_pred, Q, 1)
        else:
            exist_logit_match = exist_logit

        matches = self._match_queries_to_targets(
            pose_pred=pose2d_pred,
            pose_gt=pose2d_gt,
            exist_gt=exist_gt_match,
            exist_logit=exist_logit_match,
        )

        pose2d_loss_num = root_trans_pred.new_tensor(0.0)
        pose2d_loss_den = root_trans_pred.new_tensor(0.0)
        root_trans_loss_num = root_trans_pred.new_tensor(0.0)
        root_trans_loss_den = root_trans_pred.new_tensor(0.0)
        root_rot_loss_num = root_rot_pred.new_tensor(0.0)
        root_rot_loss_den = root_rot_pred.new_tensor(0.0)

        exist_loss = root_trans_pred.new_tensor(0.0)
        exist_target = None
        if exist_logit is not None:
            exist_target = torch.zeros_like(exist_logit_match)

        for b, (matched_q, matched_t) in enumerate(matches):
            if matched_q.numel() == 0:
                continue

            if exist_target is not None:
                exist_target[b, matched_q, 0] = 1.0

            pose2d_pred_sel = pose2d_pred[b, matched_q]  # [N, T_v, 1, 2]
            pose2d_gt_sel = pose2d_gt[b, :, matched_t].permute(1, 0, 2, 3)
            exist_mask_sel = exist_gt_match[b, :, matched_t].permute(1, 0)  # [N, T_v]
            mask_2d = exist_mask_sel.unsqueeze(-1).unsqueeze(-1).to(
                dtype=pose2d_pred_sel.dtype
            )
            diff_2d = torch.abs(pose2d_pred_sel - pose2d_gt_sel) * mask_2d
            pose2d_loss_num = pose2d_loss_num + diff_2d.sum()
            pose2d_loss_den = pose2d_loss_den + mask_2d.sum() * float(2)

            root_trans_pred_sel = root_trans_pred[b, matched_q]
            root_trans_gt_sel = root_trans_frame[b, matched_t]
            mask_trans = exist_frame[b, matched_t].unsqueeze(-1).to(
                dtype=root_trans_pred.dtype
            )
            diff_trans = torch.abs(root_trans_pred_sel - root_trans_gt_sel) * mask_trans
            root_trans_loss_num = root_trans_loss_num + diff_trans.sum()
            root_trans_loss_den = root_trans_loss_den + mask_trans.sum() * float(3)

            root_rot_pred_sel = root_rot_pred[b, matched_q]
            root_rot_gt_sel = root_rot_frame[b, matched_t]
            mask_rot = exist_frame[b, matched_t].unsqueeze(-1).to(
                dtype=root_rot_pred.dtype
            )
            diff_rot = torch.abs(root_rot_pred_sel - root_rot_gt_sel) * mask_rot
            root_rot_loss_num = root_rot_loss_num + diff_rot.sum()
            root_rot_loss_den = root_rot_loss_den + mask_rot.sum() * float(2)

        if pose2d_loss_den.item() > 0:
            pose2d_loss = pose2d_loss_num / pose2d_loss_den
        else:
            pose2d_loss = root_trans_pred.new_tensor(0.0)

        if root_trans_loss_den.item() > 0:
            root_trans_loss = root_trans_loss_num / root_trans_loss_den
        else:
            root_trans_loss = root_trans_pred.new_tensor(0.0)

        if root_rot_loss_den.item() > 0:
            root_rot_loss = root_rot_loss_num / root_rot_loss_den
        else:
            root_rot_loss = root_rot_pred.new_tensor(0.0)

        if exist_target is not None:
            exist_loss = F.binary_cross_entropy_with_logits(exist_logit_match, exist_target)

        return matches, pose2d_loss, root_trans_loss, root_rot_loss, exist_loss

    def _compute_denoise_losses(
        self,
        outputs: Mapping[str, Tensor],
        root_trans_pred: Tensor,
        root_rot_pred: Tensor,
    ) -> tuple[Tensor, Tensor]:
        denoise_root_trans_loss = root_trans_pred.new_tensor(0.0)
        denoise_root_rot_loss = root_rot_pred.new_tensor(0.0)

        denoise_mask = outputs.get("denoise_mask")
        if (
            self._lambda_denoise_root_trans > 0.0
            and denoise_mask is not None
            and outputs.get("denoise_root_trans") is not None
            and outputs.get("denoise_target_root_trans") is not None
        ):
            pred_trans = cast(Tensor, outputs["denoise_root_trans"])
            target_trans = cast(Tensor, outputs["denoise_target_root_trans"])
            mask = cast(Tensor, denoise_mask).unsqueeze(-1).to(dtype=pred_trans.dtype)
            diff = torch.abs(pred_trans - target_trans) * mask
            denom = mask.sum() * float(3)
            if denom.item() > 0:
                denoise_root_trans_loss = diff.sum() / denom

        if (
            self._lambda_denoise_root_rot > 0.0
            and denoise_mask is not None
            and outputs.get("denoise_root_rot") is not None
            and outputs.get("denoise_target_root_rot") is not None
        ):
            pred_rot = cast(Tensor, outputs["denoise_root_rot"])
            target_rot = cast(Tensor, outputs["denoise_target_root_rot"])
            mask = cast(Tensor, denoise_mask).unsqueeze(-1).to(dtype=pred_rot.dtype)
            diff = torch.abs(pred_rot - target_rot) * mask
            denom = mask.sum() * float(2)
            if denom.item() > 0:
                denoise_root_rot_loss = diff.sum() / denom

        return denoise_root_trans_loss, denoise_root_rot_loss

    def on_train_batch_start(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> None:  # pragma: no cover - scheduling side-effect
        """Update the camera curriculum before each training batch.

        Args:
            batch (Mapping[str, Tensor]): Unused Lightning batch placeholder.
            batch_idx (int): Batch index provided by Lightning.

        Returns:
            None: This hook performs side effects only.

        """
        super().on_train_batch_start(batch, batch_idx)
        self._maybe_update_camera_schedule()

    def _should_denoise(self) -> bool:
        return (
            self._lambda_denoise_root_trans > 0.0
            and self._denoise_root_trans_noise_std > 0.0
        ) or (
            self._lambda_denoise_root_rot > 0.0
            and self._denoise_root_rot_noise_std > 0.0
        )

    def _build_denoise_payload(
        self,
        batch: Mapping[str, Tensor],
        t_idx: int,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]] | None:
        root_trans_gt = batch.get("root_trans_gt")
        root_rot_gt = batch.get("root_rot_gt")
        exist_gt = batch.get("exist_3d_gt")
        if root_trans_gt is None or root_rot_gt is None or exist_gt is None:
            return None

        root_trans_frame = root_trans_gt[:, t_idx]
        root_rot_frame = root_rot_gt[:, t_idx]
        exist_frame = exist_gt[:, t_idx].to(dtype=torch.bool)
        if not torch.any(exist_frame):
            return None

        trans_repeat = root_trans_frame
        rot_repeat = root_rot_frame
        mask_repeat = exist_frame
        if self._denoise_num_samples > 1:
            trans_repeat = trans_repeat.unsqueeze(2).repeat(
                1, 1, self._denoise_num_samples, 1
            )
            rot_repeat = rot_repeat.unsqueeze(2).repeat(
                1, 1, self._denoise_num_samples, 1
            )
            mask_repeat = mask_repeat.unsqueeze(2).repeat(
                1, 1, self._denoise_num_samples
            )
            trans_repeat = trans_repeat.reshape(root_trans_frame.shape[0], -1, 3)
            rot_repeat = rot_repeat.reshape(root_rot_frame.shape[0], -1, 2)
            mask_repeat = mask_repeat.reshape(root_rot_frame.shape[0], -1)

        noise_mask = mask_repeat

        noisy_trans = trans_repeat
        if self._denoise_root_trans_noise_std > 0.0:
            noisy_trans = noisy_trans + torch.randn_like(noisy_trans).mul(
                self._denoise_root_trans_noise_std
            )

        angles = torch.atan2(rot_repeat[..., 1], rot_repeat[..., 0])
        if self._denoise_root_rot_noise_std > 0.0:
            angles = angles + torch.randn_like(angles).mul(
                self._denoise_root_rot_noise_std
            )
        noisy_rot = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        noisy_root = torch.cat([noisy_trans, noisy_rot], dim=-1)

        model_inputs = {"noisy_root": noisy_root, "mask": noise_mask}
        targets = {
            "root_trans": trans_repeat,
            "root_rot": rot_repeat,
            "mask": noise_mask,
        }
        return model_inputs, targets

    def _attach_logging_pose(
        self,
        outputs: MutableMapping[str, Tensor],
        batch: Mapping[str, Tensor],
    ) -> None:
        if "pose_3d" in outputs:
            return
        canonical_gt = batch.get("canonical_pose_gt")
        exist_gt = batch.get("exist_3d_gt")
        if canonical_gt is None or exist_gt is None:
            return

        root_trans = outputs.get("root_trans")
        root_rot = outputs.get("root_rot")
        exist_logit = outputs.get("exist_logit")
        matches = outputs.get("_root_matches")

        if root_trans is None or root_rot is None:
            return

        t_idx = canonical_gt.shape[1] // 2
        canonical_frame = canonical_gt[:, t_idx]
        exist_frame = exist_gt[:, t_idx]

        root_trans_for_pose = root_trans
        root_rot_for_pose = root_rot
        exist_conf: Tensor | None = None

        # If we have Hungarian matches, align predictions to GT player indices.
        if isinstance(matches, list) and len(matches) == root_trans.shape[0]:
            B, M, _ = root_trans.shape
            root_trans_aligned = root_trans.new_zeros(B, M, 3)
            root_rot_aligned = root_rot.new_zeros(B, M, 2)
            exist_conf_aligned = None
            if exist_logit is not None:
                exist_conf_aligned = torch.zeros_like(exist_logit)

            for b, pair in enumerate(matches):
                matched_q, matched_t = pair
                if matched_q.numel() == 0:
                    continue
                root_trans_aligned[b, matched_t] = root_trans[b, matched_q]
                root_rot_aligned[b, matched_t] = root_rot[b, matched_q]
                if exist_logit is not None and exist_conf_aligned is not None:
                    exist_conf_aligned[b, matched_t, 0] = torch.sigmoid(
                        exist_logit[b, matched_q, 0]
                    )

            root_trans_for_pose = root_trans_aligned
            root_rot_for_pose = root_rot_aligned
            if exist_logit is not None and exist_conf_aligned is not None:
                exist_conf = exist_conf_aligned

        pose = self._compose_pose(canonical_frame, root_trans_for_pose, root_rot_for_pose)
        pose = pose.unsqueeze(2)  # [B, M, 1, J, 3]
        outputs["pose_3d"] = pose

        if exist_conf is not None:
            outputs["exist_conf"] = exist_conf.to(dtype=pose.dtype)
        else:
            outputs["exist_conf"] = exist_frame.unsqueeze(-1).to(dtype=pose.dtype)

    def _compose_pose(
        self,
        canonical_frame: Tensor,
        root_trans: Tensor,
        root_rot: Tensor,
    ) -> Tensor:
        cos_theta = root_rot[..., 0:1]
        sin_theta = root_rot[..., 1:2]
        x_c = canonical_frame[..., 0]
        y_c = canonical_frame[..., 1]
        z_c = canonical_frame[..., 2]
        x_r = x_c * cos_theta - y_c * sin_theta
        y_r = x_c * sin_theta + y_c * cos_theta
        rotated = torch.stack([x_r, y_r, z_c], dim=-1)
        return rotated + root_trans.unsqueeze(-2)

    def _maybe_update_camera_schedule(self) -> None:
        if self._camera_schedule is None or self.trainer is None:
            return
        schedule = self._camera_schedule
        progress = min(
            max(float(self.global_step) / float(schedule["transition"]), 0.0), 1.0
        )
        start = schedule["start"]
        target = schedule["target"]
        current = start - (start - target) * progress
        current = int(max(target, min(start, round(current))))
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return
        setter = getattr(datamodule, "set_camera_schedule", None)
        if setter is None:
            return
        setter(max_cameras=current, min_cameras=current)

    def _log_losses(self, loss_dict: Mapping[str, Tensor], stage: str) -> None:
        for key, value in loss_dict.items():
            tag = f"{stage}/{key}"
            self.log(tag, value, prog_bar=(key == "total"), sync_dist=False)
