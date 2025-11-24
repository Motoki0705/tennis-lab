"""LightningModule wrapper for the monocular location+rotation task."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import asdict
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.models.tennis_mono_locrot import TennisMonoLocRotConfig, TennisMonoLocRotModel
from src.training.base.tennis_multi_cam_3d_pose import BaseTennisLightningModule


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
        self._lambda_root_trans = float(loss_cfg.get("lambda_root_trans", 1.0))
        self._lambda_root_rot = float(loss_cfg.get("lambda_root_rot", 1.0))
        self._lambda_denoise_root_trans = float(
            loss_cfg.get("lambda_denoise_root_trans", 0.0)
        )
        self._lambda_denoise_root_rot = float(
            loss_cfg.get("lambda_denoise_root_rot", 0.0)
        )

        denoise_cfg = _to_dict(training_cfg.get("denoise", {}))
        self._denoise_root_trans_noise_std = float(
            denoise_cfg.get("root_trans_noise_std", 0.0)
        )
        self._denoise_root_rot_noise_std = math.radians(
            float(denoise_cfg.get("root_rot_noise_deg", 0.0))
        )
        self._denoise_num_samples = max(1, int(denoise_cfg.get("num_samples", 1)))

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

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        """Execute one optimization step.

        Args:
            batch (Mapping[str, Tensor]): Training mini-batch.
            batch_idx (int): Index of the batch within the epoch.

        Returns:
            Tensor: Total loss used for backpropagation.

        """
        outputs = self.forward(batch)
        self._attach_logging_pose(outputs, batch)
        loss_dict = self._compute_loss(outputs, batch)
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
        self._attach_logging_pose(outputs, batch)
        loss_dict = self._compute_loss(outputs, batch)
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
        root_trans_pred = outputs["root_trans"]
        root_rot_pred = outputs["root_rot"]

        root_trans_gt = batch["root_trans_gt"]
        root_rot_gt = batch["root_rot_gt"]
        exist_gt = batch["exist_3d_gt"]

        if root_trans_gt.ndim != 4 or root_rot_gt.ndim != 4 or exist_gt.ndim != 3:
            msg = "unexpected ground-truth tensor shapes"
            raise ValueError(msg)

        B, T_gt, M_gt, _ = root_trans_gt.shape
        if T_gt <= 0:
            msg = "T_gt must be positive"
            raise ValueError(msg)
        t_idx = T_gt // 2

        root_trans_frame = root_trans_gt[:, t_idx]
        root_rot_frame = root_rot_gt[:, t_idx]
        exist_frame = exist_gt[:, t_idx]

        if root_trans_pred.shape[:2] != root_trans_frame.shape[:2]:
            msg = "root_trans prediction and ground truth shapes do not match"
            raise ValueError(msg)
        if root_rot_pred.shape[:2] != root_rot_frame.shape[:2]:
            msg = "root_rot prediction and ground truth shapes do not match"
            raise ValueError(msg)

        mask_trans = exist_frame.unsqueeze(-1).to(dtype=root_trans_pred.dtype)
        diff_trans = torch.abs(root_trans_pred - root_trans_frame) * mask_trans
        denom_trans = mask_trans.sum() * float(3)
        if denom_trans.item() <= 0:
            root_trans_loss = root_trans_pred.new_tensor(0.0)
        else:
            root_trans_loss = diff_trans.sum() / denom_trans

        mask_rot = exist_frame.unsqueeze(-1).to(dtype=root_rot_pred.dtype)
        diff_rot = torch.abs(root_rot_pred - root_rot_frame) * mask_rot
        denom_rot = mask_rot.sum() * float(2)
        if denom_rot.item() <= 0:
            root_rot_loss = root_rot_pred.new_tensor(0.0)
        else:
            root_rot_loss = diff_rot.sum() / denom_rot

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

        total = (
            self._lambda_root_trans * root_trans_loss
            + self._lambda_root_rot * root_rot_loss
            + self._lambda_denoise_root_trans * denoise_root_trans_loss
            + self._lambda_denoise_root_rot * denoise_root_rot_loss
        )

        return {
            "total": total,
            "root_trans_l1": root_trans_loss.detach(),
            "root_rot_l1": root_rot_loss.detach(),
            "denoise_root_trans_l1": denoise_root_trans_loss.detach(),
            "denoise_root_rot_l1": denoise_root_rot_loss.detach(),
        }

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
        if root_trans is None or root_rot is None:
            return
        t_idx = canonical_gt.shape[1] // 2
        canonical_frame = canonical_gt[:, t_idx]
        exist_frame = exist_gt[:, t_idx]
        pose = self._compose_pose(canonical_frame, root_trans, root_rot)
        pose = pose.unsqueeze(2)  # [B, M, 1, J, 3]
        outputs["pose_3d"] = pose
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
