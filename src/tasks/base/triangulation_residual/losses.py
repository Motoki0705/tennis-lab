"""Metre-valued root/relative supervision and clean-camera reprojection loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.base.triangulation_residual.balanced_loss import balanced_residual_loss
from src.tasks.base.triangulation_residual.configuration import LossConfig, V2Config
from src.tasks.base.triangulation_residual.model import reconstruct_world
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.expand_as(value)
    return torch.where(expanded, value, 0).sum() / expanded.sum().clamp_min(1)


def residual_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    config: LossConfig,
    task: str,
    *,
    v2: V2Config | None = None,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    if v2 is not None:
        if v2.loss_mode == "balanced_regret":
            return balanced_residual_loss(output, batch, config, task, v2)
        if v2.loss_mode != "legacy":
            raise ValueError(f"Unknown residual loss mode: {v2.loss_mode}")
    # Keep metre reconstruction and camera math FP32 even under mixed precision.
    with torch.autocast(device_type=batch["features"].device.type, enabled=False):
        world, root, relative = reconstruct_world(
            output, batch["root_init"], batch["relative_init"], task=task
        )
        target = batch["target_world"].float()
        target_root = (
            target[:, :, [11, 12]].mean(dim=2) if task == "plcs" else target[:, :, 0]
        )
        target_relative = target - target_root[:, :, None]
        valid = batch["frame_valid"]

        def huber(pred: Tensor, gt: Tensor) -> Tensor:
            return F.smooth_l1_loss(
                pred, gt, reduction="none", beta=config.huber_delta_m
            )

        parts = {
            "root": masked_mean(huber(root, target_root), valid[..., None]),
            "world": masked_mean(huber(world, target), valid[..., None, None]),
        }
        parts["relative"] = (
            masked_mean(huber(relative, target_relative), valid[..., None, None])
            if task == "plcs"
            else world.sum() * 0
        )
        homogeneous = torch.cat((world, torch.ones_like(world[..., :1])), dim=-1)
        projected = torch.einsum(
            "bvij,btkj->bvtki", batch["true_projection"].float(), homogeneous
        )
        depth = projected[..., 2:3]
        projected_uv = projected[..., :2] / depth.clamp_min(0.05)
        reprojection = F.smooth_l1_loss(
            projected_uv, batch["clean_uv"].float(), reduction="none", beta=0.01
        )
        parts["reprojection"] = masked_mean(
            reprojection, batch["clean_visible"][..., None]
        )
        fps = batch["fps"].reshape(-1, 1, 1, 1).float()
        velocity_error = F.smooth_l1_loss(
            torch.diff(world, dim=1) * fps,
            torch.diff(target, dim=1) * fps,
            reduction="none",
            beta=1.0,
        )
        parts["velocity"] = masked_mean(
            velocity_error, (valid[:, 1:] & valid[:, :-1])[..., None, None]
        )
        if task == "plcs":
            a, b = zip(*COCO17_BONE_LENGTH_EDGES, strict=True)
            pred_length = (world[:, :, a] - world[:, :, b]).norm(dim=-1)
            gt_length = (target[:, :, a] - target[:, :, b]).norm(dim=-1)
            parts["bone"] = masked_mean(huber(pred_length, gt_length), valid[..., None])
        else:
            parts["bone"] = world.sum() * 0
        total = sum(
            getattr(config, f"{name}_weight") * value for name, value in parts.items()
        )
        if not isinstance(total, Tensor):
            raise RuntimeError("Empty loss")
    return total, parts, world


def metric_arrays(
    world: Tensor, batch: dict[str, Tensor], task: str
) -> dict[str, Tensor]:
    target = batch["target_world"].float()
    initial = batch["init_world"].float()
    root_joints = (11, 12) if task == "plcs" else (0,)
    pred_root, gt_root, init_root = (
        x[:, :, root_joints].mean(dim=2) for x in (world, target, initial)
    )
    return {
        "world_mpjpe_m": (world - target).norm(dim=-1),
        "initial_world_mpjpe_m": (initial - target).norm(dim=-1),
        "root_error_m": (pred_root - gt_root).norm(dim=-1),
        "initial_root_error_m": (init_root - gt_root).norm(dim=-1),
        "relative_mpjpe_m": (
            (world - pred_root[:, :, None]) - (target - gt_root[:, :, None])
        ).norm(dim=-1),
        "initial_relative_mpjpe_m": (
            (initial - init_root[:, :, None]) - (target - gt_root[:, :, None])
        ).norm(dim=-1),
    }
