"""Metre-valued root/relative supervision and clean-camera reprojection loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.plcs.configuration_contracts import ResidualLossConfig
from src.tasks.plcs.models.triangulation_residual import reconstruct_world
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES


def masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    expanded = mask.expand_as(value)
    return torch.where(expanded, value, 0).sum() / expanded.sum().clamp_min(1)


def residual_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    config: ResidualLossConfig,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    # Keep metre reconstruction and camera math FP32 even under mixed precision.
    with torch.autocast(device_type=batch["features"].device.type, enabled=False):
        world, root, relative = reconstruct_world(
            output, batch["root_init"], batch["relative_init"]
        )
        target = batch["target_world"].float()
        target_root = target[:, :, [11, 12]].mean(dim=2)
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
        parts["relative"] = masked_mean(
            huber(relative, target_relative), valid[..., None, None]
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
        a, b = zip(*COCO17_BONE_LENGTH_EDGES, strict=True)
        pred_length = (world[:, :, a] - world[:, :, b]).norm(dim=-1)
        gt_length = (target[:, :, a] - target[:, :, b]).norm(dim=-1)
        parts["bone"] = masked_mean(huber(pred_length, gt_length), valid[..., None])
        total = sum(
            getattr(config, f"{name}_weight") * value for name, value in parts.items()
        )
        if not isinstance(total, Tensor):
            raise RuntimeError("Empty loss")
    return total, parts, world
