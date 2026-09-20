"""Sample- and severity-balanced v2 supervision with paired geometric regret."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.base.triangulation_residual.configuration import LossConfig, V2Config
from src.tasks.base.triangulation_residual.model import reconstruct_world
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES


def _balanced_mean(value: Tensor, mask: Tensor, strata: Tensor) -> Tensor:
    """Average points, then samples within each present severity stratum.

    A sample with no valid elements is absent from this particular loss term.
    Neither absent samples nor absent strata contribute a zero-valued example.
    """
    expanded = mask.expand_as(value)
    counts = expanded.flatten(1).sum(dim=1)
    means = torch.where(expanded, value, 0).flatten(1).sum(dim=1)
    means = means / counts.clamp_min(1)
    included = strata & (counts > 0)[None]
    stratum_counts = included.sum(dim=1)
    stratum_means = torch.where(included, means[None], 0).sum(dim=1)
    stratum_means = stratum_means / stratum_counts.clamp_min(1)
    return stratum_means.sum() / (stratum_counts > 0).sum().clamp_min(1)


def _severity_strata(severity: Tensor, samples: int) -> Tensor:
    severity = severity.reshape(-1)
    if severity.shape != (samples,):
        raise ValueError("Balanced loss requires one severity value per sample")
    strata = torch.stack((severity == 0, severity == 1, severity > 1))
    if not torch.isfinite(severity).all() or not strata.any(dim=0).all():
        raise ValueError("Severity must be clean (0), normal (1), or hard (>1)")
    return strata


def balanced_residual_loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor],
    config: LossConfig,
    task: str,
    v2: V2Config,
) -> tuple[Tensor, dict[str, Tensor], Tensor]:
    """Keep 3D errors in metres and compare each prediction with its own seed."""
    with torch.autocast(device_type=batch["features"].device.type, enabled=False):
        world, root, relative = reconstruct_world(
            output, batch["root_init"], batch["relative_init"], task=task
        )
        valid = batch["frame_valid"]
        strata = _severity_strata(batch["severity"], world.shape[0])
        point_mask = valid[..., None]
        # Mask before nonlinear operations so arbitrary padding (including NaN)
        # cannot introduce NaN derivatives into otherwise valid examples.
        loss_world = torch.where(point_mask[..., None], world, 0)
        loss_root = torch.where(valid[..., None], root, 0)
        loss_relative = torch.where(point_mask[..., None], relative, 0)
        target = torch.where(point_mask[..., None], batch["target_world"].float(), 0)
        target_root = (
            target[:, :, [11, 12]].mean(dim=2) if task == "plcs" else target[:, :, 0]
        )
        target_relative = target - target_root[:, :, None]

        def reduce(value: Tensor, mask: Tensor) -> Tensor:
            return _balanced_mean(value, mask, strata)

        def huber(pred: Tensor, gt: Tensor) -> Tensor:
            return F.smooth_l1_loss(
                pred, gt, reduction="none", beta=config.huber_delta_m
            )

        def radial_huber(pred: Tensor, gt: Tensor) -> Tensor:
            distance = torch.linalg.vector_norm(pred - gt, dim=-1)
            return huber(distance, torch.zeros_like(distance))

        zero = loss_world.sum() * 0
        parts = {
            "root": reduce(radial_huber(loss_root, target_root), valid),
            # For BLCS root and world are the same position. Count it once.
            "world": (
                reduce(radial_huber(loss_world, target), point_mask)
                if task == "plcs"
                else zero
            ),
            "relative": (
                reduce(radial_huber(loss_relative, target_relative), point_mask)
                if task == "plcs"
                else zero
            ),
        }

        clean_mask = batch["clean_visible"] & valid[:, None, :, None]
        view_has_target = clean_mask.flatten(2).any(dim=-1)
        true_projection = torch.where(
            view_has_target[..., None, None], batch["true_projection"].float(), 0
        )
        homogeneous = torch.cat(
            (loss_world, torch.ones_like(loss_world[..., :1])), dim=-1
        )
        projected = torch.einsum("bvij,btkj->bvtki", true_projection, homogeneous)
        projected = torch.where(clean_mask[..., None], projected, 0)
        projected_uv = projected[..., :2] / projected[..., 2:3].clamp_min(0.05)
        clean_uv = torch.where(clean_mask[..., None], batch["clean_uv"].float(), 0)
        reprojection = F.smooth_l1_loss(
            projected_uv, clean_uv, reduction="none", beta=0.01
        )
        parts["reprojection"] = reduce(reprojection, clean_mask[..., None])

        velocity_valid = valid[:, 1:] & valid[:, :-1]
        fps = batch["fps"].reshape(-1, 1, 1, 1).float()
        velocity_error = F.smooth_l1_loss(
            torch.diff(loss_world, dim=1) * fps,
            torch.diff(target, dim=1) * fps,
            reduction="none",
            beta=1.0,
        )
        parts["velocity"] = reduce(velocity_error, velocity_valid[..., None, None])

        if task == "plcs":
            a, b = zip(*COCO17_BONE_LENGTH_EDGES, strict=True)
            pred_length = (loss_world[:, :, a] - loss_world[:, :, b]).norm(dim=-1)
            gt_length = (target[:, :, a] - target[:, :, b]).norm(dim=-1)
            parts["bone"] = reduce(huber(pred_length, gt_length), point_mask)
        else:
            parts["bone"] = zero

        initial = torch.where(point_mask[..., None], batch["init_world"].float(), 0)
        prediction_error = torch.linalg.vector_norm(loss_world - target, dim=-1)
        initial_error = torch.linalg.vector_norm(initial - target, dim=-1)
        if not torch.isfinite(initial_error).all():
            # An infinite baseline would otherwise turn every finite error into
            # zero regret, hiding a broken geometric initializer.
            raise ValueError("Non-finite initial error on valid supervision")
        regret = F.relu(prediction_error - initial_error - v2.regret_tolerance_m)
        parts["regret"] = reduce(regret, point_mask)

        total = (
            config.root_weight * parts["root"]
            + config.relative_weight * parts["relative"]
            + v2.balanced_world_weight * parts["world"]
            + config.reprojection_weight * parts["reprojection"]
            + config.velocity_weight * parts["velocity"]
            + config.bone_weight * parts["bone"]
            + v2.regret_weight * parts["regret"]
        )
        if not torch.isfinite(total):
            raise ValueError("Non-finite balanced residual loss on valid supervision")
    return total, parts, world
