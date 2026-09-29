"""Masked x0, robust full-GMM reprojection, weak gravity and event objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    DenoiserOutput,
    MixtureCondition,
)
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
)


@dataclass(frozen=True)
class LossConfig:
    x0: float
    reprojection: float
    physics: float
    event: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) <= 0 for f in fields(self)):
            raise ValueError("All four scaffold loss weights must be positive/finite")


@dataclass(frozen=True)
class TrainingBatch:
    condition: MixtureCondition
    target_positions_m: Tensor  # B,T,3
    event_labels: Tensor  # B,T,2
    free_flight_mask: Tensor  # B,T bool, excludes hit/bounce +/-5 and non-flight
    camera_matrices: Tensor  # B,V,3,4; source-pixel pinhole matrices
    means_2d_px: Tensor  # B,V,T,K,2
    covariance_2d_px2: Tensor  # B,V,T,K,2,2
    weights_2d: Tensor  # B,V,T,K
    presence_2d: Tensor  # B,V,T; amodal presence, not detector visibility
    gravity_mps2: Tensor  # B

    def to(self, device: str | torch.device) -> TrainingBatch:
        return TrainingBatch(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    # Empty physics/reprojection support contributes exactly zero, not fake data.
    return (values * mask).sum() / mask.sum().clamp_min(1)


def robust_reprojection(positions_m: Tensor, batch: TrainingBatch) -> Tensor:
    weight = batch.presence_2d * (~batch.condition.padding_mask)[:, None]
    selected = weight > 0
    if not bool(selected.any()):
        return positions_m.sum() * 0
    camera = batch.camera_matrices
    q = torch.einsum("bvij,btj->bvti", camera[..., :3], positions_m) + camera[..., 3][:, :, None]
    depth = q[..., 2]
    front = depth > 1e-3
    # Undefined projections are NOT dropped: they receive a depth penalty.
    denominator = torch.where(front, depth, torch.ones_like(depth))
    pixel = q[..., :2] / denominator[..., None]
    # Padding/absent observations need no artificial SPD matrix or mixture mass.
    delta = (pixel[selected][:, None, :] - batch.means_2d_px[selected])[..., None]
    chol = torch.linalg.cholesky(batch.covariance_2d_px2[selected])
    white = torch.linalg.solve_triangular(chol, delta, upper=False)
    mahalanobis = white.square().sum(dim=(-1, -2))
    # 2D Student-t (nu=4), mixed using every #935 component weight.
    log_density = -math.log(2 * math.pi) - chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) - 3 * torch.log1p(mahalanobis / 4)
    robust_nll = -torch.logsumexp(batch.weights_2d[selected].log() + log_density, dim=-1)
    penalty = 10 * F.relu(1e-3 - depth[selected])
    return _masked_mean(robust_nll + penalty, weight[selected])


def gravity_residual(positions_m: Tensor, batch: TrainingBatch) -> Tensor:
    times = batch.condition.timestamps_seconds
    valid = ~batch.condition.padding_mask
    mask = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
    mask = mask & batch.free_flight_mask[:, :-2] & batch.free_flight_mask[:, 1:-1] & batch.free_flight_mask[:, 2:]
    left_dt = torch.where(mask, times[:, 1:-1] - times[:, :-2], torch.ones_like(times[:, 1:-1]))
    right_dt = torch.where(mask, times[:, 2:] - times[:, 1:-1], torch.ones_like(times[:, 1:-1]))
    v_left = (positions_m[:, 1:-1] - positions_m[:, :-2]) / left_dt[..., None]
    v_right = (positions_m[:, 2:] - positions_m[:, 1:-1]) / right_dt[..., None]
    acceleration = 2 * (v_right - v_left) / (left_dt + right_dt)[..., None]
    gravity = torch.zeros_like(acceleration)
    gravity[..., 2] = batch.gravity_mps2[:, None]
    residual = (acceleration + gravity) / batch.gravity_mps2[:, None, None]
    values = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none").mean(-1)
    return _masked_mean(values, mask)


def trajectory_loss(output: DenoiserOutput, batch: TrainingBatch, config: LossConfig) -> tuple[Tensor, dict[str, Tensor]]:
    valid = ~batch.condition.padding_mask
    positions_m = denormalize_court_position(output.positions_norm)
    terms = {
        "x0": _masked_mean((output.positions_norm - normalize_court_position(batch.target_positions_m)).square().mean(-1), valid),
        "reprojection": robust_reprojection(positions_m, batch),
        "physics": gravity_residual(positions_m, batch),
        "event": _masked_mean(F.binary_cross_entropy_with_logits(output.event_logits, batch.event_labels.float(), reduction="none").mean(-1), valid),
    }
    total = sum(getattr(config, name) * value for name, value in terms.items())
    assert isinstance(total, Tensor)
    return total, terms
