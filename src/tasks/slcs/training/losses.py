"""Loss functions for SLCS training on noisy pseudo-labels.

Design (documented in the task README):

- **Supervised terms** are masked by ``~padding_mask & target_*_valid`` and
  weighted by the per-frame pseudo-label confidence weights produced by the
  data-side quality filter, so unreliable labels contribute proportionally
  less.
- **Robustness** comes from three cooperating mechanisms: smooth-L1 base
  terms (comparable with BLCS/PLCS), Laplace negative log-likelihood terms
  with learned per-frame scale ``b = exp(log_b)`` (the model can down-weight
  frames it recognizes as unpredictable — heteroscedastic outlier handling),
  and jerk smoothness priors that suppress pseudo-label jitter.
- **Rotation** keeps the PLCS pair ``1 - cos`` + wrapped-angle smooth-L1 so
  results stay directly comparable, plus a Laplace NLL on the wrapped angular
  error with its own learned scale.
- **Geometric prior**: a hinge on negative height (court plane is z=0)
  penalizes below-ground predictions for both players and ball.
- **No reprojection loss**: the issue #634 clips carry no calibrated cameras
  and this module refuses to invent them.

Loss terms follow the PLCS registry pattern: uniform signature
``(SLCSLossInputs) -> Tensor``, registered by name, weighted via
``SLCSLossConfig.<name>_weight``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.slcs.model_io.contracts import (
    SLCSDecodedOutput,
    SLCSTrainingTargets,
)
from src.utils.geometry.angles import wrapped_angle_diff
from src.utils.losses.temporal import TemporalSmoothnessPenalty
from src.utils.tensor_utils import masked_mean


@dataclass(frozen=True)
class SLCSLossConfig:
    """Weights for the SLCS loss registry (0 disables a term)."""

    player_position_weight: float
    player_rotation_weight: float
    player_angle_weight: float
    ball_position_weight: float
    player_position_nll_weight: float
    player_rotation_nll_weight: float
    ball_position_nll_weight: float
    player_position_smoothness_weight: float
    ball_position_smoothness_weight: float
    ground_penetration_weight: float
    smoothness_order: int


@dataclass(frozen=True)
class SLCSLossInputs:
    """Tensors shared across SLCS loss terms.

    Predictions/targets are in normalized court coordinates; masks are boolean
    with padding already folded in; weights are label-confidence weights that
    are zero wherever the corresponding valid mask is False.
    """

    pred_player_position: Tensor  # (B, P, T, 3)
    pred_player_rotation: Tensor  # (B, P, T, 2)
    pred_ball_position: Tensor  # (B, T, 3)
    pred_player_position_log_b: Tensor  # (B, P, T)
    pred_player_rotation_log_b: Tensor  # (B, P, T)
    pred_ball_position_log_b: Tensor  # (B, T)
    target_player_position: Tensor  # (B, P, T, 3)
    target_player_rotation: Tensor  # (B, P, T, 2)
    target_ball_position: Tensor  # (B, T, 3)
    player_mask: Tensor  # (B, P, T) bool: ~padding_mask & target_player_valid
    player_weight: Tensor  # (B, P, T) float
    ball_mask: Tensor  # (B, T) bool: ~padding_mask & target_ball_valid
    ball_weight: Tensor  # (B, T) float
    padding_mask: Tensor  # (B, T) bool, True for padding

    @property
    def zero(self) -> Tensor:
        return self.pred_player_position.new_zeros(())


def _weighted_mean(per_frame: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    """Weighted masked mean; zero when nothing is valid."""
    effective = weight * mask.to(weight.dtype)
    denom = effective.sum()
    weighted = (per_frame * effective).sum() / denom.clamp_min(1.0)
    return torch.where(denom > 0, weighted, per_frame.new_zeros(()))


def player_position_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Smooth-L1 on player positions (masked, confidence-weighted)."""
    per_frame = nn.functional.smooth_l1_loss(
        inputs.pred_player_position, inputs.target_player_position, reduction="none"
    ).mean(dim=-1)
    return _weighted_mean(per_frame, inputs.player_mask, inputs.player_weight)


def player_rotation_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """``1 - cos`` similarity on (cos, sin) yaw (masked, confidence-weighted)."""
    pred = nn.functional.normalize(inputs.pred_player_rotation, dim=-1)
    target = nn.functional.normalize(inputs.target_player_rotation, dim=-1)
    per_frame = 1.0 - (pred * target).sum(dim=-1)
    return _weighted_mean(per_frame, inputs.player_mask, inputs.player_weight)


def _wrapped_yaw_error(pred_rotation: Tensor, target_rotation: Tensor) -> Tensor:
    pred_angle = torch.atan2(pred_rotation[..., 1], pred_rotation[..., 0])
    target_angle = torch.atan2(target_rotation[..., 1], target_rotation[..., 0])
    return wrapped_angle_diff(pred_angle, target_angle)


def player_angle_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Wrapped-angle smooth-L1 (complements ``1 - cos`` near the antipode)."""
    diff = _wrapped_yaw_error(
        inputs.pred_player_rotation, inputs.target_player_rotation
    )
    per_frame = nn.functional.smooth_l1_loss(
        diff, torch.zeros_like(diff), reduction="none"
    )
    return _weighted_mean(per_frame, inputs.player_mask, inputs.player_weight)


def ball_position_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Smooth-L1 on the ball trajectory (masked, confidence-weighted)."""
    per_frame = nn.functional.smooth_l1_loss(
        inputs.pred_ball_position, inputs.target_ball_position, reduction="none"
    ).mean(dim=-1)
    return _weighted_mean(per_frame, inputs.ball_mask, inputs.ball_weight)


def _laplace_nll(l1_error: Tensor, log_b: Tensor) -> Tensor:
    """Per-frame Laplace NLL (up to constants): ``|err| / b + log b``."""
    return l1_error * torch.exp(-log_b) + log_b


def player_position_nll_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Heteroscedastic Laplace NLL on player positions."""
    l1 = (
        (inputs.pred_player_position - inputs.target_player_position).abs().mean(dim=-1)
    )
    per_frame = _laplace_nll(l1, inputs.pred_player_position_log_b)
    return _weighted_mean(per_frame, inputs.player_mask, inputs.player_weight)


def player_rotation_nll_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Heteroscedastic Laplace NLL on the wrapped yaw error (radians)."""
    err = _wrapped_yaw_error(
        inputs.pred_player_rotation, inputs.target_player_rotation
    ).abs()
    per_frame = _laplace_nll(err, inputs.pred_player_rotation_log_b)
    return _weighted_mean(per_frame, inputs.player_mask, inputs.player_weight)


def ball_position_nll_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Heteroscedastic Laplace NLL on the ball trajectory."""
    l1 = (inputs.pred_ball_position - inputs.target_ball_position).abs().mean(dim=-1)
    per_frame = _laplace_nll(l1, inputs.pred_ball_position_log_b)
    return _weighted_mean(per_frame, inputs.ball_mask, inputs.ball_weight)


def make_player_position_smoothness_term(
    penalty: TemporalSmoothnessPenalty,
) -> Callable[[SLCSLossInputs], Tensor]:
    """Jerk prior on player positions over all real (non-padded) frames."""

    def term(inputs: SLCSLossInputs) -> Tensor:
        pred = inputs.pred_player_position
        batch, players, seq_len, dims = pred.shape
        flat = pred.reshape(batch * players, seq_len, dims)
        mask = (
            (~inputs.padding_mask)
            .unsqueeze(1)
            .expand(batch, players, seq_len)
            .reshape(batch * players, seq_len)
        )
        return cast(Tensor, penalty(flat, mask))

    return term


def make_ball_position_smoothness_term(
    penalty: TemporalSmoothnessPenalty,
) -> Callable[[SLCSLossInputs], Tensor]:
    """Jerk prior on the ball trajectory over all real frames."""

    def term(inputs: SLCSLossInputs) -> Tensor:
        return cast(Tensor, penalty(inputs.pred_ball_position, ~inputs.padding_mask))

    return term


def ground_penetration_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Hinge on negative normalized height for players and ball (court z=0)."""
    player_pen = nn.functional.relu(-inputs.pred_player_position[..., 2])
    ball_pen = nn.functional.relu(-inputs.pred_ball_position[..., 2])
    frame_valid = ~inputs.padding_mask
    player_mask = frame_valid.unsqueeze(1).expand_as(player_pen)
    player_term = masked_mean(player_pen, player_mask, binarize=True, denom_min=1.0)
    ball_term = masked_mean(ball_pen, frame_valid, binarize=True, denom_min=1.0)
    return player_term + ball_term


SLCSLossTerm = Callable[[SLCSLossInputs], Tensor]


class SLCSLoss(nn.Module):
    """Combined SLCS loss (registry pattern; see module docstring)."""

    def __init__(self, config: SLCSLossConfig) -> None:
        super().__init__()
        self.config = config
        self.temporal_smoothness = TemporalSmoothnessPenalty(
            order=config.smoothness_order,
            axis_weights=(1.0, 1.0, 1.0),
        )
        self.weighted_terms: tuple[tuple[str, SLCSLossTerm, float], ...] = (
            (
                "player_position",
                player_position_loss_term,
                config.player_position_weight,
            ),
            (
                "player_rotation",
                player_rotation_loss_term,
                config.player_rotation_weight,
            ),
            ("player_angle", player_angle_loss_term, config.player_angle_weight),
            ("ball_position", ball_position_loss_term, config.ball_position_weight),
            (
                "player_position_nll",
                player_position_nll_loss_term,
                config.player_position_nll_weight,
            ),
            (
                "player_rotation_nll",
                player_rotation_nll_loss_term,
                config.player_rotation_nll_weight,
            ),
            (
                "ball_position_nll",
                ball_position_nll_loss_term,
                config.ball_position_nll_weight,
            ),
            (
                "player_position_smoothness",
                make_player_position_smoothness_term(self.temporal_smoothness),
                config.player_position_smoothness_weight,
            ),
            (
                "ball_position_smoothness",
                make_ball_position_smoothness_term(self.temporal_smoothness),
                config.ball_position_smoothness_weight,
            ),
            (
                "ground_penetration",
                ground_penetration_loss_term,
                config.ground_penetration_weight,
            ),
        )

    def forward(
        self,
        inputs: SLCSLossInputs,
    ) -> dict[str, Tensor]:
        """Compute registered terms from boundary-validated loss tensors."""
        losses: dict[str, Tensor] = {}
        total = inputs.zero
        for name, term_fn, weight in self.weighted_terms:
            value = term_fn(inputs)
            losses[name] = value
            total = total + weight * value
        losses["total"] = total
        return losses


def build_slcs_loss_inputs(
    outputs: SLCSDecodedOutput, targets: SLCSTrainingTargets
) -> SLCSLossInputs:
    """Assemble the computation-only loss input after adapter decode."""
    return SLCSLossInputs(
        pred_player_position=outputs.player_position,
        pred_player_rotation=outputs.player_rotation,
        pred_ball_position=outputs.ball_position,
        pred_player_position_log_b=outputs.player_position_log_b,
        pred_player_rotation_log_b=outputs.player_rotation_log_b,
        pred_ball_position_log_b=outputs.ball_position_log_b,
        target_player_position=targets.target_player_position,
        target_player_rotation=targets.target_player_rotation,
        target_ball_position=targets.target_ball_position,
        player_mask=targets.player_mask,
        player_weight=targets.player_weight,
        ball_mask=targets.ball_mask,
        ball_weight=targets.ball_weight,
        padding_mask=targets.padding_mask,
    )


__all__ = [
    "SLCSLoss",
    "SLCSLossConfig",
    "SLCSLossInputs",
    "build_slcs_loss_inputs",
    "ball_position_loss_term",
    "ball_position_nll_loss_term",
    "ground_penetration_loss_term",
    "make_ball_position_smoothness_term",
    "make_player_position_smoothness_term",
    "player_angle_loss_term",
    "player_position_loss_term",
    "player_position_nll_loss_term",
    "player_rotation_loss_term",
    "player_rotation_nll_loss_term",
]
