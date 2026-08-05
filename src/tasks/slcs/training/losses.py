"""Loss functions for SLCS training on noisy pseudo-labels.

Design (documented in the task README):

- **Supervised terms** are masked by ``frame_mask & target_*_valid`` and
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

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.geometry.angles import wrapped_angle_diff
from src.utils.losses.temporal import smoothness_penalty
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
    player_mask: Tensor  # (B, P, T) bool: frame_mask & target_player_valid
    player_weight: Tensor  # (B, P, T) float
    ball_mask: Tensor  # (B, T) bool: frame_mask & target_ball_valid
    ball_weight: Tensor  # (B, T) float
    frame_mask: Tensor  # (B, T) bool (padding only; label-independent)

    @property
    def zero(self) -> Tensor:
        return self.pred_player_position.new_zeros(())


def _weighted_mean(per_frame: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
    """Weighted masked mean; zero when nothing is valid."""
    if per_frame.shape != mask.shape or per_frame.shape != weight.shape:
        raise ValueError(
            f"per-frame loss {tuple(per_frame.shape)}, mask {tuple(mask.shape)} and "
            f"weight {tuple(weight.shape)} must share the same shape."
        )
    effective = weight * mask.to(weight.dtype)
    denom = effective.sum()
    if denom.item() <= 0:
        return per_frame.new_zeros(())
    return (per_frame * effective).sum() / denom


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
    order: int,
) -> Callable[[SLCSLossInputs], Tensor]:
    """Jerk prior on player positions over all real (non-padded) frames."""

    def term(inputs: SLCSLossInputs) -> Tensor:
        pred = inputs.pred_player_position
        batch, players, seq_len, dims = pred.shape
        flat = pred.reshape(batch * players, seq_len, dims)
        mask = (
            inputs.frame_mask.unsqueeze(1)
            .expand(batch, players, seq_len)
            .reshape(batch * players, seq_len)
        )
        return smoothness_penalty(flat, mask, order=order)

    return term


def make_ball_position_smoothness_term(
    order: int,
) -> Callable[[SLCSLossInputs], Tensor]:
    """Jerk prior on the ball trajectory over all real frames."""

    def term(inputs: SLCSLossInputs) -> Tensor:
        return smoothness_penalty(
            inputs.pred_ball_position, inputs.frame_mask, order=order
        )

    return term


def ground_penetration_loss_term(inputs: SLCSLossInputs) -> Tensor:
    """Hinge on negative normalized height for players and ball (court z=0)."""
    player_pen = nn.functional.relu(-inputs.pred_player_position[..., 2])
    ball_pen = nn.functional.relu(-inputs.pred_ball_position[..., 2])
    player_mask = inputs.frame_mask.unsqueeze(1).expand_as(player_pen)
    player_term = masked_mean(player_pen, player_mask, binarize=True, denom_min=1.0)
    ball_term = masked_mean(ball_pen, inputs.frame_mask, binarize=True, denom_min=1.0)
    return player_term + ball_term


SLCSLossTerm = Callable[[SLCSLossInputs], Tensor]


class SLCSLoss(nn.Module):
    """Combined SLCS loss (registry pattern; see module docstring)."""

    def __init__(self, config: SLCSLossConfig) -> None:
        super().__init__()
        self.config = config
        order = int(self.config.smoothness_order)
        self.loss_terms: dict[str, SLCSLossTerm] = {
            "player_position": player_position_loss_term,
            "player_rotation": player_rotation_loss_term,
            "player_angle": player_angle_loss_term,
            "ball_position": ball_position_loss_term,
            "player_position_nll": player_position_nll_loss_term,
            "player_rotation_nll": player_rotation_nll_loss_term,
            "ball_position_nll": ball_position_nll_loss_term,
            "player_position_smoothness": make_player_position_smoothness_term(order),
            "ball_position_smoothness": make_ball_position_smoothness_term(order),
            "ground_penetration": ground_penetration_loss_term,
        }
        self.loss_weights = {
            "player_position": config.player_position_weight,
            "player_rotation": config.player_rotation_weight,
            "player_angle": config.player_angle_weight,
            "ball_position": config.ball_position_weight,
            "player_position_nll": config.player_position_nll_weight,
            "player_rotation_nll": config.player_rotation_nll_weight,
            "ball_position_nll": config.ball_position_nll_weight,
            "player_position_smoothness": config.player_position_smoothness_weight,
            "ball_position_smoothness": config.ball_position_smoothness_weight,
            "ground_penetration": config.ground_penetration_weight,
        }

    def weight_for(self, name: str) -> float:
        try:
            return self.loss_weights[name]
        except KeyError as error:
            raise KeyError(f"Unknown SLCS loss term {name!r}.") from error

    def forward(
        self,
        outputs: dict[str, Tensor],
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute all registered terms from model outputs and an SLCSBatch."""
        inputs = self.build_inputs(outputs, batch)
        losses: dict[str, Tensor] = {}
        total = inputs.zero
        for name, term_fn in self.loss_terms.items():
            value = term_fn(inputs)
            losses[name] = value
            total = total + self.weight_for(name) * value
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(
                f"SLCS loss became non-finite: "
                f"{ {k: float(v.detach()) for k, v in losses.items()} }"
            )
        losses["total"] = total
        return losses

    @staticmethod
    def build_inputs(
        outputs: dict[str, Tensor], batch: dict[str, Tensor]
    ) -> SLCSLossInputs:
        """Assemble loss inputs, folding padding into label masks/weights."""
        frame_mask = batch["frame_mask"] > 0
        player_mask = (batch["target_player_valid"] > 0) & frame_mask.unsqueeze(1)
        ball_mask = (batch["target_ball_valid"] > 0) & frame_mask
        player_weight = batch["target_player_weight"] * player_mask.to(
            batch["target_player_weight"].dtype
        )
        ball_weight = batch["target_ball_weight"] * ball_mask.to(
            batch["target_ball_weight"].dtype
        )
        return SLCSLossInputs(
            pred_player_position=outputs["player_position"],
            pred_player_rotation=outputs["player_rotation"],
            pred_ball_position=outputs["ball_position"],
            pred_player_position_log_b=outputs["player_position_log_b"],
            pred_player_rotation_log_b=outputs["player_rotation_log_b"],
            pred_ball_position_log_b=outputs["ball_position_log_b"],
            target_player_position=batch["target_player_position"],
            target_player_rotation=batch["target_player_rotation"],
            target_ball_position=batch["target_ball_position"],
            player_mask=player_mask,
            player_weight=player_weight,
            ball_mask=ball_mask,
            ball_weight=ball_weight,
            frame_mask=frame_mask,
        )


__all__ = [
    "SLCSLoss",
    "SLCSLossConfig",
    "SLCSLossInputs",
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
