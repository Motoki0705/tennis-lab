"""Temporal smoothness / ballistic priors for trajectory predictions.

Both PLCS (player pelvis position) and BLCS (ball position) predict per-frame
3D coordinate sequences. Empirically their supervised position loss pins the
low-frequency trajectory but leaves the high-frequency, physically implausible
jitter unconstrained: on broadcast test splits the predicted second difference
(acceleration) is 20-70x noisier than ground truth. These priors add the
missing temporal constraint.

Design notes
------------
- All functions operate on **normalized court coordinates** ``(B, T, C)`` so the
  penalties live on the same scale as the supervised (normalized) position loss.
- Finite differences are taken along the time axis (``dim=1``) with a **masked**
  reduction, so padded frames never leak into the loss.
- Nothing here divides by ``dt``. A per-frame ``k``-th difference is proportional
  to the ``k``-th derivative, so a ``k``-th-difference penalty *is* a derivative
  penalty up to a constant that folds into the loss weight. Keeping the raw
  differences also keeps the magnitudes small and comparable to the position
  loss instead of exploding by ``1/dt**k``.
- The one quantity that needs an absolute physical scale is gravity. It enters
  through :func:`ballistic_second_difference`, which converts ``-g`` into the
  dimensionless target second difference of *normalized* height.

Choice of order
---------------
``TemporalSmoothnessPenalty(order=3)`` penalizes the **third** difference (jerk).
jerk enforces piecewise-*constant acceleration* — exactly ballistic free flight
(``a = -g``) for the ball and smooth locomotion for the player — without biasing
the real, non-zero acceleration of gravity, aerodynamic drag, or a running
player toward zero (which an acceleration penalty would). Sparse impulses
(bounces, direction changes) produce large jerk but are absorbed by the robust
(Smooth-L1) reduction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def finite_difference(x: Tensor, order: int) -> Tensor:
    """Successive first differences of ``x`` along the time axis (``dim=1``).

    Args:
        x: Sequence tensor, shape ``(B, T, ...)``.
        order: Number of first differences to apply (``>= 1``).

    Returns:
        Tensor of shape ``(B, T - order, ...)``. The value at output index ``i``
        depends on input frames ``i .. i + order`` (``order + 1`` frames).
    """
    if type(order) is not int or order < 1:
        raise ValueError(f"order must be a positive int, got {order!r}.")
    return torch.diff(x, n=order, dim=1)


def _consecutive_and(mask: Tensor, width: int) -> Tensor:
    """AND of ``width`` consecutive frame-validity flags along ``dim=1``.

    Args:
        mask: Boundary-prepared boolean frame-validity mask, shape ``(B, T)``.
        width: Number of consecutive frames that must all be valid.

    Returns:
        Boolean mask of shape ``(B, T - width + 1)``; entry ``i`` is ``True`` iff
        frames ``i .. i + width - 1`` are all valid.
    """
    length = mask.shape[1]
    out = mask[:, : length - width + 1]
    for offset in range(1, width):
        out = out & mask[:, offset : length - width + 1 + offset]
    return out


class TemporalSmoothnessPenalty(nn.Module):
    """Constructor-prepared robust temporal-difference penalty.

    Static option validation and axis-weight conversion happen once during
    construction. ``forward`` accepts only the validated prediction and mask
    tensors and performs no Python value conversion or option selection.
    """

    def __init__(
        self,
        *,
        order: int,
        axis_weights: Sequence[float],
        beta: float = 1e-3,
    ) -> None:
        super().__init__()
        if type(order) is not int or order < 1:
            raise ValueError(f"order must be a positive int, got {order!r}.")
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"beta must be finite and positive, got {beta!r}.")
        weights = tuple(float(weight) for weight in axis_weights)
        if not weights:
            raise ValueError("axis_weights must be non-empty.")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError(
                "axis_weights must contain only finite non-negative values."
            )
        self.order = order
        self.window_width = order + 1
        self.beta = float(beta)
        self.axis_weights: Tensor
        self.register_buffer(
            "axis_weights",
            torch.tensor(weights, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )

    def forward(self, prediction: Tensor, valid_mask: Tensor) -> Tensor:
        """Compute the penalty from boundary-validated ``(B,T,C)`` tensors."""
        difference = torch.diff(prediction, n=self.order, dim=1)
        per_value = F.smooth_l1_loss(
            difference,
            torch.zeros_like(difference),
            beta=self.beta,
            reduction="none",
        )
        weighted = per_value * self.axis_weights
        window_mask = _consecutive_and(valid_mask, self.window_width)
        expanded_mask = window_mask.unsqueeze(-1).expand_as(weighted)
        return weighted.masked_fill(
            ~expanded_mask, 0.0
        ).sum() / expanded_mask.sum().clamp_min(1)


def ballistic_second_difference(
    gravity: float, dt: float, height_scale: float
) -> float:
    """Target per-frame second difference of *normalized* height in free fall.

    In court coordinates height is ``+z`` and the ball obeys ``a_z = -g`` during
    free flight, so the second difference of physical height is ``-g * dt**2``.
    Dividing by the normalization scale gives the value in normalized units::

        Δ²z_norm = -g * dt**2 / height_scale

    Args:
        gravity: Gravitational acceleration in m/s**2 (e.g. ``9.81``).
        dt: Time between output frames in seconds (e.g. ``1 / 30``).
        height_scale: Normalization scale of the height axis in metres
            (``COURT_COORD_SCALE_Z``).

    Returns:
        The (negative) dimensionless target second difference.
    """
    if height_scale <= 0:
        raise ValueError(f"height_scale must be positive, got {height_scale}")
    return -gravity * dt * dt / height_scale


class BallisticGravityPenalty(nn.Module):
    """Pin the vertical curvature of a trajectory to the ballistic value.

    Penalizes the deviation of the per-frame second difference of normalized
    height from ``target_second_difference`` (see
    :func:`ballistic_second_difference`). This provides the *absolute* vertical
    curvature that — coupled with the 2D reprojection constraint — fixes the
    otherwise ambiguous monocular depth: a trajectory placed at the wrong depth
    projects to the same image only if its 3D scale (hence its vertical
    curvature) is wrong, which this term penalizes.

    The robust (Smooth-L1) reduction keeps sparse bounce impulses, where the true
    second difference is far from ``-g``, from dominating the free-flight signal.

    The target and robust-loss option are validated and stored at construction,
    leaving ``forward`` as tensor-only loss computation.
    """

    def __init__(
        self,
        *,
        target_second_difference: float,
        beta: float = 5e-3,
    ) -> None:
        super().__init__()
        if not math.isfinite(target_second_difference):
            raise ValueError("target_second_difference must be finite.")
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"beta must be finite and positive, got {beta!r}.")
        self.beta = float(beta)
        self.target_second_difference: Tensor
        self.register_buffer(
            "target_second_difference",
            torch.tensor(float(target_second_difference), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, pred_height: Tensor, valid_mask: Tensor) -> Tensor:
        """Compute gravity consistency from validated ``(B,T)`` tensors."""
        acceleration = torch.diff(pred_height, n=2, dim=1)
        residual = acceleration - self.target_second_difference
        per_value = F.smooth_l1_loss(
            residual,
            torch.zeros_like(residual),
            beta=self.beta,
            reduction="none",
        )
        window_mask = _consecutive_and(valid_mask, 3)
        return per_value.masked_fill(
            ~window_mask, 0.0
        ).sum() / window_mask.sum().clamp_min(1)
