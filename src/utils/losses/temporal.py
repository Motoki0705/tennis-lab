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
``smoothness_penalty`` defaults to the **third** difference (jerk). Penalizing
jerk enforces piecewise-*constant acceleration* — exactly ballistic free flight
(``a = -g``) for the ball and smooth locomotion for the player — without biasing
the real, non-zero acceleration of gravity, aerodynamic drag, or a running
player toward zero (which an acceleration penalty would). Sparse impulses
(bounces, direction changes) produce large jerk but are absorbed by the robust
(Smooth-L1) reduction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.tensor_utils import masked_mean


def finite_difference(x: Tensor, order: int) -> Tensor:
    """Successive first differences of ``x`` along the time axis (``dim=1``).

    Args:
        x: Sequence tensor, shape ``(B, T, ...)``.
        order: Number of first differences to apply (``>= 1``).

    Returns:
        Tensor of shape ``(B, T - order, ...)``. The value at output index ``i``
        depends on input frames ``i .. i + order`` (``order + 1`` frames).
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    for _ in range(order):
        x = x[:, 1:] - x[:, :-1]
    return x


def _consecutive_and(mask: Tensor, width: int) -> Tensor:
    """AND of ``width`` consecutive frame-validity flags along ``dim=1``.

    Args:
        mask: Frame validity mask, shape ``(B, T)`` (any dtype; ``> 0`` is valid).
        width: Number of consecutive frames that must all be valid.

    Returns:
        Boolean mask of shape ``(B, T - width + 1)``; entry ``i`` is ``True`` iff
        frames ``i .. i + width - 1`` are all valid.
    """
    valid = mask > 0
    length = valid.shape[1]
    out = valid[:, : length - width + 1]
    for offset in range(1, width):
        out = out & valid[:, offset : length - width + 1 + offset]
    return out


def smoothness_penalty(
    pred: Tensor,
    mask: Tensor | None = None,
    *,
    order: int = 3,
    beta: float = 1e-3,
    axis_weights: Sequence[float] | None = None,
) -> Tensor:
    """Robust penalty on the ``order``-th finite difference of a sequence.

    With ``order=3`` (default) this penalizes jerk, i.e. encourages piecewise
    constant acceleration without biasing gravity / drag / locomotion.

    Args:
        pred: Predicted normalized positions, shape ``(B, T, C)``.
        mask: Optional per-frame validity mask, shape ``(B, T)``.
        order: Finite-difference order (``3`` = jerk, ``2`` = acceleration).
        beta: Smooth-L1 transition point; small values keep sparse spikes
            (bounces, direction changes) in the robust linear regime.
        axis_weights: Optional per-channel weights, length ``C``.

    Returns:
        Scalar smoothness loss (``0`` when the sequence is too short).
    """
    if pred.shape[1] <= order:
        return pred.new_zeros(())
    diff = finite_difference(pred, order)  # (B, T - order, C)
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    if axis_weights is not None:
        weights = torch.as_tensor(axis_weights, device=per.device, dtype=per.dtype)
        if weights.numel() != per.shape[-1]:
            raise ValueError(
                f"axis_weights must have length {per.shape[-1]}, got {weights.numel()}"
            )
        per = per * weights.view(*([1] * (per.ndim - 1)), -1)
    if mask is None:
        return per.mean()
    window_mask = _consecutive_and(mask, order + 1)  # (B, T - order)
    return masked_mean(per, window_mask.to(per.dtype), denom_min=1.0)


def ballistic_second_difference(gravity: float, dt: float, height_scale: float) -> float:
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


def ballistic_gravity_penalty(
    pred_height: Tensor,
    mask: Tensor | None = None,
    *,
    target_second_difference: float,
    beta: float = 5e-3,
) -> Tensor:
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

    Args:
        pred_height: Predicted normalized height, shape ``(B, T)``.
        mask: Optional per-frame validity mask, shape ``(B, T)``.
        target_second_difference: Expected normalized second difference in free
            flight (negative).
        beta: Smooth-L1 transition point.

    Returns:
        Scalar gravity-consistency loss (``0`` when the sequence is too short).
    """
    if pred_height.shape[1] <= 2:
        return pred_height.new_zeros(())
    accel = finite_difference(pred_height.unsqueeze(-1), 2).squeeze(-1)  # (B, T - 2)
    resid = accel - target_second_difference
    per = F.smooth_l1_loss(resid, torch.zeros_like(resid), beta=beta, reduction="none")
    if mask is None:
        return per.mean()
    window_mask = _consecutive_and(mask, 3)  # (B, T - 2)
    return masked_mean(per, window_mask.to(per.dtype), denom_min=1.0)
