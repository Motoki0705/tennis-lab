"""Shared transformer utility functions for multi-view models.

Provides standalone helpers that were previously duplicated as static methods
across ``BLCSMultiViewModel``, ``BLCSMultiViewAxialModel``, and
``PLCSMultiViewAxialModel``.
"""

from __future__ import annotations

from torch import Tensor


def _require_rope_theta(value: float, *, axis: str) -> float:
    if type(value) is not float:
        raise TypeError(
            f"rope_theta_{axis} must be exactly float; got {type(value).__name__}."
        )
    if value <= 0.0:
        raise ValueError(f"rope_theta_{axis} must be > 0; got {value}.")
    return value


def resolve_rope_bases(
    *,
    rope_theta_time: float,
    rope_theta_camera: float,
    rope_theta_type: float,
) -> tuple[float, float, float]:
    """Return explicit ``(time, camera, type)`` RoPE bases.

    Every axis must be selected by the validated model contract.  This helper
    never derives an axis value from another model field.
    """
    return (
        _require_rope_theta(rope_theta_time, axis="time"),
        _require_rope_theta(rope_theta_camera, axis="camera"),
        _require_rope_theta(rope_theta_type, axis="type"),
    )


def resolve_axial_rope_bases(
    *,
    rope_theta_time: float,
    rope_theta_camera: float,
) -> tuple[float, float]:
    """Return explicit ``(time, camera)`` RoPE bases for axial models."""
    return (
        _require_rope_theta(rope_theta_time, axis="time"),
        _require_rope_theta(rope_theta_camera, axis="camera"),
    )


def build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
    """Build a self-attention keep-mask from a boolean valid mask.

    Converts a token validity mask into a square attention mask suitable for
    ``TransformerBlock``'s ``attn_mask`` argument (True = this (query, key)
    pair is attended to).

    NaN-prevention repair
    ---------------------
    If an entire sequence in the batch is fully masked (all tokens invalid),
    every attention weight would be ``-inf`` after masking, producing NaN
    after softmax.  To prevent this the function forces token 0 of any
    fully-masked sequence to be valid before building the attention mask.
    The caller should zero-out the corresponding output token after the
    attention layer if it wants to preserve the semantics of "no valid tokens".

    Args:
        valid: Boolean valid mask with shape ``(B*, S)`` where ``B*`` is any
            batch dimension and ``S`` is the sequence length.  An element is
            treated as valid (attending) when it is ``True``.

    Returns:
        tuple:
            - ``attn_mask``: Boolean keep-mask of shape ``(B*, S, S)``.
              Element ``[b, i, j]`` is ``True`` when query ``i`` is allowed to
              attend to key ``j`` in batch element ``b``.
            - ``valid_fixed``: Boolean valid mask of shape ``(B*, S)``, equal
              to ``valid.bool()`` but with token 0 forced to ``True`` for any
              fully-masked sequence (the NaN-prevention repair described above).
    """
    valid_fixed = valid.bool()
    fully_masked = ~valid_fixed.any(dim=1)
    if fully_masked.any():
        valid_fixed = valid_fixed.clone()
        valid_fixed[fully_masked, 0] = True
    attn_mask = valid_fixed[:, None, :].expand(
        valid_fixed.shape[0],
        valid_fixed.shape[1],
        valid_fixed.shape[1],
    )
    return attn_mask, valid_fixed


def validate_rope_dim(*, rope_dim: int, head_dim: int) -> None:
    """Validate that ``rope_dim`` is a legal RoPE dimension.

    Args:
        rope_dim: The RoPE dimension to validate.
        head_dim: The attention head dimension that ``rope_dim`` may not exceed.

    Raises:
        ValueError: If ``rope_dim`` is odd or greater than ``head_dim``.
    """
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if rope_dim > head_dim:
        raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")
