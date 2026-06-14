"""Shared transformer utility functions for multi-view models.

Provides standalone helpers that were previously duplicated as static methods
across ``BLCSMultiViewModel``, ``BLCSMultiViewAxialModel``, and
``PLCSMultiViewAxialModel``.
"""

from __future__ import annotations

from torch import Tensor


def resolve_rope_bases(
    rope_theta: float,
    rope_theta_time: float | None,
    rope_theta_camera: float | None,
    rope_theta_type: float | None = None,
) -> tuple[float, ...]:
    """Resolve per-axis RoPE theta bases with fallback to ``rope_theta``.

    Reproduces the ``rope_bases`` tuple construction shared by the PLCS/BLCS
    frame models (3-tuple including a type axis) and the axial multi-view
    models (2-tuple, no type axis).

    Each axis-specific theta falls back to ``rope_theta`` when ``None``; the
    resulting bases are ``(time, camera[, type])``.  The third (``type``)
    element is only included when ``rope_theta_type is not None``.

    Args:
        rope_theta: Default RoPE theta used when an axis value is ``None``.
        rope_theta_time: Theta for the time axis, or ``None`` to fall back.
        rope_theta_camera: Theta for the camera axis, or ``None`` to fall back.
        rope_theta_type: Theta for the type axis.  When ``None``, the type
            axis is omitted (yielding a 2-tuple); otherwise it is included.

    Returns:
        tuple[float, ...]: ``(time, camera)`` or ``(time, camera, type)``.
    """
    bases = [
        float(rope_theta if rope_theta_time is None else rope_theta_time),
        float(rope_theta if rope_theta_camera is None else rope_theta_camera),
    ]
    if rope_theta_type is not None:
        bases.append(float(rope_theta_type))
    return tuple(bases)


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
