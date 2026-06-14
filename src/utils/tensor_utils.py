"""Shared tensor helpers for masking and reduction.

These utilities consolidate logic that was previously duplicated across task
training modules (``plcs``/``blcs`` losses, metrics and lightning modules).
The functions are intentionally parameterized so each historical call site can
reproduce its exact numerical behavior while sharing a single implementation.
"""

from __future__ import annotations

from torch import Tensor


def masked_mean(
    values: Tensor,
    mask: Tensor | None = None,
    *,
    binarize: bool = False,
    broadcast: bool = True,
    denom_min: float | None = None,
    eps: float = 0.0,
) -> Tensor:
    """Compute the mean of ``values`` over an optional ``mask``.

    Parameters
    ----------
    values:
        Tensor to reduce.
    mask:
        Optional mask. When ``None`` a plain ``values.mean()`` is returned.
    binarize:
        If ``True`` the mask is treated as boolean via ``mask > 0`` before being
        cast to ``values.dtype`` (matches the PLCS convention).
    broadcast:
        If ``True`` the mask is unsqueezed/expanded to ``values.ndim`` (matches
        the BLCS convention). For equal-rank inputs this is a no-op.
    denom_min:
        If set, the summed mask (denominator) is clamped to this minimum
        (PLCS uses ``1.0``).
    eps:
        Added to the denominator after optional clamping (BLCS uses ``1e-8``).
    """
    if mask is None:
        return values.mean()
    m = (mask > 0).to(dtype=values.dtype) if binarize else mask.to(dtype=values.dtype)
    if broadcast:
        while m.ndim < values.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(values)
    numerator = (values * m).sum()
    denominator = m.sum()
    if denom_min is not None:
        denominator = denominator.clamp_min(denom_min)
    return numerator / (denominator + eps)


def normalize_padding_mask(
    mask: Tensor | None,
    *,
    flatten: bool = False,
) -> Tensor | None:
    """Normalize a multi-dim padding mask to a frame-level validity mask.

    Supported input shapes and their reductions to a ``(B, T)`` (or ``(B,)``)
    boolean frame mask:

    - ``(B,)``    -> ``mask > 0``
    - ``(B, T)``  -> ``mask > 0``
    - ``(B, N, T)`` -> ``(mask > 0).any(dim=1)``
    - ``(B, N, T, J)`` -> ``(mask > 0).any(dim=1).any(dim=-1)``

    Parameters
    ----------
    mask:
        Padding mask, or ``None``.
    flatten:
        If ``True`` the result is flattened with ``reshape(-1)`` (used by metric
        code that indexes a flattened batch).
    """
    if mask is None:
        return None
    dim = mask.dim()
    if dim in (1, 2):
        frame_valid = mask > 0
    elif dim == 3:
        frame_valid = (mask > 0).any(dim=1)
    elif dim == 4:
        frame_valid = (mask > 0).any(dim=1).any(dim=-1)
    else:
        raise ValueError(
            "mask must be (B,), (B,T), (B,N,T), or (B,N,T,J), "
            f"got shape {tuple(mask.shape)}"
        )
    return frame_valid.reshape(-1) if flatten else frame_valid


__all__ = ["masked_mean", "normalize_padding_mask"]
