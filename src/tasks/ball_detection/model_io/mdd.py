"""Shared two-channel luminance difference, independent of detector architecture."""

from __future__ import annotations

import math

from torch import Tensor


def mdd_coefficients(a: float, b: float) -> tuple[float, float]:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("MDD parameters must be finite")
    return 5.0 / (0.45 * abs(math.tanh(a)) + 1e-6), 0.6 * math.tanh(b)


def luminance_to_mdd(luminance: Tensor, *, gain: float, offset: float) -> Tensor:
    """B,T,H,W -> B,2,T,H,W; first frame has no fabricated predecessor.

    Preserve the existing ConvNeXt sigmoid MDD convention exactly. Normalization
    of source RGB is the caller's explicit policy, never inferred here.
    """
    if luminance.ndim != 4 or luminance.shape[1] < 1:
        raise ValueError("Expected nonempty B,T,H,W luminance")
    diff = luminance[:, 1:] - luminance[:, :-1]
    output = luminance.new_zeros((len(luminance), 2, *luminance.shape[1:]))
    for channel, delta in enumerate((diff, -diff)):
        output[:, channel, 1:] = ((delta.clamp_min(0) - offset) * gain).clamp(-80, 80).sigmoid()
    return output
