"""Window-local observed feature anchors for missing ball tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MissingBallTemporalContext(nn.Module):
    """Interpolate encoded observations, not physical UV or 3D trajectories.

    Each batch row is an independent offline, uniformly sampled contiguous
    window. Both observed anchors must exist inside that window. No state or
    extrapolation is used. Sources precede court residual/entity/time additions.
    Direct zero allocation preserves the legacy model's initialization RNG.
    """

    def __init__(self, *, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.weight = nn.Parameter(torch.zeros(dim, dim))

    def forward(
        self, ball_tokens: Tensor, source_valid: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """Return a residual only for bracketed missing, nonpadding frames."""
        if ball_tokens.ndim != 3 or ball_tokens.shape[-1] != self.weight.shape[1]:
            raise ValueError("ball_tokens must have shape (B,T,D) with configured D.")
        if ball_tokens.shape[1] == 0:
            raise ValueError("ball_tokens must contain at least one frame.")
        for name, mask in (
            ("source_valid", source_valid),
            ("padding_mask", padding_mask),
        ):
            if mask.shape != ball_tokens.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be a bool tensor of shape (B,T).")
        valid = source_valid & ~padding_mask
        length = ball_tokens.shape[1]
        index = torch.arange(length, device=ball_tokens.device).expand_as(valid)
        left = torch.where(valid, index, -1).cummax(dim=1).values
        right = (
            torch.where(valid, index, length).flip([1]).cummin(dim=1).values.flip([1])
        )
        active = ~valid & ~padding_mask & (left >= 0) & (right < length)
        # Mask before gathering/arithmetic: invalid placeholders, including
        # NaN/Inf, must neither contribute features nor receive gradients.
        sources = torch.where(valid.unsqueeze(-1), ball_tokens, 0.0)
        left_features = sources.gather(
            1, left.clamp(min=0).unsqueeze(-1).expand_as(sources)
        )
        right_features = sources.gather(
            1, right.clamp(max=length - 1).unsqueeze(-1).expand_as(sources)
        )
        fraction = ((index - left) / (right - left).clamp(min=1)).to(ball_tokens.dtype)
        interpolated = left_features + fraction.unsqueeze(-1) * (
            right_features - left_features
        )
        interpolated = torch.where(active.unsqueeze(-1), interpolated, 0.0)
        return F.linear(interpolated, self.weight)
