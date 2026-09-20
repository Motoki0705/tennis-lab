"""Observed feature context for window edges with exactly one anchor side."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MissingBallOneSidedContext(nn.Module):
    """Project nearest observed features, direction and log frame distance.

    Sources are original court+ball embeddings in an independent contiguous,
    equally spaced offline window. This is context, not physical extrapolation.
    Direct zero allocation consumes no RNG and preserves shared initialization.
    """

    def __init__(self, *, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.weight = nn.Parameter(torch.zeros(dim, dim + 3))

    def forward(
        self, ball_tokens: Tensor, source_valid: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """Consume model-owned (B,T,D) features and validated bool (B,T) masks."""
        valid = source_valid & ~padding_mask
        length = ball_tokens.shape[1]
        index = torch.arange(length, device=ball_tokens.device).expand_as(valid)
        left = torch.where(valid, index, -1).cummax(dim=1).values
        right = (
            torch.where(valid, index, length).flip([1]).cummin(dim=1).values.flip([1])
        )
        has_left, has_right = left >= 0, right < length
        active = ~valid & ~padding_mask & (has_left ^ has_right)
        anchor = torch.where(has_left, left, right).clamp(0, length - 1)
        # Sanitize invalid placeholders before gather/arithmetic, not by
        # multiplication: NaN/Inf and gradients must not leak from hidden data.
        sources = torch.where(valid.unsqueeze(-1), ball_tokens, 0.0)
        features = sources.gather(1, anchor.unsqueeze(-1).expand_as(sources))
        distance = (index - anchor).abs().to(ball_tokens.dtype).log1p()
        context = torch.cat(
            [
                features,
                has_left.unsqueeze(-1).to(ball_tokens.dtype),
                has_right.unsqueeze(-1).to(ball_tokens.dtype),
                distance.unsqueeze(-1),
            ],
            dim=-1,
        )
        context = torch.where(active.unsqueeze(-1), context, 0.0)
        return F.linear(context, self.weight)
