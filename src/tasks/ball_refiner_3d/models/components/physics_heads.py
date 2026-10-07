"""Field and flight-segment parameter heads over trunk tokens.

Both heads pool tokens with learned attention scores restricted to a mask: the
field head over the whole (unpadded) sequence, the segment head within each
flight segment.  The segment head also reads the token of the segment's first
frame, where its initial state is defined.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.ball_refiner_3d.physics.units import (
    FIELD_CHANNELS,
    STATE_CHANNELS,
    SURFACES,
)
from src.utils.models.components import RMSNorm

# Finite so that rows without members stay differentiable; they are masked later.
_EXCLUDED = -1e9


def _mlp(inputs: int, width: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, outputs))


class FieldHead(nn.Module):
    """Rally field ``(B,4)`` and surface logits ``(B,3)``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.score = nn.Linear(width, 1)
        self.mlp = _mlp(width, width, FIELD_CHANNELS + len(SURFACES))

    def forward(self, tokens: Tensor, padding: Tensor) -> tuple[Tensor, Tensor]:
        features = self.norm(tokens)
        weights = (
            self.score(features)[..., 0].masked_fill(padding, _EXCLUDED).softmax(dim=-1)
        )
        output = self.mlp((weights[..., None] * features).sum(dim=1))
        return output[:, :FIELD_CHANNELS], output[:, FIELD_CHANNELS:]


class SegmentHead(nn.Module):
    """Initial state ``(B,S,9)`` of every labelled segment."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.score = nn.Linear(width, 1)
        self.mlp = _mlp(2 * width, width, STATE_CHANNELS)

    def forward(self, tokens: Tensor, member: Tensor, start: Tensor) -> Tensor:
        """``member (B,S,T)`` assigns frames to segments; ``start (B,S)``."""
        features = self.norm(tokens)
        score = self.score(features)[..., 0]  # (B,T)
        weights = (
            score[:, None, :].masked_fill(~member, _EXCLUDED).softmax(dim=-1)
        )  # (B,S,T)
        pooled = weights @ features
        first = features.gather(
            1,
            start.clamp(max=tokens.shape[1] - 1)[..., None].expand(
                -1, -1, features.shape[-1]
            ),
        )
        result: Tensor = self.mlp(torch.cat((pooled, first), dim=-1))
        return result
