"""Observed court context for missing-ball tokens, without target information."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MissingBallCourtContext(nn.Module):
    """Zero-initialized, bias-free projection of masked UV and validity flags.

    Allocate zeros directly so enabling this module consumes no random numbers
    and leaves every existing model parameter's seeded initialization unchanged.
    Inputs follow the model's validated observation contract.
    """

    def __init__(self, *, num_court_kp: int, dim: int) -> None:
        super().__init__()
        if num_court_kp <= 0 or dim <= 0:
            raise ValueError("num_court_kp and dim must be positive.")
        self.num_court_kp = num_court_kp
        self.weight = nn.Parameter(torch.zeros(dim, num_court_kp * 3))

    def forward(
        self,
        court_kp: Tensor,
        court_vis: Tensor,
        ball_vis: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """Return context only for real missing-ball frames with observed court."""
        court_valid = court_vis > 0
        # Reapply the ordinary coordinate mask, also excluding masked nonfinite
        # placeholders before projection. The public adapter validates inputs.
        coordinates = torch.where(court_valid.unsqueeze(-1), court_kp, 0.0)
        features = torch.cat(
            (coordinates.flatten(-2), court_valid.to(court_kp.dtype)), dim=-1
        )
        context = F.linear(features, self.weight)
        active = ~(ball_vis > 0) & ~padding_mask & court_valid.any(dim=-1)
        return torch.where(active.unsqueeze(-1), context, 0.0)
