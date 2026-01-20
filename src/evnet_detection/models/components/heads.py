"""Prediction heads for event detection models."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class EventLogitsHead(nn.Module):
    """Per-frame event logits head.

    Args:
        input_dim: Input hidden dimension.
        num_events: Number of event classes.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int, num_events: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_events),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute logits.

        Args:
            x: Hidden states, (B, T, D).

        Returns:
            Logits, (B, T, E).
        """
        return self.net(x)


if __name__ == "__main__":
    import torch

    head = EventLogitsHead(input_dim=32, num_events=2, dropout=0.0)
    x = torch.randn(2, 16, 32)
    y = head(x)
    assert y.shape == (2, 16, 2)
    print("heads smoke ok")

