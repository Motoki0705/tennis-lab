"""Event logits head adapter for multi-task learning."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.event_detection.models.components.heads import EventLogitsHead


class EventLogitsHeadAdapter(nn.Module):
    """Predict per-frame event logits from sequence features."""

    def __init__(self, input_dim: int, num_events: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.head = EventLogitsHead(
            input_dim=int(input_dim),
            num_events=int(num_events),
            dropout=float(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: Hidden states, shape (B, T, D).

        Returns:
            Event logits, shape (B, T, E).
        """
        return self.head(x)


if __name__ == "__main__":
    import torch

    head = EventLogitsHeadAdapter(input_dim=32, num_events=2, dropout=0.0)
    x = torch.randn(2, 8, 32)
    y = head(x)
    assert y.shape == (2, 8, 2)
    print("event_head smoke ok")
