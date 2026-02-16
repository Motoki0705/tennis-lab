"""In-frame classification head for multi-task learning."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class InFrameHead(nn.Module):
    """Predict per-frame in-frame logits from sequence features.

    Args:
        input_dim: Input hidden dimension.
        hidden_dim: Hidden dimension for the head MLP.
        dropout: Dropout probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward.

        Args:
            x: Hidden states, shape (B, T, D).

        Returns:
            In-frame logits, shape (B, T).
        """
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    import torch

    head = InFrameHead(input_dim=32, dropout=0.0)
    x = torch.randn(2, 8, 32)
    y = head(x)
    assert y.shape == (2, 8)
    print("in_frame_head smoke ok")
