"""3D ball trajectory adapter for token embeddings."""

from __future__ import annotations

from torch import Tensor
import torch.nn as nn

from src.common.models.embeddings import Ball3DEmbedding, InvisibleTokenEmbedding


class Ball3DTokenAdapter(nn.Module):
    """Adapter that maps 3D ball positions to backbone token embeddings."""

    def __init__(self, dim: int, dropout: float, invisible_token: InvisibleTokenEmbedding) -> None:
        super().__init__()
        self.embed = Ball3DEmbedding(dim=int(dim), dropout=float(dropout), invisible_token=invisible_token)

    def forward(self, ball_pos: Tensor, ball_vis: Tensor | None = None) -> Tensor:
        """Forward.

        Args:
            ball_pos: Ball positions, shape (B, T, 3).
            ball_vis: Optional visibility mask, shape (B, T).

        Returns:
            Embedded tokens, shape (B, T, D).
        """
        return self.embed(ball_pos, ball_vis)


if __name__ == "__main__":
    import torch

    invisible = InvisibleTokenEmbedding(dim=16)
    adapter = Ball3DTokenAdapter(dim=16, dropout=0.0, invisible_token=invisible)
    x = torch.randn(2, 8, 3)
    y = adapter(x)
    assert y.shape == (2, 8, 16)
    print("ball_3d_adapter smoke ok")
