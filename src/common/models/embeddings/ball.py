"""Ball embeddings with invisible-token substitution."""

from __future__ import annotations

import torch
from torch import nn, Tensor

from src.common.models.embeddings.shared import InvisibleTokenEmbedding


class BallUVEmbedding(nn.Module):
    """Embed 2D ball positions with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        dropout: Dropout probability.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        dropout: float = 0.1,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, int(dim)),
            nn.Dropout(float(dropout)),
        )
        self.invisible_token = invisible_token

    def forward(self, ball_uv: Tensor, ball_vis: Tensor | None = None) -> Tensor:
        """Embed 2D ball positions.

        Args:
            ball_uv: Ball UV positions, shape (B, T, 2).
            ball_vis: Visibility/observation mask, shape (B, T). Optional.

        Returns:
            Tensor: Embedded tokens, shape (B, T, D).
        """
        feat = self.proj(ball_uv)
        if ball_vis is None:
            return feat

        mask = (ball_vis > 0).unsqueeze(-1)
        inv = self.invisible_token().to(dtype=feat.dtype, device=feat.device)
        inv = inv.view(1, 1, -1).expand_as(feat)
        return torch.where(mask, feat, inv)


class Ball3DEmbedding(nn.Module):
    """Embed 3D ball positions with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        dropout: Dropout probability.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        dropout: float = 0.1,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, int(dim)),
            nn.Dropout(float(dropout)),
        )
        self.invisible_token = invisible_token

    def forward(self, ball_pos: Tensor, ball_vis: Tensor | None = None) -> Tensor:
        """Embed 3D ball positions.

        Args:
            ball_pos: Ball positions, shape (B, T, 3).
            ball_vis: Visibility/observation mask, shape (B, T). Optional.

        Returns:
            Tensor: Embedded tokens, shape (B, T, D).
        """
        feat = self.proj(ball_pos)
        if ball_vis is None:
            return feat

        mask = (ball_vis > 0).unsqueeze(-1)
        inv = self.invisible_token().to(dtype=feat.dtype, device=feat.device)
        inv = inv.view(1, 1, -1).expand_as(feat)
        return torch.where(mask, feat, inv)


if __name__ == "__main__":
    torch.manual_seed(0)
    invisible = InvisibleTokenEmbedding(dim=8)
    uv_embed = BallUVEmbedding(dim=8, dropout=0.0, invisible_token=invisible)
    ball_uv = torch.randn(2, 4, 2)
    ball_vis = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
    out_uv = uv_embed(ball_uv, ball_vis)
    assert out_uv.shape == (2, 4, 8)

    pos_embed = Ball3DEmbedding(dim=8, dropout=0.0, invisible_token=invisible)
    ball_pos = torch.randn(2, 4, 3)
    out_pos = pos_embed(ball_pos, ball_vis)
    assert out_pos.shape == (2, 4, 8)
    print("Ball embeddings smoke ok")
