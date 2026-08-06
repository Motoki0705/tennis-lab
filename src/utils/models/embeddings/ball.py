"""Ball embeddings with invisible-token substitution."""

from __future__ import annotations

from torch import Tensor, nn

from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)


class BallUVEmbedding(nn.Module):
    """Embed 2D ball positions with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = CoordinateProjection(input_dim=2, dim=int(dim))
        self.invisible_token = invisible_token

    def forward(self, ball_uv: Tensor, ball_vis: Tensor) -> Tensor:
        """Embed 2D ball positions.

        Args:
            ball_uv: Ball UV positions, shape (B, T, 2).
            ball_vis: Visibility/observation mask, shape (B, T).

        Returns:
            Tensor: Embedded tokens, shape (B, T, D).
        """
        feat = self.proj(ball_uv)
        return apply_visibility_mask(feat, ball_vis, self.invisible_token)


class Ball3DEmbedding(nn.Module):
    """Embed 3D ball positions with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = CoordinateProjection(input_dim=3, dim=int(dim))
        self.invisible_token = invisible_token

    def forward(self, ball_pos: Tensor, ball_vis: Tensor) -> Tensor:
        """Embed 3D ball positions.

        Args:
            ball_pos: Ball positions, shape (B, T, 3).
            ball_vis: Visibility/observation mask, shape (B, T).

        Returns:
            Tensor: Embedded tokens, shape (B, T, D).
        """
        feat = self.proj(ball_pos)
        return apply_visibility_mask(feat, ball_vis, self.invisible_token)
