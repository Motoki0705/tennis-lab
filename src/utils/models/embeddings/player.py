"""Player keypoint embeddings with invisible-token substitution."""

from __future__ import annotations

from torch import Tensor, nn

from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)


class PlayerKPUVEmbedding(nn.Module):
    """Embed player keypoints with invisible-token substitution.

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

    def forward(self, human_kp: Tensor, human_vis: Tensor) -> Tensor:
        """Embed human keypoints.

        Args:
            human_kp: Human keypoints, shape (B, 17, 2).
            human_vis: Visibility flags, shape (B, 17).

        Returns:
            Tensor: Embedded tokens, shape (B, NUM_HUMAN_KP, D).
        """
        feat = self.proj(human_kp)
        return apply_visibility_mask(feat, human_vis, self.invisible_token)
