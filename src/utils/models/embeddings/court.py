"""Court keypoint embeddings with invisible-token substitution."""

from __future__ import annotations

from torch import Tensor, nn

from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)


class CourtKPUVEmbedding(nn.Module):
    """Embed court keypoints with invisible-token substitution.

    Args:
        dim: Embedding dimension.
        dropout: Retained for API compatibility; ignored by the current projection stack.
        invisible_token: Shared invisible token module.
    """

    def __init__(
        self,
        *,
        dim: int,
        dropout: float = 0.0,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.proj = CoordinateProjection(input_dim=2, dim=int(dim))
        self.invisible_token = invisible_token

    def forward(self, court_kp: Tensor, court_vis: Tensor | None = None) -> Tensor:
        """Embed court keypoints.

        Args:
            court_kp: Court keypoints, shape (B, N*2) or (B, N, 2).
            court_vis: Visibility flags, shape (B, N). Optional.

        Returns:
            Tensor: Embedded tokens, shape (B, N, D).
        """
        batch_size = int(court_kp.shape[0])
        if court_kp.dim() == 2:
            court_kp = court_kp.reshape(batch_size, -1, 2)

        feat = self.proj(court_kp)
        return apply_visibility_mask(feat, court_vis, self.invisible_token)
