"""Court keypoint embeddings with invisible-token substitution."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)
from src.utils.schema.court import NUM_COURT_KP


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


class CourtCameraEmbedding(nn.Module):
    """Map an unordered court-point set to one token per camera and frame.

    Invisible coordinates are zeroed before a shared point projection. Mean
    pooling then makes the result invariant to the 14-point input order.
    """

    def __init__(self, *, dim: int, num_court_points: int = NUM_COURT_KP) -> None:
        super().__init__()
        self.num_court_points = int(num_court_points)
        self.point_projection = CoordinateProjection(input_dim=2, dim=int(dim))

    def forward(self, court_kp: Tensor, court_vis: Tensor) -> Tensor:
        """Return camera court tokens with shape ``(..., D)``."""
        if court_kp.shape[-2:] != (self.num_court_points, 2):
            raise ValueError(
                "court_kp must end with "
                f"({self.num_court_points}, 2), got {tuple(court_kp.shape)}."
            )
        if court_vis.shape != court_kp.shape[:-1]:
            raise ValueError(
                "court_vis must match court_kp without its UV axis, got "
                f"court_vis={tuple(court_vis.shape)} and "
                f"court_kp={tuple(court_kp.shape)}."
            )
        visible = court_vis if court_vis.dtype == torch.bool else court_vis > 0
        masked_court = court_kp.masked_fill(~visible.unsqueeze(-1), 0.0)
        return self.point_projection(masked_court).mean(dim=-2)
