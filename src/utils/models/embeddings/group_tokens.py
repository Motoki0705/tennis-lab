"""Court-context group-token embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP


class _CourtContextGroupEmbedding(nn.Module):
    """Project one court/object pair without changing its leading-axis order."""

    def __init__(
        self,
        *,
        dim: int,
        num_court_tokens: int,
        group_input_dim: int,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.num_court_tokens = int(num_court_tokens)
        self.proj = CoordinateProjection(
            input_dim=self.num_court_tokens * 2 + int(group_input_dim),
            dim=int(dim),
        )
        self.invisible_token = invisible_token

    def _embed_group(
        self,
        *,
        court_flat: Tensor,
        group_flat: Tensor,
        group_vis: Tensor,
    ) -> Tensor:
        feat = self.proj(torch.cat((court_flat, group_flat), dim=-1))
        return apply_visibility_mask(feat, group_vis, self.invisible_token)


class CourtBallGroupEmbedding(_CourtContextGroupEmbedding):
    """Embed court and ball coordinates into one token per object element.

    The token visibility is controlled only by ``group_vis``. Callers should
    provide one visibility flag per output token. Leading dimensions are
    preserved exactly: the returned token at each position remains aligned
    with the caller-provided leading-axis element at that position.
    """

    def __init__(
        self,
        *,
        dim: int,
        invisible_token: InvisibleTokenEmbedding,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__(
            dim=dim,
            num_court_tokens=num_court_tokens,
            group_input_dim=2,
            invisible_token=invisible_token,
        )

    def forward(
        self,
        court_kp: Tensor,
        ball_uv: Tensor,
        group_vis: Tensor,
    ) -> Tensor:
        """Return one embedding token for each leading court/ball element."""
        court_flat = court_kp.flatten(-2)
        return self._embed_group(
            court_flat=court_flat,
            group_flat=ball_uv,
            group_vis=group_vis,
        )


class CourtPlayerGroupEmbedding(_CourtContextGroupEmbedding):
    """Embed court and player coordinates into one token per object element.

    The token visibility is controlled only by ``group_vis``. Callers should
    provide one visibility flag per output token. Leading dimensions are
    preserved exactly: the returned token at each position remains aligned
    with the caller-provided leading-axis element at that position.
    """

    def __init__(
        self,
        *,
        dim: int,
        invisible_token: InvisibleTokenEmbedding,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__(
            dim=dim,
            num_court_tokens=num_court_tokens,
            group_input_dim=NUM_HUMAN_KP * 2,
            invisible_token=invisible_token,
        )

    def forward(
        self,
        court_kp: Tensor,
        human_kp: Tensor,
        group_vis: Tensor,
    ) -> Tensor:
        """Return one embedding token for each leading court/player element."""
        court_flat = court_kp.flatten(-2)
        human_flat = human_kp.flatten(-2)
        return self._embed_group(
            court_flat=court_flat,
            group_flat=human_flat,
            group_vis=group_vis,
        )
