"""Court-context group-token embeddings."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.utils.models.embeddings.court import CourtLineMapEmbedding
from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import (
    CoordinateProjection,
    apply_visibility_mask,
)
from src.utils.schema.court import NUM_COURT_KP
from src.utils.schema.player import NUM_HUMAN_KP


def _flatten_court_kp(court_kp: Tensor, *, num_court_tokens: int) -> Tensor:
    if court_kp.dim() >= 2 and tuple(court_kp.shape[-2:]) == (num_court_tokens, 2):
        return court_kp.reshape(*court_kp.shape[:-2], num_court_tokens * 2)
    if court_kp.dim() >= 1 and court_kp.shape[-1] == num_court_tokens * 2:
        return court_kp
    raise ValueError(
        "court_kp must have shape "
        f"(..., {num_court_tokens}, 2) or (..., {num_court_tokens * 2}).",
    )


def _flatten_ball_uv(ball_uv: Tensor) -> Tensor:
    if ball_uv.dim() >= 2 and tuple(ball_uv.shape[-2:]) == (1, 2):
        return ball_uv.reshape(*ball_uv.shape[:-2], 2)
    if ball_uv.dim() >= 1 and ball_uv.shape[-1] == 2:
        return ball_uv
    raise ValueError("ball_uv must have shape (..., 2) or (..., 1, 2).")


def _flatten_human_kp(human_kp: Tensor) -> Tensor:
    if human_kp.dim() >= 2 and tuple(human_kp.shape[-2:]) == (NUM_HUMAN_KP, 2):
        return human_kp.reshape(*human_kp.shape[:-2], NUM_HUMAN_KP * 2)
    if human_kp.dim() >= 1 and human_kp.shape[-1] == NUM_HUMAN_KP * 2:
        return human_kp
    raise ValueError(
        "human_kp must have shape (..., NUM_HUMAN_KP, 2) or (..., NUM_HUMAN_KP * 2).",
    )


def _normalize_group_visibility(
    group_vis: Tensor | None,
    *,
    expected_shape: torch.Size,
) -> Tensor | None:
    if group_vis is None:
        return None

    mask = group_vis if group_vis.dtype == torch.bool else group_vis > 0
    if tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            "group_vis must match the group token leading shape "
            f"{tuple(expected_shape)}, got {tuple(mask.shape)}.",
        )
    return mask


class _CourtContextGroupEmbedding(nn.Module):
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
        group_vis: Tensor | None,
    ) -> Tensor:
        if tuple(court_flat.shape[:-1]) != tuple(group_flat.shape[:-1]):
            raise ValueError(
                "court and group inputs must share the same leading dimensions."
            )

        feat = self.proj(torch.cat((court_flat, group_flat), dim=-1))
        visible = _normalize_group_visibility(
            group_vis,
            expected_shape=court_flat.shape[:-1],
        )
        return apply_visibility_mask(feat, visible, self.invisible_token)


class CourtBallGroupEmbedding(_CourtContextGroupEmbedding):
    """Embed court and ball coordinates into one token per camera/time element.

    The token visibility is controlled only by ``group_vis``. Callers should
    provide one visibility flag per output token, for example shape ``(B, T)``.
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
        group_vis: Tensor | None = None,
    ) -> Tensor:
        """Return one embedding token for each leading court/ball element."""
        court_flat = _flatten_court_kp(court_kp, num_court_tokens=self.num_court_tokens)
        ball_flat = _flatten_ball_uv(ball_uv)
        return self._embed_group(
            court_flat=court_flat,
            group_flat=ball_flat,
            group_vis=group_vis,
        )


class CourtPlayerGroupEmbedding(_CourtContextGroupEmbedding):
    """Embed court and player coordinates into one token per camera/time element.

    The token visibility is controlled only by ``group_vis``. Callers should
    provide one visibility flag per output token, for example shape ``(B, T)``.
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
        group_vis: Tensor | None = None,
    ) -> Tensor:
        """Return one embedding token for each leading court/player element."""
        court_flat = _flatten_court_kp(court_kp, num_court_tokens=self.num_court_tokens)
        human_flat = _flatten_human_kp(human_kp)
        return self._embed_group(
            court_flat=court_flat,
            group_flat=human_flat,
            group_vis=group_vis,
        )


class _CourtLineMapObjectGroupEmbedding(nn.Module):
    """Build separate court/object tokens with a learned token-type embedding."""

    def __init__(
        self,
        *,
        dim: int,
        line_map_channels: tuple[int, ...],
        object_input_dim: int,
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__()
        self.court_embed = CourtLineMapEmbedding(
            dim=dim,
            channels=line_map_channels,
        )
        self.object_proj = CoordinateProjection(input_dim=object_input_dim, dim=dim)
        self.token_type = nn.Embedding(2, dim)
        self.invisible_token = invisible_token

    def _embed(
        self,
        *,
        court_line_map: Tensor,
        object_flat: Tensor,
        object_vis: Tensor | None,
    ) -> Tensor:
        court_token = self.court_embed(court_line_map)
        if tuple(court_token.shape[:-1]) != tuple(object_flat.shape[:-1]):
            raise ValueError("court and object inputs must share leading dimensions.")
        object_token = self.object_proj(object_flat)
        object_token = apply_visibility_mask(
            object_token,
            object_vis,
            self.invisible_token,
        )
        tokens = torch.stack((court_token, object_token), dim=-2)
        type_ids = torch.arange(2, device=tokens.device)
        type_embedding = self.token_type(type_ids).to(dtype=tokens.dtype)
        return cast(Tensor, tokens + type_embedding)


class CourtLineMapBallGroupEmbedding(_CourtLineMapObjectGroupEmbedding):
    """Return ``[court token, ball token]`` for each camera/time element."""

    def __init__(
        self,
        *,
        dim: int,
        line_map_channels: tuple[int, ...],
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__(
            dim=dim,
            line_map_channels=line_map_channels,
            object_input_dim=2,
            invisible_token=invisible_token,
        )

    def forward(
        self,
        court_line_map: Tensor,
        ball_uv: Tensor,
        ball_vis: Tensor | None = None,
    ) -> Tensor:
        return self._embed(
            court_line_map=court_line_map,
            object_flat=_flatten_ball_uv(ball_uv),
            object_vis=ball_vis,
        )


class CourtLineMapPlayerGroupEmbedding(_CourtLineMapObjectGroupEmbedding):
    """Return ``[court token, player token]`` for each camera/time element."""

    def __init__(
        self,
        *,
        dim: int,
        line_map_channels: tuple[int, ...],
        invisible_token: InvisibleTokenEmbedding,
    ) -> None:
        super().__init__(
            dim=dim,
            line_map_channels=line_map_channels,
            object_input_dim=NUM_HUMAN_KP * 2,
            invisible_token=invisible_token,
        )

    def forward(
        self,
        court_line_map: Tensor,
        human_kp: Tensor,
        player_vis: Tensor | None = None,
    ) -> Tensor:
        return self._embed(
            court_line_map=court_line_map,
            object_flat=_flatten_human_kp(human_kp),
            object_vis=player_vis,
        )
