from __future__ import annotations

import torch

from src.utils.models.embeddings import (
    CourtLineMapBallGroupEmbedding,
    InvisibleTokenEmbedding,
)


def test_line_map_group_orders_ball_before_court_tokens_without_type_parameters() -> None:
    embedding = CourtLineMapBallGroupEmbedding(
        dim=8,
        line_map_channels=(4,),
        num_line_map_tokens=4,
        invisible_token=InvisibleTokenEmbedding(dim=8),
    )
    ball_uv = torch.rand(2, 2)

    tokens = embedding(
        court_line_map=torch.rand(2, 1, 24, 40),
        ball_uv=ball_uv,
        ball_vis=torch.ones(2),
    )

    assert tokens.shape == (2, 5, 8)
    assert torch.allclose(tokens[:, 0], embedding.object_proj(ball_uv))
    assert not any("token_type" in name for name, _ in embedding.named_parameters())
