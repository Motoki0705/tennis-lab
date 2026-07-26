"""Tests for ID-ordered court/object group-token embeddings."""

from __future__ import annotations

import torch

from src.utils.models.embeddings import (
    CourtBallGroupEmbedding,
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
)


def test_ball_group_tokens_preserve_object_id_axis_alignment() -> None:
    torch.manual_seed(31)
    embedding = CourtBallGroupEmbedding(
        dim=8,
        invisible_token=InvisibleTokenEmbedding(dim=8),
        num_court_tokens=14,
    ).eval()
    court = torch.rand(1, 2, 14, 2)
    balls_by_object_id = torch.tensor([[[0.1, 0.2], [0.8, 0.9]]])
    visible = torch.ones(1, 2, dtype=torch.bool)

    with torch.no_grad():
        ordered = embedding(court, balls_by_object_id, visible)
        expected_by_object_id = embedding.proj(
            torch.cat((court.flatten(-2), balls_by_object_id), dim=-1)
        )

    torch.testing.assert_close(ordered, expected_by_object_id)


def test_player_group_tokens_preserve_object_id_axis_alignment() -> None:
    torch.manual_seed(32)
    embedding = CourtPlayerGroupEmbedding(
        dim=8,
        invisible_token=InvisibleTokenEmbedding(dim=8),
        num_court_tokens=14,
    ).eval()
    court = torch.rand(1, 2, 14, 2)
    players_by_object_id = torch.rand(1, 2, 17, 2)
    visible = torch.ones(1, 2, dtype=torch.bool)

    with torch.no_grad():
        ordered = embedding(court, players_by_object_id, visible)
        expected_by_object_id = embedding.proj(
            torch.cat(
                (court.flatten(-2), players_by_object_id.flatten(-2)),
                dim=-1,
            )
        )

    torch.testing.assert_close(ordered, expected_by_object_id)
