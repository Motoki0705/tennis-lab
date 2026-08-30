"""Tests for caller-ordered court/object group-token embeddings."""

from __future__ import annotations

import torch

from src.utils.models.embeddings import (
    CourtBallGroupEmbedding,
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
)


def test_ball_group_tokens_preserve_caller_axis_alignment() -> None:
    torch.manual_seed(31)
    embedding = CourtBallGroupEmbedding(
        dim=8,
        invisible_token=InvisibleTokenEmbedding(dim=8),
        num_court_tokens=14,
    ).eval()
    court = torch.rand(1, 2, 14, 2)
    balls_in_slot_order = torch.tensor([[[0.1, 0.2], [0.8, 0.9]]])
    visible = torch.ones(1, 2, dtype=torch.bool)

    with torch.no_grad():
        tokens = embedding(court, balls_in_slot_order, visible)
        expected = embedding.proj(
            torch.cat((court.flatten(-2), balls_in_slot_order), dim=-1)
        )

    torch.testing.assert_close(tokens, expected)


def test_player_group_tokens_preserve_caller_axis_alignment() -> None:
    torch.manual_seed(32)
    embedding = CourtPlayerGroupEmbedding(
        dim=8,
        invisible_token=InvisibleTokenEmbedding(dim=8),
        num_court_tokens=14,
    ).eval()
    court = torch.rand(1, 2, 14, 2)
    players_in_slot_order = torch.rand(1, 2, 17, 2)
    visible = torch.ones(1, 2, dtype=torch.bool)

    with torch.no_grad():
        tokens = embedding(court, players_in_slot_order, visible)
        expected = embedding.proj(
            torch.cat(
                (court.flatten(-2), players_in_slot_order.flatten(-2)),
                dim=-1,
            )
        )

    torch.testing.assert_close(tokens, expected)
