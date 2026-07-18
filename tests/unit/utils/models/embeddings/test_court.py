from __future__ import annotations

import torch

from src.utils.models.embeddings import CourtCameraEmbedding


def test_camera_embedding_is_invariant_to_court_point_order() -> None:
    torch.manual_seed(4)
    embedding = CourtCameraEmbedding(dim=16, num_court_points=14).eval()
    court = torch.rand(2, 3, 14, 2)
    visible = torch.rand(2, 3, 14) > 0.2
    permutation = torch.randperm(14)

    with torch.no_grad():
        expected = embedding(court, visible)
        actual = embedding(court[..., permutation, :], visible[..., permutation])

    torch.testing.assert_close(actual, expected)


def test_invisible_court_coordinates_are_zeroed_before_embedding() -> None:
    torch.manual_seed(5)
    embedding = CourtCameraEmbedding(dim=16, num_court_points=14).eval()
    court = torch.rand(1, 14, 2)
    visible = torch.ones(1, 14, dtype=torch.bool)
    visible[:, 3] = False
    changed = court.clone()
    changed[:, 3] = torch.nan

    with torch.no_grad():
        expected = embedding(court, visible)
        actual = embedding(changed, visible)

    torch.testing.assert_close(actual, expected)
