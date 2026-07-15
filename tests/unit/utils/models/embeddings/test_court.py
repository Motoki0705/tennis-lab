from __future__ import annotations

import pytest
import torch

from src.utils.models.embeddings import CourtLineMapEmbedding


def test_court_line_map_embedding_shape_and_backward() -> None:
    embedding = CourtLineMapEmbedding(dim=32, channels=(4, 8))
    court_line_map = torch.rand(2, 3, 1, 32, 48, requires_grad=True)

    output = embedding(court_line_map)
    output.square().mean().backward()

    assert output.shape == (2, 3, 32)
    assert court_line_map.grad is not None
    assert torch.isfinite(court_line_map.grad).all()


def test_court_line_map_embedding_handles_empty_map_without_nan() -> None:
    embedding = CourtLineMapEmbedding(dim=16, channels=(4, 8))
    output = embedding(torch.zeros(4, 1, 24, 40))
    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()


def test_court_line_map_embedding_rejects_wrong_contract() -> None:
    embedding = CourtLineMapEmbedding(dim=16, channels=(4, 8))
    with pytest.raises(ValueError, match="court_line_map must have shape"):
        embedding(torch.zeros(4, 2, 24, 40))


def test_court_line_map_embedding_is_spatially_adaptive() -> None:
    embedding = CourtLineMapEmbedding(dim=16, channels=(4, 8))
    assert embedding(torch.rand(2, 1, 32, 48)).shape == (2, 16)
    assert embedding(torch.rand(2, 1, 40, 64)).shape == (2, 16)
