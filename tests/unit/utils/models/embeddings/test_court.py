from __future__ import annotations

import pytest
import torch

from src.utils.models.embeddings import CourtLineEmbedding


def test_court_line_embedding_shape_and_backward() -> None:
    embedding = CourtLineEmbedding(dim=32, max_court_lines=12)
    court_lines = torch.randn(2, 3, 12, 4, requires_grad=True)

    output = embedding(court_lines)
    output.square().mean().backward()

    assert output.shape == (2, 3, 32)
    assert court_lines.grad is not None
    assert torch.isfinite(court_lines.grad).all()


def test_court_line_embedding_handles_no_lines_without_nan() -> None:
    embedding = CourtLineEmbedding(dim=16, max_court_lines=5)
    output = embedding(torch.zeros(4, 5, 4))
    assert output.shape == (4, 16)
    assert torch.isfinite(output).all()


def test_court_line_embedding_rejects_wrong_contract() -> None:
    embedding = CourtLineEmbedding(dim=16, max_court_lines=5)
    with pytest.raises(ValueError, match="court_lines must have shape"):
        embedding(torch.zeros(4, 6, 4))
