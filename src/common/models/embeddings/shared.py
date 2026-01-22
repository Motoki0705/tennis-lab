"""Shared embedding utilities for visibility-aware tokenization."""

from __future__ import annotations

import torch
from torch import nn


class InvisibleTokenEmbedding(nn.Module):
    """Learnable token used to represent invisible observations.

    Args:
        dim: Embedding dimension.
        init_std: Truncated normal initialization std.
    """

    def __init__(self, *, dim: int, init_std: float = 0.02) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.empty(int(dim)))
        nn.init.trunc_normal_(self.token, std=float(init_std))

    def forward(self) -> torch.Tensor:
        """Return the invisible token embedding.

        Returns:
            Tensor of shape (D,).
        """
        return self.token


if __name__ == "__main__":
    torch.manual_seed(0)
    emb = InvisibleTokenEmbedding(dim=16)
    tok = emb()
    assert tok.shape == (16,)
    print("InvisibleTokenEmbedding smoke ok")
