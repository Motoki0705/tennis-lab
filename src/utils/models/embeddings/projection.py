"""Shared projection and masking helpers for coordinate embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.utils.models.components.norm import RMSNorm
from src.utils.models.embeddings.invisible_embedding import InvisibleTokenEmbedding


class CoordinateProjection(nn.Module):
    """Project coordinate features with the shared two-layer stack."""

    def __init__(self, *, input_dim: int, dim: int) -> None:
        super().__init__()
        input_dim = int(input_dim)
        dim = int(dim)
        if dim % 2 != 0:
            raise ValueError(
                "CoordinateProjection requires an even dim so the hidden layer can use dim / 2.",
            )
        hidden_dim = dim // 2
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            RMSNorm(dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project coordinate features into the model embedding space."""
        projected: Tensor = self.layers(x)
        return projected


def apply_visibility_mask(
    feat: Tensor,
    visible: Tensor,
    invisible_token: InvisibleTokenEmbedding,
) -> Tensor:
    """Replace non-visible elements using a boundary-validated mask tensor."""
    mask = (visible > 0).unsqueeze(-1)

    inv = invisible_token().to(dtype=feat.dtype, device=feat.device)
    inv = inv.view(*([1] * (feat.dim() - 1)), -1).expand_as(feat)
    return torch.where(mask, feat, inv)
