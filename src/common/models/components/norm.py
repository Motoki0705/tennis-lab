"""Normalization layers for Transformer models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A computationally efficient alternative to LayerNorm that normalizes
    by the root mean square of the input, without centering.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension to normalize.
            eps: Small constant for numerical stability.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of any shape with last dim = dim.

        Returns:
            Normalized tensor of same shape.

        """
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
