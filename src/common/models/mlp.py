"""MLP layers for Transformer models."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLUMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    A gated linear unit with SiLU (Swish) activation, commonly used
    in modern LLM architectures like LLaMA.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        """Initialize SwiGLU MLP.

        Args:
            dim: Input and output dimension.
            ffn_dim: Hidden dimension (intermediate size).
            dropout: Dropout probability.

        """
        super().__init__()
        self.wu = nn.Linear(dim, ffn_dim, bias=False)
        self.wg = nn.Linear(dim, ffn_dim, bias=False)
        self.wd = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).

        """
        h = self.wu(x) * F.silu(self.wg(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.wd(h)
