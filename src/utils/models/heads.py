"""Shared output head modules for transformer-based models.

Provides a generic MLP head that captures the layer-construction pattern
duplicated across the PLCS heads (``PositionHead``/``RotationHead``/
``CanonicalPoseHead``) and the BLCS heads (``Trajectory3DHead``/
``VelocityHead``).
"""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torch import Tensor


class MLPHead(nn.Module):
    """Generic MLP head.

    Builds ``[Linear -> LayerNorm -> GELU -> Dropout] x num_layers -> Linear``
    and exposes it as ``self.mlp`` (a ``nn.Sequential``).  The attribute is
    named ``mlp`` so that subclasses reuse the exact ``mlp.*`` state_dict keys
    produced by the original task-specific heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the MLP head.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension of the final linear projection.
            num_layers: Number of ``Linear -> LayerNorm -> GELU -> Dropout``
                blocks before the final linear projection.
            dropout: Dropout probability.
        """
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        hidden_dim = int(hidden_dim)

        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP to the input features."""
        return cast(Tensor, self.mlp(x))
