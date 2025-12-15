from __future__ import annotations

import torch
from torch import nn

from .common import PositionalEncoding


class TrajectoryDeltaTransformer(nn.Module):
    """Transformer refiner mapping `[B, T, 2] -> [B, T, 2]` (delta in normalized coords)."""

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_dim: int = 2,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        return self.output_proj(h)

