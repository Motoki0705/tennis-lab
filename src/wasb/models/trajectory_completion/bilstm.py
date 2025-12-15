from __future__ import annotations

import torch
from torch import nn


class TrajectoryBiLSTM(nn.Module):
    """BiLSTM architecture mapping `[B, T, 2] -> [B, T, 2]` (normalized coords)."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        input_dim: int = 2,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        return self.output_proj(h)

