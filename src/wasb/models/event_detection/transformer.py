"""Transformer encoder for per-frame trajectory event detection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _sinusoidal_positional_encoding(length: int, d_model: int, device: torch.device) -> Tensor:
    pe = torch.zeros(length, d_model, device=device, dtype=torch.float32)
    position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class TrajectoryEventTransformer(nn.Module):
    """Transformer encoder that predicts an event class per timestep."""

    _pos_embed_cache: Tensor

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 128,
        num_classes: int = 3,
        max_len: int = 512,
        positional_encoding: str = "sin",
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2")
        if max_len <= 0:
            raise ValueError("max_len must be positive")

        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.positional_encoding = str(positional_encoding)

        self.input_proj = nn.Linear(2, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

        if self.positional_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(self.max_len, self.d_model))
            nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        elif self.positional_encoding == "sin":
            self.register_buffer(
                "_pos_embed_cache", torch.empty(0), persistent=False
            )
        else:
            raise ValueError("positional_encoding must be 'sin' or 'learned'")

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.head = nn.Sequential(
            nn.Linear(self.d_model, int(mlp_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden_dim), int(num_classes)),
        )

    def _positional_embed(self, length: int, device: torch.device) -> Tensor:
        if length > self.max_len:
            raise ValueError(f"sequence length {length} exceeds max_len={self.max_len}")

        if self.positional_encoding == "learned":
            return self.pos_embed[:length].to(device=device)

        cache: Tensor = self._pos_embed_cache
        if cache.numel() == 0 or cache.shape[0] < length or cache.device != device:
            cache = _sinusoidal_positional_encoding(length, self.d_model, device)
            self._pos_embed_cache = cache
        return cache[:length]

    def forward(self, xy_norm: Tensor, *, key_padding_mask: Tensor | None = None) -> Tensor:
        """Predict per-timestep logits.

        Args:
            xy_norm: Normalized coordinates, shape (B, T, 2).
            key_padding_mask: Optional boolean mask (B, T) where True indicates
                timesteps to ignore (e.g., invisible ball).

        """
        if xy_norm.dim() != 3 or xy_norm.shape[-1] != 2:
            raise ValueError(f"xy_norm must have shape (B, T, 2), got {tuple(xy_norm.shape)}")

        b, t, _ = xy_norm.shape
        x = self.input_proj(xy_norm)  # (B, T, D)
        x = x + self._positional_embed(t, xy_norm.device).unsqueeze(0).expand(b, t, -1)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.head(x)  # (B, T, C)
        return logits
