"""Positionless-pose-query Transformer over DINOv3 patch tokens."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.court_detection.configuration import CourtQueryTaskEncoderConfig
from src.tasks.court_detection.models.query_encoder.contracts import (
    CourtEncoderTap,
    CourtTaskEncoderOutput,
    PatchTokenBatch,
)
from src.tasks.court_detection.models.query_encoder.rope import (
    PatchRoPEMultiheadAttention,
)


class CourtQueryEncoderBlock(nn.Module):
    def __init__(self, config: CourtQueryTaskEncoderConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        mlp_dim = int(round(hidden_dim * config.mlp_ratio))
        if mlp_dim <= 0:
            raise ValueError("Task-encoder MLP width must be positive.")
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = PatchRoPEMultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=config.num_heads,
            rope_dim=config.rope_dim,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, tokens: Tensor, *, grid_hw: tuple[int, int]) -> Tensor:
        tokens = tokens + self.attention_dropout(
            self.attention(self.attention_norm(tokens), grid_hw=grid_hw)
        )
        return tokens + self.mlp(self.mlp_norm(tokens))


class CourtQueryTaskEncoder(nn.Module):
    """Project patch tokens, prepend exactly one learned query, and expose taps."""

    def __init__(
        self,
        *,
        input_dim: int,
        config: CourtQueryTaskEncoderConfig,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Task-encoder input_dim must be positive.")
        self.input_dim = input_dim
        self.hidden_dim = config.hidden_dim
        self.depth = config.depth
        self.tap_indices = config.tap_indices
        self.patch_projection = nn.Linear(input_dim, self.hidden_dim)
        self.pose_query = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        nn.init.normal_(self.pose_query, std=0.02)
        self.blocks = nn.ModuleList(
            CourtQueryEncoderBlock(config) for _ in range(self.depth)
        )
        self.tap_norms = nn.ModuleDict(
            {str(index): nn.LayerNorm(self.hidden_dim) for index in self.tap_indices}
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, patch_batch: PatchTokenBatch) -> CourtTaskEncoderOutput:
        if patch_batch.embed_dim != self.input_dim:
            raise ValueError(
                "DINO patch embedding width disagrees with task-encoder input_dim."
            )
        patches = self.patch_projection(patch_batch.tokens)
        query = self.pose_query.expand(patch_batch.batch_size, -1, -1)
        tokens = torch.cat((query, patches), dim=1)
        captured: list[CourtEncoderTap] = []
        tap_set = set(self.tap_indices)
        for layer_index, block in enumerate(self.blocks):
            tokens = block(tokens, grid_hw=patch_batch.grid_hw)
            if layer_index in tap_set:
                tap_norm = self.tap_norms[str(layer_index)]
                captured.append(
                    CourtEncoderTap(
                        layer_index=layer_index,
                        patch_tokens=tap_norm(tokens[:, 1:]),
                        grid_hw=patch_batch.grid_hw,
                    )
                )
        if tuple(tap.layer_index for tap in captured) != self.tap_indices:
            raise RuntimeError("Task encoder did not produce every declared tap exactly once.")
        return CourtTaskEncoderOutput(
            pose_query=self.output_norm(tokens[:, 0]),
            taps=tuple(captured),
        )


__all__ = ["CourtQueryEncoderBlock", "CourtQueryTaskEncoder"]
