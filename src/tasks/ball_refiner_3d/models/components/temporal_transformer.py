"""Shared RoPE trunk with stable event-head checkpoint parameter names."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.model_io.contracts import RefinerOutput
from src.tasks.ball_refiner_3d.models.components.heads import (
    event_head,
    trajectory_head,
)
from src.utils.models.components import (
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)


class TemporalTransformer(nn.Module):
    """Both generators share a trunk; their forward signatures remain separate."""

    def __init__(self, config: ModelConfig, *, input_channels: int) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Linear(input_channels, config.width)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=config.width,
                        n_heads=config.heads,
                        ffn_dim=config.ffn_dim,
                        head_dim=config.width // config.heads,
                        rope_dim=config.rope_dim,
                        attn_dropout=config.dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=config.rope_theta,
                        ffn_type=config.ffn_type,
                    )
                )
                for _ in range(config.layers)
            ]
        )
        self.output = trajectory_head(config.width)
        self.event_head = event_head(config.width)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=config.rope_dim,
                seqlen=config.window_length,
                base=config.rope_theta,
            ),
            persistent=False,
        )

    def encode(
        self, features: Tensor, time_features: Tensor | None = None
    ) -> RefinerOutput:
        token = self.input_dropout(self.input(features))
        if time_features is not None:
            token = token + time_features[:, None]
        length = features.shape[1]
        keep = torch.ones(
            (len(features), length, length), dtype=torch.bool, device=features.device
        )
        for block in self.blocks:
            token = block(token, freqs_cis=self.freqs_cis[:length], attn_mask=keep)
        return RefinerOutput(self.output(token), self.event_head(token))
