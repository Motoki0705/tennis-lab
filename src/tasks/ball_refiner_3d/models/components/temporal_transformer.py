"""Shared RoPE trunk with stable event-head checkpoint parameter names."""

from __future__ import annotations

from torch import Tensor, nn

from src.tasks.ball_refiner_3d.configuration.model import ModelConfig
from src.tasks.ball_refiner_3d.model_io.contracts import PhysicsOutput, RefinerOutput
from src.tasks.ball_refiner_3d.models.components.heads import (
    event_head,
    trajectory_head,
)
from src.tasks.ball_refiner_3d.models.components.physics_heads import (
    FieldHead,
    SegmentHead,
)
from src.tasks.ball_refiner_3d.physics.reconstruction import segment_layout
from src.utils.models.components import (
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)

# Longest sequence in frames (68 s at 60 fps): whole clips are one forward.
MAX_FRAMES = 4096


class TemporalTransformer(nn.Module):
    """Both generators share a trunk; their forward signatures remain separate.

    Sequences have any length up to :data:`MAX_FRAMES`; padded frames are
    excluded as attention keys.
    """

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
        self.field_head = FieldHead(config.width) if config.physics_heads else None
        self.segment_head = SegmentHead(config.width) if config.physics_heads else None
        self.freqs_cis: Tensor
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                dim=config.rope_dim, seqlen=MAX_FRAMES, base=config.rope_theta
            ),
            persistent=False,
        )

    def trunk(
        self, features: Tensor, padding: Tensor, time_features: Tensor | None = None
    ) -> Tensor:
        token = self.input_dropout(self.input(features))
        if time_features is not None:
            token = token + time_features[:, None]
        length = features.shape[1]
        if length > MAX_FRAMES:
            raise ValueError(f"Sequences are limited to {MAX_FRAMES} frames")
        keep = (~padding)[:, None, :].expand(-1, length, -1)
        for block in self.blocks:
            token = block(token, freqs_cis=self.freqs_cis[:length], attn_mask=keep)
        return token

    def heads(
        self, tokens: Tensor, padding: Tensor, segment: Tensor | None
    ) -> RefinerOutput:
        coordinates, events = self.output(tokens), self.event_head(tokens)
        if self.field_head is None or self.segment_head is None:
            if segment is not None:
                raise ValueError("A segmentation requires physics heads")
            return RefinerOutput(coordinates, events)
        field, surface = self.field_head(tokens, padding)
        states = None
        if segment is not None:
            layout = segment_layout(segment)
            states = self.segment_head(tokens, layout.member, layout.start)
        return RefinerOutput(coordinates, events, PhysicsOutput(field, surface, states))

    def encode(
        self,
        features: Tensor,
        padding: Tensor,
        time_features: Tensor | None = None,
        segment: Tensor | None = None,
    ) -> RefinerOutput:
        return self.heads(
            self.trunk(features, padding, time_features), padding, segment
        )
