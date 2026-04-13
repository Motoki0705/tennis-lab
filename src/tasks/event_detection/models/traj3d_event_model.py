"""3D-trajectory-based event detection model (trajectory only, no velocity)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.utils.models.embeddings import Ball3DEmbedding, InvisibleTokenEmbedding
from src.tasks.event_detection.models.components.heads import EventLogitsHead

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Traj3DEventModel(nn.Module):
    """Predict event logits from 3D ball trajectory points."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        max_seq_len: int = 256,
        num_events: int = 2,
        ffn_dim: int | None = None,
        ffn_type: str = "swiglu",
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_events = int(num_events)

        if self.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=invisible_init_std
        )
        self.embed = Ball3DEmbedding(
            dim=self.hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                        rope_base=float(rope_theta),
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = EventLogitsHead(input_dim=self.hidden_dim, num_events=self.num_events, dropout=dropout)

        freqs_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=self.max_seq_len,
            base=float(rope_theta),
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> Traj3DEventModel:
        model_cfg = config.get("model", {}) or {}
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_events=int(model_cfg.get("num_events", 2)),
            ffn_dim=model_cfg.get("ffn_dim"),
            ffn_type=str(model_cfg.get("ffn_type", "swiglu")),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def forward(self, ball_pos_world: Tensor, seq_len: Tensor | None = None) -> Tensor:
        """Forward.

        Args:
            ball_pos_world: (B, T, 3)
            seq_len: (B,) optional sequence lengths.

        Returns:
            Logits (B, T, E)
        """
        if ball_pos_world.shape[1] > self.max_seq_len:
            ball_pos_world = ball_pos_world[:, : self.max_seq_len]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=self.max_seq_len)
        x = self.embed(ball_pos_world)
        S = x.shape[1]
        freqs_cis = self.freqs_cis[:S]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if seq_len is not None:
            B = x.shape[0]
            t = torch.arange(S, device=x.device)[None, :]
            key_padding_mask = t < seq_len.to(torch.long).view(B, 1)  # (B, S)
            attn_mask = key_padding_mask[:, None, :].expand(B, S, S)

        for block in self.blocks:
            x = block(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )
        x = self.final_norm(x)
        return self.head(x)


if __name__ == "__main__":
    model = Traj3DEventModel(hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32, num_events=2)
    ball_pos = torch.randn(2, 32, 3)
    seq_len = torch.tensor([32, 16])
    logits = model(ball_pos, seq_len=seq_len)
    assert logits.shape == (2, 32, 2)
    print("3d model smoke ok")
