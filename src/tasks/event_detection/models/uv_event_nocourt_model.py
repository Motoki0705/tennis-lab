"""UV-based event detection model without court tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.event_detection.models.components.heads import EventLogitsHead
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
)
from src.utils.models.embeddings import BallUVEmbedding, InvisibleTokenEmbedding

if TYPE_CHECKING:
    from omegaconf import DictConfig


class UVEventNoCourtModel(nn.Module):
    """Predict event logits from ball UV trajectory only."""

    uses_court_context = False

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        max_seq_len: int = 256,
        num_events: int = 2,
        ffn_dim: int | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        invisible_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_events = int(num_events)

        self._validate_init_args(hidden_dim=self.hidden_dim, num_heads=num_heads)
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim
        rope_base = float(rope_theta if rope_theta_time is None else rope_theta_time)

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=invisible_init_std
        )
        self.ball_embed = BallUVEmbedding(
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
                        rope_base=rope_base,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = EventLogitsHead(
            input_dim=self.hidden_dim, num_events=self.num_events, dropout=dropout
        )

        freqs_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=self.max_seq_len,
            base=rope_base,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _validate_init_args(*, hidden_dim: int, num_heads: int) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

    @classmethod
    def from_config(cls, config: DictConfig) -> UVEventNoCourtModel:
        model_cfg = config.get("model", {}) or {}
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_events=int(model_cfg.get("num_events", 2)),
            ffn_dim=model_cfg.get("ffn_dim"),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
        )

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            ball_uv: (B, T, 2)
            ball_vis: (B, T) or None
            ball_mask: (B, T) or None
            seq_len: (B,) optional sequence lengths

        Returns:
            Logits (B, T, E)
        """
        B, T, _ = ball_uv.shape
        if self.max_seq_len < T:
            ball_uv = ball_uv[:, : self.max_seq_len]
            if ball_vis is not None:
                ball_vis = ball_vis[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=self.max_seq_len)
            T = self.max_seq_len

        x = self.ball_embed(ball_uv, ball_vis)
        freqs_cis = self.freqs_cis[:T]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if ball_mask is None and seq_len is not None:
            t = torch.arange(T, device=x.device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(B, 1)
        if ball_mask is not None:
            key_padding_mask = ball_mask > 0
            attn_mask = key_padding_mask[:, None, :].expand(B, T, T)

        for block in self.blocks:
            x = block(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)
        return self.head(x)
