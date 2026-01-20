"""UV-based event detection model.

Architecture is aligned with src/blcs/models/blcs_model.py:
- Tokenize court keypoints + ball UV as tokens
- Decoder-only Transformer blocks with RoPE
- Predict per-frame event logits from ball tokens
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.common.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    YaRNConfig,
    precompute_freqs_cis,
)
from src.evnet_detection.models.components.embeddings import BallUVTokenEmbedding, CourtTokenEmbedding
from src.evnet_detection.models.components.heads import EventLogitsHead
from src.utils.geometry import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class UVEventModel(nn.Module):
    """Predict event logits from ball UV trajectory and court keypoints."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        causal: bool = False,
        max_seq_len: int = 256,
        num_events: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_events = int(num_events)
        self.causal = bool(causal)

        if self.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim
        max_tokens = int(NUM_COURT_KP + self.max_seq_len)

        self.court_embed = CourtTokenEmbedding(dim=self.hidden_dim, dropout=dropout)
        self.ball_embed = BallUVTokenEmbedding(dim=self.hidden_dim, dropout=dropout)
        self.type_embed = nn.Embedding(2, self.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=int((8 * self.hidden_dim) / 3),
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                        rope_base=float(rope_theta),
                        yarn=yarn,
                        use_moe=False,
                        moe_config=None,
                    )
                )
                for _ in range(int(num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = EventLogitsHead(input_dim=self.hidden_dim, num_events=self.num_events, dropout=dropout)

        freqs_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=max_tokens,
            base=float(rope_theta),
            yarn=yarn,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> UVEventModel:
        model_cfg = config.get("model", {}) or {}
        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            causal=bool(model_cfg.get("causal", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_events=int(model_cfg.get("num_events", 2)),
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            ball_uv: (B, T, 2)
            court_kp: (B, 20, 2)
            ball_mask: (B, T) or None
            court_vis: (B, 20) or None

        Returns:
            Logits (B, T, E)
        """
        B, T, _ = ball_uv.shape
        if T > self.max_seq_len:
            ball_uv = ball_uv[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=self.max_seq_len)
            T = self.max_seq_len

        court_tokens = self.court_embed(court_kp, court_vis)  # (B, 20, D)
        ball_tokens = self.ball_embed(ball_uv, ball_mask)  # (B, T, D)

        court_type = self.type_embed(
            torch.zeros(NUM_COURT_KP, device=ball_uv.device, dtype=torch.long)
        )[None, :, :]
        ball_type = self.type_embed(
            torch.ones(T, device=ball_uv.device, dtype=torch.long)
        )[None, :, :]

        x = torch.cat([court_tokens + court_type, ball_tokens + ball_type], dim=1)  # (B, S, D)
        S = x.shape[1]

        key_padding_mask: Tensor | None = None
        if ball_mask is not None or seq_len is not None:
            court_valid = torch.ones(B, NUM_COURT_KP, device=x.device, dtype=torch.bool)
            if ball_mask is None:
                ball_valid = torch.ones(B, T, device=x.device, dtype=torch.bool)
            else:
                ball_valid = ball_mask > 0
            if seq_len is not None:
                t = torch.arange(T, device=x.device)[None, :]
                ball_valid = ball_valid & (t < seq_len.to(torch.long).view(B, 1))
            key_padding_mask = torch.cat([court_valid, ball_valid], dim=1)  # (B, S)

        if S > self.freqs_cis.shape[0]:
            raise ValueError("Sequence length exceeds max_seq_len; increase model.max_seq_len.")
        freqs_cis = self.freqs_cis[:S]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)

        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, :].expand(B, S, S)

        residual = None
        for block in self.blocks:
            x, residual = block(
                x,
                residual,
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                is_causal=self.causal,
            )

        if residual is None:
            x = self.final_norm(x)
        else:
            x, _ = self.final_norm(x, residual)

        ball_h = x[:, NUM_COURT_KP:, :]
        return self.head(ball_h)


if __name__ == "__main__":
    model = UVEventModel(hidden_dim=64, num_layers=2, num_heads=4, max_seq_len=32, num_events=2)
    ball_uv = torch.rand(2, 32, 2)
    ball_mask = torch.ones(2, 32)
    court_kp = torch.rand(2, 20, 2)
    court_vis = torch.ones(2, 20)
    seq_len = torch.tensor([32, 16])
    logits = model(ball_uv, court_kp, ball_mask, court_vis, seq_len=seq_len)
    assert logits.shape == (2, 32, 2)
    print("uv model smoke ok")
