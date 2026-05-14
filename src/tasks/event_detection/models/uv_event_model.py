"""UV-based event detection model.

Architecture is aligned with src/tasks/blcs/models/blcs_model.py:
- Tokenize court keypoints + ball UV as tokens
- Decoder-only Transformer blocks with RoPE
- Predict per-frame event logits from ball tokens
"""

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
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import (
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.schema.court import NUM_COURT_KP

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
        rope_theta_time: float | None = None,
        rope_theta_camera: float | None = None,
        rope_theta_type: float = 100.0,
        max_seq_len: int = 256,
        num_events: int = 2,
        ffn_dim: int | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        invisible_init_std: float = 0.02,
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_events = int(num_events)
        self.num_court_tokens = int(num_court_tokens)

        self._validate_init_args(hidden_dim=self.hidden_dim, num_heads=num_heads)
        head_dim = self.hidden_dim // int(num_heads)
        rope_dim = head_dim
        self.rope_theta = float(rope_theta)
        self.rope_bases = (
            float(self.rope_theta if rope_theta_time is None else rope_theta_time),
            float(self.rope_theta if rope_theta_camera is None else rope_theta_camera),
            float(rope_theta_type),
        )

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=invisible_init_std
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=self.hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
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
                        rope_base=self.rope_theta,
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

        freqs_cis = precompute_freqs_cis_nd(
            dim=rope_dim,
            pos=self._build_rope_positions(),
            base=self.rope_bases,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @staticmethod
    def _validate_init_args(*, hidden_dim: int, num_heads: int) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

    @classmethod
    def from_config(cls, config: DictConfig) -> UVEventModel:
        model_cfg = config.get("model", {}) or {}
        data_cfg = config.get("data", {}) or {}
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        return cls(
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_layers", 6)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            rope_theta_type=model_cfg.get("rope_theta_type", 100.0),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            num_events=int(model_cfg.get("num_events", 2)),
            ffn_dim=model_cfg.get("ffn_dim"),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            num_court_tokens=int(data_cfg.get("num_court_kp", NUM_COURT_KP)),
        )

    def _build_rope_positions(self) -> Tensor:
        """Build 3-axis RoPE positions for `[court, ball]` tokens."""
        court_idx = torch.arange(self.num_court_tokens, dtype=torch.long)
        ball_time = torch.arange(self.max_seq_len, dtype=torch.long) + 1

        court_pos = torch.stack(
            [
                torch.zeros_like(court_idx),
                court_idx,
                torch.zeros_like(court_idx),
            ],
            dim=-1,
        )
        ball_pos = torch.stack(
            [
                ball_time,
                torch.zeros_like(ball_time),
                torch.ones_like(ball_time),
            ],
            dim=-1,
        )
        return torch.cat([court_pos, ball_pos], dim=0)

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Forward.

        Args:
            ball_uv: (B, T, 2)
            court_kp: (B, K, 2)
            ball_vis: (B, T) or None
            ball_mask: (B, T) or None
            court_vis: (B, K) or None

        Returns:
            Logits (B, T, E)
        """
        B, T, _ = ball_uv.shape
        ball_uv, ball_vis, ball_mask, seq_len, T = self._clip_sequence_inputs(
            ball_uv=ball_uv,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            seq_len=seq_len,
            seq_len_in=T,
        )

        court_tokens = self.court_embed(court_kp, court_vis)  # (B, K, D)
        self._validate_court_tokens(court_tokens)
        ball_tokens = self.ball_embed(ball_uv, ball_vis)  # (B, T, D)

        K = self.num_court_tokens
        x = torch.cat([court_tokens, ball_tokens], dim=1)  # (B, S, D)
        S = x.shape[1]

        key_padding_mask: Tensor | None = None
        if ball_mask is None and seq_len is not None:
            t = torch.arange(T, device=x.device)[None, :]
            ball_mask = t < seq_len.to(torch.long).view(B, 1)
        if ball_mask is not None:
            court_valid = torch.ones(B, K, device=x.device, dtype=torch.bool)
            ball_valid = ball_mask > 0
            key_padding_mask = torch.cat([court_valid, ball_valid], dim=1)  # (B, S)

        freqs_cis = self._freqs_for_sequence(x=x, seq_len=S)

        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, :].expand(B, S, S)

        for block in self.blocks:
            x = block(
                x,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
            )

        x = self.final_norm(x)

        ball_h = x[:, K:, :]
        return cast(Tensor, self.head(ball_h))

    def _clip_sequence_inputs(
        self,
        *,
        ball_uv: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        seq_len: Tensor | None,
        seq_len_in: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, int]:
        if self.max_seq_len < seq_len_in:
            ball_uv = ball_uv[:, : self.max_seq_len]
            if ball_vis is not None:
                ball_vis = ball_vis[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            if seq_len is not None:
                seq_len = torch.clamp(seq_len, max=self.max_seq_len)
            seq_len_in = self.max_seq_len
        return ball_uv, ball_vis, ball_mask, seq_len, seq_len_in

    def _validate_court_tokens(self, court_tokens: Tensor) -> None:
        if court_tokens.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tokens.shape[1]}"
            )

    def _freqs_for_sequence(self, *, x: Tensor, seq_len: int) -> Tensor:
        freqs_cis = cast(Tensor, self.freqs_cis)
        if freqs_cis.shape[0] < seq_len:
            raise ValueError(
                "Sequence length exceeds max_seq_len; increase model.max_seq_len."
            )
        freqs_cis = freqs_cis[:seq_len]
        if freqs_cis.device != x.device:
            freqs_cis = freqs_cis.to(x.device)
        return freqs_cis


if __name__ == "__main__":
    num_court_tokens = 12
    model = UVEventModel(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        num_events=2,
        num_court_tokens=num_court_tokens,
    )
    ball_uv = torch.rand(2, 32, 2)
    ball_vis = torch.ones(2, 32)
    ball_mask = torch.ones(2, 32)
    court_kp = torch.rand(2, num_court_tokens, 2)
    court_vis = torch.ones(2, num_court_tokens)
    seq_len = torch.tensor([32, 16])
    logits = model(
        ball_uv,
        court_kp,
        ball_vis=ball_vis,
        ball_mask=ball_mask,
        court_vis=court_vis,
        seq_len=seq_len,
    )
    assert logits.shape == (2, 32, 2)
    print("uv model smoke ok")
