"""Small-token self-attention fusion for court and ball image keypoints."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.configuration import PointFusionConfig
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    build_self_attn_mask,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import InvisibleTokenEmbedding
from src.utils.models.embeddings.projection import CoordinateProjection


class CourtBallPointFusion(nn.Module):
    """Fuse ``[court_0..K, ball_0..P]`` before projecting ball tokens.

    Court indices occupy the first RoPE axis and ball-candidate indices occupy
    the second.  Attention runs in the small ``token_dim`` space; only the
    resulting ball tokens are projected to the downstream model dimension.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        num_court_points: int,
        config: PointFusionConfig,
        invisible_init_std: float,
    ) -> None:
        super().__init__()
        self.token_dim = int(config.token_dim)
        self.num_heads = int(config.num_heads)
        self.num_layers = int(config.num_layers)
        self.num_court_points = int(num_court_points)
        if self.token_dim <= 0 or self.num_heads <= 0 or self.num_layers <= 0:
            raise ValueError(
                "point_fusion token_dim, num_heads, and num_layers must be positive."
            )
        if self.token_dim % self.num_heads != 0:
            raise ValueError("point_fusion token_dim must be divisible by num_heads.")
        head_dim = self.token_dim // self.num_heads
        self.rope_dim = config.rope_dim
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "point_fusion rope_dim must be even and no larger than its head dim."
            )

        self.coordinate_projection = CoordinateProjection(
            input_dim=2,
            dim=self.token_dim,
        )
        self.token_type_embedding = nn.Parameter(torch.empty(2, self.token_dim))
        nn.init.trunc_normal_(self.token_type_embedding, std=0.02)
        self.invisible_ball_token = InvisibleTokenEmbedding(
            dim=self.token_dim,
            init_std=invisible_init_std,
        )
        block_config = TransformerBlockConfig(
            dim=self.token_dim,
            n_heads=self.num_heads,
            ffn_dim=config.ffn_dim,
            head_dim=head_dim,
            rope_dim=self.rope_dim,
            attn_dropout=config.dropout,
            attention_type="mha",
            n_kv_heads=None,
            rope_base=10000.0,
            ffn_type="swiglu",
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(block_config) for _ in range(self.num_layers)]
        )
        self.output_norm = RMSNorm(self.token_dim)
        self.output_projection = nn.Linear(self.token_dim, int(output_dim))

    @staticmethod
    def build_rope_coordinates(
        *,
        num_court_points: int,
        num_ball_points: int,
        device: torch.device,
    ) -> Tensor:
        """Return two-axis coordinates for serial court-then-ball tokens."""
        if num_court_points <= 0 or num_ball_points <= 0:
            raise ValueError("court and ball token counts must be positive.")
        court = torch.zeros(num_court_points, 2, device=device, dtype=torch.long)
        court[:, 0] = torch.arange(num_court_points, device=device)
        ball = torch.zeros(num_ball_points, 2, device=device, dtype=torch.long)
        ball[:, 1] = torch.arange(num_ball_points, device=device)
        return torch.cat((court, ball), dim=0)

    def forward(
        self,
        *,
        court_kp: Tensor,
        court_visible: Tensor,
        ball_uv: Tensor,
        ball_visible: Tensor,
        context_valid: Tensor,
        mask_invisible_ball: bool,
    ) -> Tensor:
        """Return fused ball tokens shaped ``(..., P, output_dim)``."""
        if court_kp.shape[-2:] != (self.num_court_points, 2):
            raise ValueError(
                "court_kp must end in "
                f"({self.num_court_points}, 2), got {tuple(court_kp.shape)}."
            )
        if ball_uv.ndim < 2 or ball_uv.shape[-1] != 2:
            raise ValueError("ball_uv must have shape (..., P, 2).")
        if court_kp.shape[:-2] != ball_uv.shape[:-2]:
            raise ValueError("court_kp and ball_uv must share leading dimensions.")
        if court_visible.shape != court_kp.shape[:-1]:
            raise ValueError("court_visible must match court_kp without XY.")
        if ball_visible.shape != ball_uv.shape[:-1]:
            raise ValueError("ball_visible must match ball_uv without XY.")
        if context_valid.shape != ball_uv.shape[:-2]:
            raise ValueError("context_valid must match court/ball leading dimensions.")

        court_visible = court_visible.bool()
        ball_visible = ball_visible.bool()
        context_valid = context_valid.bool()
        num_ball_points = int(ball_uv.shape[-2])
        if num_ball_points <= 0:
            raise ValueError("At least one ball token is required.")

        safe_court = court_kp.masked_fill(~court_visible.unsqueeze(-1), 0.0)
        safe_ball = ball_uv.masked_fill(~ball_visible.unsqueeze(-1), 0.0)
        court_tokens = self.coordinate_projection(safe_court)
        ball_tokens = self.coordinate_projection(safe_ball)
        invisible = self.invisible_ball_token().to(
            device=ball_tokens.device,
            dtype=ball_tokens.dtype,
        )
        ball_tokens = torch.where(
            ball_visible.unsqueeze(-1),
            ball_tokens,
            invisible.view(*([1] * (ball_tokens.ndim - 1)), -1),
        )
        court_tokens = court_tokens + self.token_type_embedding[0]
        ball_tokens = ball_tokens + self.token_type_embedding[1]
        tokens = torch.cat((court_tokens, ball_tokens), dim=-2)

        court_key_valid = court_visible & context_valid.unsqueeze(-1)
        ball_context_valid = context_valid.unsqueeze(-1).expand_as(ball_visible)
        ball_key_valid = (
            ball_context_valid & ball_visible
            if mask_invisible_ball
            else ball_context_valid
        )
        token_valid = torch.cat((court_key_valid, ball_key_valid), dim=-1)

        sequence_length = self.num_court_points + num_ball_points
        flat_tokens = tokens.reshape(-1, sequence_length, self.token_dim)
        flat_valid = token_valid.reshape(-1, sequence_length)
        attention_mask, _ = build_self_attn_mask(flat_valid)
        rope_coordinates = self.build_rope_coordinates(
            num_court_points=self.num_court_points,
            num_ball_points=num_ball_points,
            device=ball_uv.device,
        )
        frequencies = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=rope_coordinates,
        )
        for block in self.blocks:
            flat_tokens = block(
                flat_tokens,
                freqs_cis=frequencies,
                attn_mask=attention_mask,
            )

        fused_ball = flat_tokens[:, self.num_court_points :]
        fused_ball = fused_ball.reshape(*ball_uv.shape[:-1], self.token_dim)
        ball_state_valid = (
            ball_context_valid & ball_visible
            if mask_invisible_ball
            else ball_context_valid
        )
        fused_ball = self.output_projection(self.output_norm(fused_ball))
        return cast(Tensor, fused_ball * ball_state_valid.unsqueeze(-1))


__all__ = ["CourtBallPointFusion"]
