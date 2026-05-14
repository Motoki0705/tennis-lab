"""UV trajectory completion model with staged cross/self attention.

Architecture:
- Stage 1: Ball->Court cross-attention + Ball temporal self-attention
- Stage 2: Query->[processed ball tokens, raw ball tokens] cross-attention
           + Query temporal self-attention

Forward I/F is kept unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from src.utils.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
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


class UVTrajectoryCompletionModel(nn.Module):
    """Complete ball UV trajectory from noisy/masked inputs and court keypoints."""

    def __init__(
        self,
        *,
        # Core
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        # Architecture depth
        num_ball_layers: int = 6,
        num_query_layers: int = 2,
        # Positional encoding
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_entity: float | None = None,
        rope_theta_type: float = 100.0,
        rope_dim: int | None = None,
        # FFN
        ffn_dim: int | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        # Embedding init
        invisible_init_std: float = 0.02,
        query_init_std: float = 0.02,
        # Court tokens
        num_court_tokens: int = NUM_COURT_KP,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_court_tokens = int(num_court_tokens)

        self._validate_depths(
            num_ball_layers=num_ball_layers,
            num_query_layers=num_query_layers,
        )
        head_dim, rope_dim_value = self._resolve_attention_dims(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            rope_dim=rope_dim,
        )
        self.rope_dim = int(rope_dim_value)
        self.rope_theta = float(rope_theta)
        self.rope_bases = (
            float(self.rope_theta if rope_theta_time is None else rope_theta_time),
            float(self.rope_theta if rope_theta_entity is None else rope_theta_entity),
            float(rope_theta_type),
        )

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.court_id_embed = nn.Embedding(self.num_court_tokens, self.hidden_dim)
        self.query_base = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * float(query_init_std)
        )

        self.ball_to_court_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        # Stage1 cross does not use RoPE by architecture.
                        rope_dim=0,
                        attn_dropout=float(dropout),
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_ball_layers))
            ]
        )
        self.ball_temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_ball_layers))
            ]
        )
        self.query_to_ball_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_query_layers))
            ]
        )
        self.query_temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(int(num_query_layers))
            ]
        )

        # Keep this alias for existing training code that references len(self.model.blocks).
        self.blocks = self.ball_temporal_layers

        self.final_norm = RMSNorm(self.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
        )
        self.in_frame_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )

        court_freqs_cis = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_court_positions(),
            base=self.rope_bases,
        )
        self.register_buffer("court_freqs_cis", court_freqs_cis, persistent=False)

    @staticmethod
    def _validate_depths(*, num_ball_layers: int, num_query_layers: int) -> None:
        if num_ball_layers < 0:
            raise ValueError("num_ball_layers must be non-negative.")
        if num_query_layers < 0:
            raise ValueError("num_query_layers must be non-negative.")

    @staticmethod
    def _resolve_attention_dims(
        *,
        hidden_dim: int,
        num_heads: int,
        rope_dim: int | None,
    ) -> tuple[int, int]:
        if hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        head_dim = hidden_dim // int(num_heads)
        rope_dim_value = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim_value % 2 != 0:
            raise ValueError("rope_dim must be even.")
        if rope_dim_value > head_dim:
            raise ValueError("rope_dim must be <= head_dim.")
        return head_dim, rope_dim_value

    @classmethod
    def from_config(cls, config: DictConfig) -> UVTrajectoryCompletionModel:
        model_cfg = config.get("model", {}) or {}
        data_cfg = config.get("data", {}) or {}

        hidden_dim = int(model_cfg.get("hidden_dim", 256))

        num_ball_layers = model_cfg.get("num_ball_layers", 6)
        num_query_layers = model_cfg.get("num_query_layers", 2)

        return cls(
            hidden_dim=hidden_dim,
            num_heads=int(model_cfg.get("num_heads", 8)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            max_seq_len=int(
                model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 256))
            ),
            num_ball_layers=int(num_ball_layers),
            num_query_layers=int(num_query_layers),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_entity=model_cfg.get("rope_theta_entity", None),
            rope_theta_type=model_cfg.get("rope_theta_type", 100.0),
            rope_dim=model_cfg.get("rope_dim", None),
            ffn_dim=model_cfg.get("ffn_dim"),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
            num_court_tokens=int(
                model_cfg.get(
                    "num_court_tokens", data_cfg.get("num_court_kp", NUM_COURT_KP)
                )
            ),
        )

    def _build_court_positions(self) -> Tensor:
        court_idx = torch.arange(self.num_court_tokens, dtype=torch.long)
        return torch.stack(
            [
                torch.zeros_like(court_idx),
                court_idx,
                torch.zeros_like(court_idx),
            ],
            dim=-1,
        )

    @staticmethod
    def _build_stream_positions(
        seq_len: int,
        device: torch.device,
        *,
        token_type: int,
    ) -> Tensor:
        time_idx = torch.arange(seq_len, device=device, dtype=torch.long) + 1
        entity_idx = torch.zeros(seq_len, device=device, dtype=torch.long)
        type_idx = torch.full((seq_len,), token_type, device=device, dtype=torch.long)
        return torch.stack([time_idx, entity_idx, type_idx], dim=-1)

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        valid_fixed = valid.bool()
        fully_masked = ~valid_fixed.any(dim=1)
        if fully_masked.any():
            valid_fixed = valid_fixed.clone()
            valid_fixed[fully_masked, 0] = True
        attn_mask = valid_fixed[:, None, :].expand(
            valid_fixed.shape[0], valid_fixed.shape[1], valid_fixed.shape[1]
        )
        return attn_mask, valid_fixed

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        *,
        return_intermediate_ball_hidden: bool = False,
        return_in_frame_logits: bool = False,
    ) -> (
        Tensor
        | tuple[Tensor, list[Tensor]]
        | tuple[Tensor, Tensor]
        | tuple[Tensor, list[Tensor], Tensor]
    ):
        """Forward.

        Args:
            ball_uv: (B, T, 2) corrupted inputs (missing frames should be 0).
            court_kp: (B, K, 2) court keypoints.
            ball_vis: (B, T) observed mask (1=observed, 0=missing).
            ball_mask: (B, T) padding mask (1=valid).
            court_vis: (B, K) court visibility mask.
            return_intermediate_ball_hidden: If True, also return stage-1 ball token
                hidden states after each temporal layer.
            return_in_frame_logits: If True, also return per-frame in-frame logits.

        Returns:
            Completed UV predictions: (B, T, 2).
            If ``return_intermediate_ball_hidden=True``, returns
            ``(pred, intermediate_ball_hidden_list)``.
            If ``return_in_frame_logits=True``, returns additional ``in_frame_logits``
            with shape ``(B, T)``.
        """
        B, T, _ = ball_uv.shape
        ball_uv, ball_vis, ball_mask, T = self._clip_sequence_inputs(
            ball_uv=ball_uv,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            seq_len_in=T,
        )

        court_tok = self.court_embed(court_kp, court_vis)
        self._validate_court_tokens(court_tok)
        ball_tok = self.ball_embed(ball_uv, ball_vis)

        court_ids = torch.arange(
            self.num_court_tokens, device=ball_uv.device, dtype=torch.long
        )
        court_tokens = court_tok + self.court_id_embed(court_ids).unsqueeze(0)

        ball_valid = torch.ones(B, T, device=ball_uv.device, dtype=torch.bool)
        if ball_mask is not None:
            ball_valid = ball_mask > 0

        ball_attn_mask, ball_valid = self._build_self_attn_mask(ball_valid)
        court_valid = torch.ones(
            B, self.num_court_tokens, device=ball_uv.device, dtype=torch.bool
        )

        freqs_ball_tokens = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_stream_positions(T, ball_uv.device, token_type=1),
            base=self.rope_bases,
        )
        freqs_ball_raw = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_stream_positions(T, ball_uv.device, token_type=2),
            base=self.rope_bases,
        )
        freqs_query = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_stream_positions(T, ball_uv.device, token_type=3),
            base=self.rope_bases,
        )
        court_freqs_cis = cast(Tensor, self.court_freqs_cis)
        if court_freqs_cis.device != ball_uv.device:
            court_freqs_cis = court_freqs_cis.to(ball_uv.device)

        # Stage 1: ball tokens attend to court, then temporal self-attention.
        ball_raw = ball_tok
        ball_tokens = ball_raw
        intermediate_ball_hidden: list[Tensor] = []
        for cross_layer, self_layer in zip(
            self.ball_to_court_cross_layers,
            self.ball_temporal_layers,
            strict=True,
        ):
            ball_tokens = cross_layer(
                ball_tokens,
                court_tokens,
                key_valid=court_valid,
                freqs_q_cis=None,
                freqs_k_cis=court_freqs_cis,
            )
            ball_tokens = self_layer(
                ball_tokens,
                freqs_cis=freqs_ball_tokens,
                attn_mask=ball_attn_mask,
            )
            if return_intermediate_ball_hidden:
                intermediate_ball_hidden.append(ball_tokens)

        # Preserve raw token memory to anchor observed-frame representation.
        ball_memory = torch.cat([ball_tokens, ball_raw], dim=1)
        ball_memory_valid = torch.cat([ball_valid, ball_valid], dim=1)

        # Independent RoPE streams for query and memory branches.
        freqs_ball_memory = torch.cat([freqs_ball_tokens, freqs_ball_raw], dim=0)

        # Stage 2: learned query attends to processed+raw memories, then temporal self-attention.
        query = self.query_base.expand(B, T, -1)
        for cross_layer, self_layer in zip(
            self.query_to_ball_cross_layers,
            self.query_temporal_layers,
            strict=True,
        ):
            query = cross_layer(
                query,
                ball_memory,
                key_valid=ball_memory_valid,
                freqs_q_cis=freqs_query,
                freqs_k_cis=freqs_ball_memory,
            )
            query = self_layer(
                query,
                freqs_cis=freqs_query,
                attn_mask=ball_attn_mask,
            )

        query = self.final_norm(query)
        pred = self.head(query)
        in_frame_logits = self.in_frame_head(query).squeeze(-1)

        if return_intermediate_ball_hidden and return_in_frame_logits:
            return pred, intermediate_ball_hidden, in_frame_logits
        if return_intermediate_ball_hidden:
            return pred, intermediate_ball_hidden
        if return_in_frame_logits:
            return pred, in_frame_logits
        return pred

    def _clip_sequence_inputs(
        self,
        *,
        ball_uv: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        seq_len_in: int,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, int]:
        if self.max_seq_len < seq_len_in:
            ball_uv = ball_uv[:, : self.max_seq_len]
            if ball_vis is not None:
                ball_vis = ball_vis[:, : self.max_seq_len]
            if ball_mask is not None:
                ball_mask = ball_mask[:, : self.max_seq_len]
            seq_len_in = self.max_seq_len
        return ball_uv, ball_vis, ball_mask, seq_len_in

    def _validate_court_tokens(self, court_tokens: Tensor) -> None:
        if court_tokens.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tokens.shape[1]}"
            )


if __name__ == "__main__":
    model = UVTrajectoryCompletionModel(
        hidden_dim=64,
        num_ball_layers=2,
        num_query_layers=2,
        num_heads=4,
        max_seq_len=32,
    )
    ball_uv = torch.rand(2, 32, 2)
    ball_vis = torch.randint(0, 2, (2, 32)).float()
    court_kp = torch.rand(2, 20, 2)
    court_vis = torch.ones(2, 20)
    ball_mask = torch.ones(2, 32)
    out = model(
        ball_uv,
        court_kp,
        ball_vis,
        ball_mask,
        court_vis,
    )
    assert out.shape == (2, 32, 2)
    print("trajectory_completion.model smoke ok")
