"""Query-based BLCS model with staged court/temporal/readout attention."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.blcs.models.components.heads import Trajectory3DHead, VelocityHead
from src.common.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    YaRNConfig,
    precompute_freqs_cis,
)
from src.common.models.embeddings import (
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
)
from src.utils.geometry import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSQueryModel(nn.Module):
    """BLCS query-based 3D reconstruction model.

    Stages:
    - Stage A/B: Interleaved Ball->Court cross-attention and Ball temporal self-attention
    - Stage C: Query->Ball cross-attention (time RoPE on query/ball Q,K)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        num_ball_layers: int = 4,
        num_query2ball_layers: int = 2,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        num_court_tokens: int = NUM_COURT_KP,
        invisible_init_std: float = 0.02,
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if num_ball_layers < 0:
            raise ValueError(f"num_ball_layers must be non-negative, got {num_ball_layers}")

        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.predict_velocity = bool(predict_velocity)
        self.num_court_tokens = int(num_court_tokens)

        head_dim = hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.invisible_token = InvisibleTokenEmbedding(
            dim=hidden_dim,
            init_std=invisible_init_std,
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=hidden_dim,
            dropout=dropout,
            invisible_token=self.invisible_token,
        )

        self.court_id_embed = nn.Embedding(self.num_court_tokens, hidden_dim)

        self.query_base = nn.Parameter(torch.randn(1, 1, hidden_dim) * query_init_std)

        self.ball_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        mlp_inter_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                    )
                )
                for _ in range(num_ball_layers)
            ]
        )
        self.ball_self_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        mlp_inter_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                        rope_base=rope_theta,
                        yarn=yarn,
                    )
                )
                for _ in range(num_ball_layers)
            ]
        )
        self.ball_temporal_norm = RMSNorm(hidden_dim)
        self.query2ball_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=hidden_dim,
                        n_heads=num_heads,
                        mlp_inter_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=rope_dim,
                        attn_dropout=dropout,
                    )
                )
                for _ in range(num_query2ball_layers)
            ]
        )

        self.final_norm = RMSNorm(hidden_dim)

        self.position_head = Trajectory3DHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity_head = None
        if self.predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
            )

        freqs_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=self.max_seq_len,
            base=rope_theta,
            yarn=yarn,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSQueryModel:
        """Create model from Hydra/OmegaConf config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        yarn_cfg = model_cfg.get("yarn", None)
        yarn: YaRNConfig | None = None
        if yarn_cfg is not None:
            yarn_cfg = dict(yarn_cfg)
            if yarn_cfg.get("original_seq_len") is not None:
                yarn = YaRNConfig(**yarn_cfg)

        num_ball_layers = model_cfg.get("num_ball_layers", None)
        if num_ball_layers is None:
            num_ball_layers = model_cfg.get(
                "num_ball_temporal_layers",
                model_cfg.get("num_ball2court_layers", model_cfg.get("num_layers", 4)),
            )

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            yarn=yarn,
            num_ball_layers=int(num_ball_layers),
            num_query2ball_layers=int(model_cfg.get("num_query2ball_layers", 2)),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))),
            num_court_tokens=int(model_cfg.get("num_court_tokens", NUM_COURT_KP)),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
        )

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass."""
        batch_size, seq_len, _ = ball_uv.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )

        court_tok = self.court_embed(court_kp, court_vis)
        if court_tok.shape[1] != self.num_court_tokens:
            raise ValueError(
                f"Expected {self.num_court_tokens} court tokens, got {court_tok.shape[1]}"
            )
        ball_tok = self.ball_embed(ball_uv, ball_vis)

        court_valid = torch.ones(
            batch_size, court_tok.shape[1], device=ball_uv.device, dtype=torch.bool
        )
        ball_valid = torch.ones(
            batch_size, seq_len, device=ball_uv.device, dtype=torch.bool
        )
        if ball_mask is not None:
            ball_valid = ball_mask > 0

        court_ids = torch.arange(court_tok.shape[1], device=ball_uv.device, dtype=torch.long)
        court_tok = court_tok + self.court_id_embed(court_ids).unsqueeze(0)
        query_tok = self.query_base.expand(batch_size, seq_len, -1)

        freqs_cis = self.freqs_cis[:seq_len]
        if freqs_cis.device != ball_uv.device:
            freqs_cis = freqs_cis.to(ball_uv.device)

        ball_attn_mask: Tensor | None = None
        if ball_mask is not None:
            ball_key_valid = ball_valid
            empty_ball = ~ball_key_valid.any(dim=1)
            if empty_ball.any():
                ball_key_valid = ball_key_valid.clone()
                ball_key_valid[empty_ball, 0] = True
            ball_attn_mask = ball_key_valid[:, None, :].expand(batch_size, seq_len, seq_len)

        # Stage A/B: interleaved ball->court cross-attn and ball temporal self-attn
        ball_x = ball_tok
        residual: Tensor | None = None
        for cross_layer, self_layer in zip(self.ball_cross_layers, self.ball_self_layers):
            ball_x = cross_layer(
                ball_x,
                court_tok,
                key_valid=court_valid,
            )
            ball_x, residual = self_layer(
                ball_x,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_cis,
                attn_mask=ball_attn_mask,
                is_causal=False,
            )

        if len(self.ball_self_layers) > 0:
            if residual is None:
                ball_b = self.ball_temporal_norm(ball_x)
            else:
                ball_b, _ = self.ball_temporal_norm(ball_x, residual)
        else:
            ball_b = ball_x

        # Stage C: query -> ball readout (time RoPE on query and ball)
        query_c = query_tok
        for layer in self.query2ball_layers:
            query_c = layer(
                query_c,
                ball_b,
                key_valid=ball_valid,
                freqs_q_cis=freqs_cis,
                freqs_k_cis=freqs_cis,
            )

        query_c = self.final_norm(query_c)

        out: dict[str, Tensor] = {"position": self.position_head(query_c)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(query_c)
        return out
