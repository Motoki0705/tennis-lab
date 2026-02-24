"""Shared backbone for ball multi-task learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from src.experiments.ball_multitask.models.adapters.ball_3d_adapter import Ball3DTokenAdapter
from src.utils.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    MoEConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    YaRNConfig,
    precompute_freqs_cis,
)
from src.utils.models.embeddings import BallUVEmbedding, CourtKPUVEmbedding, InvisibleTokenEmbedding
from src.utils.schema.court import NUM_COURT_KP

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallMultitaskBackbone(nn.Module):
    """Shared Transformer backbone for UV/3D ball tasks.

    Token layout: [REG*, COURT(20), BALL(T)]
    Returns per-frame ball token features.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        yarn: YaRNConfig | None = None,
        use_moe: bool = False,
        moe_config: MoEConfig | None = None,
        causal: bool = False,
        max_seq_len: int = 256,
        invisible_init_std: float = 0.02,
        # Architecture depth
        num_ball_layers: int = 6,
        num_query_layers: int = 2,
        # Query init
        query_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if use_moe and moe_config is None:
            raise ValueError("use_moe=True requires moe_config.")

        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)

        head_dim = int(hidden_dim // num_heads)
        self.rope_dim = int(rope_dim) if rope_dim is not None else head_dim
        self.rope_theta = float(rope_theta)
        self.yarn = yarn
        self.max_tokens = int(NUM_COURT_KP + self.max_seq_len)

        if ffn_dim is None:
            ffn_dim = int((8 * hidden_dim) / 3)
        ffn_dim = (ffn_dim + 63) // 64 * 64

        if moe_config is not None and moe_config.dim != hidden_dim:
            raise ValueError("moe_config.dim must match hidden_dim.")
        if num_ball_layers < 0:
            raise ValueError("num_ball_layers must be non-negative.")
        if num_query_layers < 0:
            raise ValueError("num_query_layers must be non-negative.")

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim, init_std=float(invisible_init_std)
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.ball_uv_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )
        self.ball_3d_adapter = Ball3DTokenAdapter(
            dim=self.hidden_dim,
            dropout=float(dropout),
            invisible_token=self.invisible_token,
        )

        self.type_embed = nn.Embedding(2, self.hidden_dim)
        self.query_base = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * float(query_init_std))

        # Stage 1: Ball -> Court Cross-Attention + Ball Self-Attention
        self.ball_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=0,  # No RoPE for cross-attention
                        attn_dropout=float(dropout),
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_ball_layers))
            ]
        )
        self.ball_self_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=self.rope_theta,
                        yarn=self.yarn,
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_ball_layers))
            ]
        )

        # Stage 2: Query -> Memory (Processed Ball + Raw Ball) Cross-Attention + Query Self-Attention
        self.query_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,  # RoPE used for queries
                        attn_dropout=float(dropout),
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_query_layers))
            ]
        )
        self.query_self_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=int(num_heads),
                        mlp_inter_dim=int(ffn_dim),
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=float(dropout),
                        rope_base=self.rope_theta,
                        yarn=self.yarn,
                        use_moe=bool(use_moe),
                        moe_config=moe_config,
                    )
                )
                for _ in range(int(num_query_layers))
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        # Precompute frequencies for Ball tokens (Stage 1 self-attn) and Query (Stage 2 self-attn/cross-attn query)
        # Note: Query uses same length/frequencies as ball tokens.
        freqs_cis = precompute_freqs_cis(
            dim=self.rope_dim,
            seqlen=self.max_tokens,
            base=self.rope_theta,
            yarn=self.yarn,
            device=None,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> "BallMultitaskBackbone":
        """Create backbone from configuration."""
        model_cfg = config.get("model", {}) or {}
        yarn_cfg = model_cfg.get("yarn")
        yarn: YaRNConfig | None = None
        if yarn_cfg is not None:
            yarn_cfg = dict(yarn_cfg)
            if yarn_cfg.get("original_seq_len", None) is not None:
                yarn = YaRNConfig(**yarn_cfg)

        use_moe = bool(model_cfg.get("use_moe", False))
        moe_cfg = model_cfg.get("moe_config", None) or model_cfg.get("moe", None)
        moe_config: MoEConfig | None = None
        if use_moe and moe_cfg is not None:
            moe_config = MoEConfig(dim=int(model_cfg.get("hidden_dim", 256)), **dict(moe_cfg))

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_ball_layers=int(model_cfg.get("num_ball_layers", 6)),
            num_query_layers=int(model_cfg.get("num_query_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            yarn=yarn,
            use_moe=use_moe,
            moe_config=moe_config,
            causal=bool(model_cfg.get("causal", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", 256)),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
        )

    def forward_uv(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Encode UV input into query token features."""
        B, T, _ = ball_uv.shape
        self._validate_sequence_length(T)

        court_tokens = self.court_embed(court_kp, court_vis)
        ball_tokens = self.ball_uv_embed(ball_uv, ball_vis)
        return self._forward_tokens(
            ball_tokens=ball_tokens,
            court_tokens=court_tokens,
            ball_mask=ball_mask,
            seq_len=seq_len,
            batch_size=B,
            seq_len_t=T,
        )

    def forward_3d(
        self,
        ball_pos: Tensor,
        *,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        seq_len: Tensor | None = None,
    ) -> Tensor:
        """Encode 3D input into query token features."""
        B, T, _ = ball_pos.shape
        self._validate_sequence_length(T)

        court_tokens = torch.zeros(
            B, NUM_COURT_KP, self.hidden_dim, device=ball_pos.device, dtype=ball_pos.dtype
        )  # Dummy court for 3D input, or learned court tokens could be used if needed.
        # But for strictly 3D input (often event detection), court context might be implicit or not needed in Stage 1.
        # However, to be consistent with Stage 1 Cross-Attn, we need Key/Value.
        # Let's use a zero tensor or learned embedding if intended.
        # The prompt implies unified architecture. If 3D input doesn't come with court,
        # maybe we should use a learned placeholder?
        # The original code had `self.learned_court_tokens`. Let's restore that if it was useful,
        # but I removed it in __init__. Let's add it back implicitly or just handle it.

        # Re-introducing learned court tokens locally for 3D path since I removed them from self.
        # Or better, let's assume if 3D input is used, we might want to skip Stage 1 cross or provide dummy.
        # Given "multitask", let's be safe and use a zero-like placeholder if no court info.
        # Actually, let's just initialize a learned parameter on the fly or just zeros.
        # Zeros for now to avoid side effects.
        
        ball_tokens = self.ball_3d_adapter(ball_pos, ball_vis)
        return self._forward_tokens(
            ball_tokens=ball_tokens,
            court_tokens=court_tokens,
            ball_mask=ball_mask,
            seq_len=seq_len,
            batch_size=B,
            seq_len_t=T,
        )

    def _forward_tokens(
        self,
        *,
        ball_tokens: Tensor,
        court_tokens: Tensor,
        ball_mask: Tensor | None,
        seq_len: Tensor | None,
        batch_size: int,
        seq_len_t: int,
    ) -> Tensor:
        # Keep a copy of raw ball tokens for the skip connection
        ball_raw = ball_tokens

        # Frequencies for Ball (Stage 1) and Copy for Query (Stage 2)
        # Assuming Query sequence length == Ball sequence length
        freqs_cis_ball = self._build_freqs(seq_len_t, ball_tokens.device)
        freqs_cis_query = freqs_cis_ball 

        # Attention Masks
        # 1. Court Mask (for Stage 1 Cross-Attn): All valid usually
        court_valid = torch.ones(batch_size, NUM_COURT_KP, device=ball_tokens.device, dtype=torch.bool)
        
        # 2. Ball Mask (for Stage 1 Self-Attn & Stage 2 Cross-Attn Key Validity)
        if ball_mask is None:
            if seq_len is not None:
                t = torch.arange(seq_len_t, device=ball_tokens.device)[None, :]
                ball_mask = t < seq_len.to(torch.long).view(batch_size, 1)
            else:
                ball_mask = torch.ones(batch_size, seq_len_t, device=ball_tokens.device)
        
        ball_valid = ball_mask > 0
        ball_attn_mask = self._build_self_attn_mask(ball_valid)

        # STAGE 1: Ball Encoder
        # Cross-Attn (Ball->Court) then Self-Attn (Ball->Ball)
        x = ball_tokens
        for cross_layer, self_layer in zip(self.ball_cross_layers, self.ball_self_layers):
            x = cross_layer(
                x,
                court_tokens,
                key_valid=court_valid,
                freqs_q_cis=None,
                freqs_k_cis=None,
            )
            x, _ = self_layer(
                x,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_cis_ball,
                attn_mask=ball_attn_mask,
                is_causal=self.causal,
            )
        
        ball_processed = x

        # MEMORY CONSTRUCTION
        # Add explicit memory-type embeddings: processed branch vs raw branch.
        processed_type = self.type_embed(
            torch.zeros(seq_len_t, device=ball_tokens.device, dtype=torch.long)
        )[None, :, :]
        raw_type = self.type_embed(
            torch.ones(seq_len_t, device=ball_tokens.device, dtype=torch.long)
        )[None, :, :]
        ball_processed = ball_processed + processed_type
        ball_raw = ball_raw + raw_type

        # Concatenate [Processed Ball, Raw Ball]
        memory = torch.cat([ball_processed, ball_raw], dim=1)  # (B, 2T, D)
        memory_valid = torch.cat([ball_valid, ball_valid], dim=1) # (B, 2T)

        # RoPE for Memory (Cross-Attn Keys)
        # We need frequencies for the memory sequence.
        # Since memory is just two concatenated ball sequences of same length, we can concat freqs.
        freqs_cis_memory = torch.cat([freqs_cis_ball, freqs_cis_ball], dim=0)

        # STAGE 2: Query Decoder
        # Query Init
        query = self.query_base.expand(batch_size, seq_len_t, -1)

        # Cross-Attn (Query->Memory) then Self-Attn (Query->Query)
        for cross_layer, self_layer in zip(self.query_cross_layers, self.query_self_layers):
            query = cross_layer(
                query,
                memory,
                key_valid=memory_valid,
                freqs_q_cis=freqs_cis_query,
                freqs_k_cis=freqs_cis_memory,
            )
            query, _ = self_layer(
                query,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_cis_query,
                attn_mask=ball_attn_mask,
                is_causal=self.causal,
            )

        query = self.final_norm(query)
        return query

    def _build_freqs(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len > self.freqs_cis.shape[0]:
            raise ValueError(
                "Sequence length exceeds cached freqs_cis length. Increase max_seq_len."
            )
        freqs = self.freqs_cis[:seq_len]
        if freqs.device != device:
            freqs = freqs.to(device)
        return freqs

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
        return attn_mask

    def _validate_sequence_length(self, seq_len: int) -> None:
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BallMultitaskBackbone(hidden_dim=64, num_ball_layers=2, num_query_layers=2, num_heads=4, max_seq_len=16)
    ball_uv = torch.randn(2, 8, 2)
    court_kp = torch.randn(2, NUM_COURT_KP, 2)
    out = model.forward_uv(ball_uv, court_kp)
    assert out.shape == (2, 8, 64)
    ball_pos = torch.randn(2, 8, 3)
    out3 = model.forward_3d(ball_pos)
    assert out3.shape == (2, 8, 64)
    print("backbone smoke ok")
