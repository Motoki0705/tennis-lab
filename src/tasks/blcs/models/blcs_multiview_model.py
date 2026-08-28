"""Query-based multi-view BLCS model (single-stage iterative query)."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.configuration import MultiViewModelConfig
from src.tasks.blcs.models.components.heads import build_trajectory_output
from src.tasks.blcs.models.components.padding import (
    build_multiview_padding_masks,
    mask_trajectory_outputs,
)
from src.utils.models import (
    CrossAttnBlock,
    CrossAttnBlockConfig,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis_nd,
    resolve_rope_bases,
    validate_rope_dim,
)
from src.utils.models.components.ffn_layers import FFNType
from src.utils.models.embeddings import (
    BallUVEmbedding,
    CourtKPUVEmbedding,
    InvisibleTokenEmbedding,
)


class BLCSMultiViewModel(nn.Module):
    """Query-based multi-view BLCS model.

    Per iteration:
    1) Query (per frame) cross-attends to same-timestamp multi-view tokens.
    2) Query performs temporal self-attention across frames.

    Input memory tokens are built per camera as:
    [court_kp_0, ..., court_kp_(K-1), ball], then flattened across cameras.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        ffn_type: FFNType,
        dropout: float,
        rope_dim: int,
        rope_theta: float,
        rope_theta_time: float,
        rope_theta_camera: float,
        rope_theta_type: float,
        num_layers: int,
        predict_velocity: bool,
        max_seq_len: int,
        max_num_cameras: int,
        num_court_tokens: int,
        invisible_init_std: float,
        query_init_std: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self._validate_init_args(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            max_num_cameras=max_num_cameras,
            num_layers=num_layers,
        )

        self.max_seq_len = int(max_seq_len)
        self.max_num_cameras = int(max_num_cameras)
        self.num_court_tokens = int(num_court_tokens)

        head_dim = self.hidden_dim // num_heads
        validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_bases = resolve_rope_bases(
            rope_theta_time=rope_theta_time,
            rope_theta_camera=rope_theta_camera,
            rope_theta_type=rope_theta_type,
        )

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
        )
        self.court_embed = CourtKPUVEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
        )
        self.ball_embed = BallUVEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
        )

        self.query_base = nn.Parameter(
            torch.randn(1, 1, self.hidden_dim) * query_init_std
        )

        self.query_to_frame_cross_layers = nn.ModuleList(
            [
                CrossAttnBlock(
                    CrossAttnBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.query_temporal_layers = nn.ModuleList(
            [
                TransformerBlock(
                    TransformerBlockConfig(
                        dim=self.hidden_dim,
                        n_heads=num_heads,
                        ffn_dim=ffn_dim,
                        head_dim=head_dim,
                        rope_dim=self.rope_dim,
                        attn_dropout=dropout,
                        attention_type="mha",
                        n_kv_heads=None,
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        self.output_head = build_trajectory_output(
            input_dim=self.hidden_dim,
            dropout=dropout,
            predict_velocity=predict_velocity,
        )

        query_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_query_positions(seq_len=self.max_seq_len),
            base=self.rope_bases,
        )
        frame_freqs = precompute_freqs_cis_nd(
            dim=self.rope_dim,
            pos=self._build_frame_positions(
                n_cams=self.max_num_cameras,
                seq_len=self.max_seq_len,
            ),
            base=self.rope_bases,
        )
        self.register_buffer("query_freqs_cis", query_freqs, persistent=False)
        self.register_buffer("frame_freqs_cis", frame_freqs, persistent=False)

    @staticmethod
    def _validate_init_args(
        *,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        max_num_cameras: int,
        num_layers: int,
    ) -> None:
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_num_cameras <= 0:
            raise ValueError(f"max_num_cameras must be positive, got {max_num_cameras}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")

    @classmethod
    def from_config(cls, config: MultiViewModelConfig) -> BLCSMultiViewModel:
        """Create model from Hydra/OmegaConf config."""
        return cls(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            ffn_type=config.ffn_type,
            dropout=config.dropout,
            rope_dim=config.rope_dim,
            rope_theta=config.rope_theta,
            rope_theta_time=config.rope_theta_time,
            rope_theta_camera=config.rope_theta_camera,
            rope_theta_type=config.rope_theta_type,
            num_layers=config.num_layers,
            predict_velocity=config.predict_velocity,
            max_seq_len=config.max_seq_len,
            max_num_cameras=config.max_num_cameras,
            num_court_tokens=config.num_court_tokens,
            invisible_init_std=config.invisible_init_std,
            query_init_std=config.query_init_std,
        )

    def _build_frame_positions(
        self,
        *,
        n_cams: int,
        seq_len: int,
    ) -> Tensor:
        per_time: list[Tensor] = []
        for time_idx in range(seq_len):
            per_camera: list[Tensor] = []
            for cam_idx in range(n_cams):
                court_time = torch.full(
                    (self.num_court_tokens,),
                    time_idx + 1,
                    dtype=torch.long,
                )
                court_camera = torch.full(
                    (self.num_court_tokens,),
                    cam_idx,
                    dtype=torch.long,
                )
                court_type = torch.zeros(self.num_court_tokens, dtype=torch.long)
                ball_pos = torch.tensor(
                    [[time_idx + 1, cam_idx, 1]],
                    dtype=torch.long,
                )
                per_camera.append(
                    torch.cat(
                        [
                            torch.stack([court_time, court_camera, court_type], dim=-1),
                            ball_pos,
                        ],
                        dim=0,
                    )
                )
            per_time.append(torch.cat(per_camera, dim=0))
        return torch.stack(per_time, dim=0)

    def _build_query_positions(self, *, seq_len: int) -> Tensor:
        time_idx = torch.arange(seq_len, dtype=torch.long) + 1
        return torch.stack(
            [
                time_idx,
                torch.zeros_like(time_idx),
                torch.full_like(time_idx, 2),
            ],
            dim=-1,
        )

    def _build_frame_tokens(
        self,
        ball_tok: Tensor,
        court_tok: Tensor,
    ) -> Tensor:
        """Build per-frame memory tokens.

        Args:
            ball_tok: (B, N, T, hidden_dim)
            court_tok: (B, N, T, K, hidden_dim)
        Returns:
            frame_tokens: (B, T, N * (K + 1), hidden_dim)
        """
        batch_size, n_cams, seq_len_in = court_tok.shape[:3]

        per_cam_tokens = torch.cat([court_tok, ball_tok.unsqueeze(3)], dim=3)
        frame_tokens = per_cam_tokens.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            seq_len_in,
            n_cams * (self.num_court_tokens + 1),
            self.hidden_dim,
        )
        return frame_tokens

    def forward(
        self,
        ball_uv: Tensor,
        ball_vis: Tensor,
        court_kp: Tensor,
        court_vis: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, N, T, 2).
            court_kp: Court keypoints, shape (B, N, T, K, 2) or (B, N, K, 2).
            ball_vis: Ball visibility mask, shape (B, N, T). 1=visible.
            court_vis: Court visibility mask, shape (B, N, T, K).

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.
        """
        batch_size, n_cams, seq_len_in = ball_uv.shape[:3]
        masks = build_multiview_padding_masks(
            padding_mask,
            num_court_tokens=self.num_court_tokens,
        )

        ball_uv_bn = ball_uv.reshape(batch_size * n_cams, seq_len_in, 2)
        ball_vis_bn = ball_vis.reshape(batch_size * n_cams, seq_len_in)
        ball_tok = self.ball_embed(ball_uv_bn, ball_vis_bn).reshape(
            batch_size, n_cams, seq_len_in, self.hidden_dim
        )

        court_kp_flat = court_kp.reshape(
            batch_size * n_cams * seq_len_in, self.num_court_tokens, 2
        )
        court_vis_flat = court_vis.reshape(
            batch_size * n_cams * seq_len_in,
            self.num_court_tokens,
        )
        court_tok = self.court_embed(court_kp_flat, court_vis_flat).reshape(
            batch_size, n_cams, seq_len_in, self.num_court_tokens, self.hidden_dim
        )

        freqs_time = self.query_freqs_cis[:seq_len_in]
        frame_freqs = self.frame_freqs_cis[
            :seq_len_in, : n_cams * (self.num_court_tokens + 1)
        ]

        frame_tokens = self._build_frame_tokens(
            ball_tok=ball_tok,
            court_tok=court_tok,
        )

        query_x = self.query_base.expand(batch_size, seq_len_in, -1)

        for cross_layer, temporal_layer in zip(
            self.query_to_frame_cross_layers,
            self.query_temporal_layers,
            strict=True,
        ):
            query_bt = query_x.reshape(batch_size * seq_len_in, 1, self.hidden_dim)
            frame_bt = frame_tokens.reshape(
                batch_size * seq_len_in,
                n_cams * (self.num_court_tokens + 1),
                self.hidden_dim,
            )
            frame_bt = frame_bt * masks.frame_token_valid.reshape(
                batch_size * seq_len_in,
                n_cams * (self.num_court_tokens + 1),
                1,
            )
            query_freqs_bt = freqs_time.unsqueeze(0).expand(
                batch_size, seq_len_in, 1, self.rope_dim // 2
            )
            query_freqs_bt = query_freqs_bt.reshape(
                batch_size * seq_len_in, 1, 1, self.rope_dim // 2
            )
            frame_freqs_bt = (
                frame_freqs.unsqueeze(0)
                .expand(
                    batch_size,
                    seq_len_in,
                    n_cams * (self.num_court_tokens + 1),
                    1,
                    self.rope_dim // 2,
                )
                .reshape(
                    batch_size * seq_len_in,
                    n_cams * (self.num_court_tokens + 1),
                    1,
                    self.rope_dim // 2,
                )
            )
            query_bt = cross_layer(
                query_bt,
                frame_bt,
                attn_mask=masks.cross_attention_keep_mask,
                freqs_q_cis=query_freqs_bt,
                freqs_k_cis=frame_freqs_bt,
            )
            query_x = query_bt.reshape(batch_size, seq_len_in, self.hidden_dim)

            query_x = query_x * masks.frame_valid.unsqueeze(-1).to(dtype=query_x.dtype)
            query_x = temporal_layer(
                query_x,
                freqs_cis=freqs_time,
                attn_mask=masks.query_attention_keep_mask,
            )

        query_x = self.final_norm(query_x)

        outputs = cast("dict[str, Tensor]", self.output_head(query_x))
        return mask_trajectory_outputs(outputs, masks.frame_valid)

    query_freqs_cis: Tensor
    frame_freqs_cis: Tensor
