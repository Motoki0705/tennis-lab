"""Query-based multi-view BLCS model (single-stage iterative query)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import Tensor, nn

from src.tasks.blcs.models.components.heads import Trajectory3DHead, VelocityHead
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
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int | None = None,
        ffn_type: Literal["swiglu", "mlp"] = "swiglu",
        dropout: float = 0.1,
        rope_dim: int | None = None,
        rope_theta: float = 10000.0,
        rope_theta_time: float | None = None,
        rope_theta_camera: float | None = None,
        rope_theta_type: float = 100.0,
        num_layers: int = 4,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        max_num_cameras: int = 8,
        num_court_tokens: int = NUM_COURT_KP,
        invisible_init_std: float = 0.02,
        query_init_std: float = 0.02,
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
        self.predict_velocity = bool(predict_velocity)
        self.num_court_tokens = int(num_court_tokens)

        head_dim = self.hidden_dim // num_heads
        rope_dim = head_dim if rope_dim is None else int(rope_dim)
        self._validate_rope_dim(rope_dim=rope_dim, head_dim=head_dim)
        self.rope_dim = int(rope_dim)
        self.rope_theta = float(rope_theta)
        self.rope_bases = (
            float(self.rope_theta if rope_theta_time is None else rope_theta_time),
            float(self.rope_theta if rope_theta_camera is None else rope_theta_camera),
            float(rope_theta_type),
        )

        if ffn_dim is None:
            ffn_dim = int((8 * self.hidden_dim) / 3)
            ffn_dim = (ffn_dim + 63) // 64 * 64

        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=invisible_init_std,
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
                        rope_base=self.rope_theta,
                        ffn_type=ffn_type,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(self.hidden_dim)

        self.position_head = Trajectory3DHead(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            output_dim=3,
            num_layers=2,
            dropout=dropout,
        )
        self.velocity_head = None
        if self.predict_velocity:
            self.velocity_head = VelocityHead(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim // 2,
                output_dim=3,
                num_layers=2,
                dropout=dropout,
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

    @staticmethod
    def _validate_rope_dim(*, rope_dim: int, head_dim: int) -> None:
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim}")
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}")

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSMultiViewModel:
        """Create model from Hydra/OmegaConf config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            ffn_type=cast(
                Literal["swiglu", "mlp"], str(model_cfg.get("ffn_type", "swiglu"))
            ),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            rope_theta_time=model_cfg.get("rope_theta_time", None),
            rope_theta_camera=model_cfg.get("rope_theta_camera", None),
            rope_theta_type=model_cfg.get("rope_theta_type", 100.0),
            num_layers=int(
                model_cfg.get("num_layers", model_cfg.get("num_stage2_layers", 4))
            ),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(
                model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))
            ),
            max_num_cameras=int(
                model_cfg.get("max_num_cameras", model_cfg.get("max_views", 8))
            ),
            num_court_tokens=int(
                model_cfg.get(
                    "num_court_tokens", data_cfg.get("num_court_kp", NUM_COURT_KP)
                )
            ),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
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

    @staticmethod
    def _build_self_attn_mask(valid: Tensor) -> tuple[Tensor, Tensor]:
        """Build self-attention mask from valid mask.

        Args:
            valid: Boolean valid mask, shape (B*, S).

        Returns:
            tuple:
              - attn_mask: Attention keep mask, shape (B*, S, S).
              - valid_fixed: Potentially fixed valid mask with at least one valid token.
        """
        valid_fixed = valid.bool()
        fully_masked = ~valid_fixed.any(dim=1)
        if fully_masked.any():
            valid_fixed = valid_fixed.clone()
            valid_fixed[fully_masked, 0] = True
        attn_mask = valid_fixed[:, None, :].expand(
            valid_fixed.shape[0], valid_fixed.shape[1], valid_fixed.shape[1]
        )
        return attn_mask, valid_fixed

    def _build_frame_tokens(
        self,
        ball_tok: Tensor,
        court_tok: Tensor,
        ball_valid: Tensor,
        court_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build per-frame memory tokens and validity mask.

        Args:
            ball_tok: (B, N, T, hidden_dim)
            court_tok: (B, N, T, K, hidden_dim)
            ball_valid: (B, N, T) bool
            court_valid: (B, N) bool camera-valid mask (time-invariant)

        Returns:
            tuple:
              - frame_tokens: (B, T, N * (K + 1), hidden_dim)
              - frame_valid: (B, T, N * (K + 1)) bool
        """
        batch_size, n_cams, seq_len_in, n_kp, _ = court_tok.shape
        if n_kp != self.num_court_tokens:
            raise ValueError(
                f"Expected court_tok with K={self.num_court_tokens}, got K={n_kp}."
            )

        per_cam_tokens = torch.cat([court_tok, ball_tok.unsqueeze(3)], dim=3)
        court_valid_expanded = court_valid[:, :, None, None].expand(
            batch_size, n_cams, seq_len_in, self.num_court_tokens
        )
        per_cam_valid = torch.cat(
            [court_valid_expanded, ball_valid.unsqueeze(3)], dim=3
        )

        frame_tokens = per_cam_tokens.permute(0, 2, 1, 3, 4).reshape(
            batch_size,
            seq_len_in,
            n_cams * (self.num_court_tokens + 1),
            self.hidden_dim,
        )
        frame_valid = per_cam_valid.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len_in, n_cams * (self.num_court_tokens + 1)
        )
        return frame_tokens, frame_valid

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, N, T, 2).
            court_kp: Court keypoints, shape (B, N, T, K, 2) or (B, N, K, 2).
            ball_vis: Ball visibility mask, shape (B, N, T). 1=visible.
            ball_mask: Ball validity mask, shape (B, N, T).
            court_vis: Court visibility mask, shape (B, N, T, K) or (B, N, K). Optional.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.
        """
        (
            court_kp,
            ball_vis,
            ball_mask,
            court_vis,
            batch_size,
            n_cams,
            seq_len_in,
        ) = self._prepare_forward_inputs(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
        )

        ball_uv_bn = ball_uv.reshape(batch_size * n_cams, seq_len_in, 2)
        ball_vis_bn = ball_vis.reshape(batch_size * n_cams, seq_len_in)
        ball_tok = self.ball_embed(ball_uv_bn, ball_vis_bn).reshape(
            batch_size, n_cams, seq_len_in, self.hidden_dim
        )

        court_kp_flat = court_kp.reshape(
            batch_size * n_cams * seq_len_in, self.num_court_tokens, 2
        )
        court_vis_flat = (
            court_vis.reshape(batch_size * n_cams * seq_len_in, self.num_court_tokens)
            if court_vis is not None
            else None
        )
        court_tok = self.court_embed(court_kp_flat, court_vis_flat).reshape(
            batch_size, n_cams, seq_len_in, self.num_court_tokens, self.hidden_dim
        )

        ball_valid = ball_mask > 0
        query_valid = ball_valid.any(dim=1)
        # Court validity is camera-level padding validity, derived from ball validity.
        # Shape: (B, N)
        court_valid = ball_valid.any(dim=2)

        freqs_time = self.query_freqs_cis[:seq_len_in]
        frame_freqs = self.frame_freqs_cis[
            :seq_len_in, : n_cams * (self.num_court_tokens + 1)
        ]

        frame_tokens, frame_token_valid = self._build_frame_tokens(
            ball_tok=ball_tok,
            court_tok=court_tok,
            ball_valid=ball_valid,
            court_valid=court_valid,
        )

        query_x = self.query_base.expand(batch_size, seq_len_in, -1)
        query_mask, query_valid_fixed = self._build_self_attn_mask(query_valid)

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
            key_valid = frame_token_valid.reshape(
                batch_size * seq_len_in, n_cams * (self.num_court_tokens + 1)
            )
            query_freqs_bt = freqs_time.unsqueeze(0).expand(
                batch_size, seq_len_in, self.rope_dim // 2
            )
            query_freqs_bt = query_freqs_bt.reshape(
                batch_size * seq_len_in, 1, self.rope_dim // 2
            )
            frame_freqs_bt = (
                frame_freqs.unsqueeze(0)
                .expand(
                    batch_size,
                    seq_len_in,
                    n_cams * (self.num_court_tokens + 1),
                    self.rope_dim // 2,
                )
                .reshape(
                    batch_size * seq_len_in,
                    n_cams * (self.num_court_tokens + 1),
                    self.rope_dim // 2,
                )
            )
            query_bt = cross_layer(
                query_bt,
                frame_bt,
                key_valid=key_valid,
                freqs_q_cis=query_freqs_bt,
                freqs_k_cis=frame_freqs_bt,
            )
            query_x = query_bt.reshape(batch_size, seq_len_in, self.hidden_dim)

            query_x = query_x * query_valid_fixed.unsqueeze(-1).to(dtype=query_x.dtype)
            query_x = temporal_layer(
                query_x,
                freqs_cis=freqs_time,
                attn_mask=query_mask,
            )

        query_x = self.final_norm(query_x)

        out: dict[str, Tensor] = {"position": self.position_head(query_x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(query_x)
        return out

    def _prepare_forward_inputs(
        self,
        *,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, int, int, int]:
        if ball_uv.dim() != 4:
            raise ValueError(
                f"ball_uv must have shape (B, N, T, 2), got {tuple(ball_uv.shape)}"
            )

        batch_size, n_cams, seq_len_in, _ = ball_uv.shape
        if seq_len_in > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len_in} exceeds max_seq_len={self.max_seq_len}. "
                "Increase model.max_seq_len."
            )
        if n_cams > self.max_num_cameras:
            raise ValueError(
                f"n_cams={n_cams} exceeds max_num_cameras={self.max_num_cameras}."
            )

        if court_kp.dim() == 4:
            court_kp = court_kp.unsqueeze(2).expand(-1, -1, seq_len_in, -1, -1)
        if court_kp.dim() != 5:
            raise ValueError(
                "court_kp must have shape "
                f"(B, N, T, {self.num_court_tokens}, 2) or "
                f"(B, N, {self.num_court_tokens}, 2), "
                f"got {tuple(court_kp.shape)}"
            )
        if court_kp.shape[-2] != self.num_court_tokens:
            raise ValueError(
                f"Expected court_kp with K={self.num_court_tokens}, got K={court_kp.shape[-2]}."
            )

        if court_vis is not None:
            if court_vis.dim() == 3:
                court_vis = court_vis.unsqueeze(2).expand(-1, -1, seq_len_in, -1)
            if court_vis.dim() != 4:
                raise ValueError(
                    "court_vis must have shape "
                    f"(B, N, T, {self.num_court_tokens}) or "
                    f"(B, N, {self.num_court_tokens}), "
                    f"got {tuple(court_vis.shape)}"
                )
            if court_vis.shape[-1] != self.num_court_tokens:
                raise ValueError(
                    f"Expected court_vis with K={self.num_court_tokens}, got K={court_vis.shape[-1]}."
                )
        if ball_vis is None:
            raise ValueError("ball_vis is required for BLCSMultiViewModel forward.")
        if ball_mask is None:
            raise ValueError("ball_mask is required for BLCSMultiViewModel forward.")
        if ball_vis.shape != (batch_size, n_cams, seq_len_in):
            raise ValueError(
                f"ball_vis must have shape {(batch_size, n_cams, seq_len_in)}, "
                f"got {tuple(ball_vis.shape)}"
            )
        if ball_mask.shape != (batch_size, n_cams, seq_len_in):
            raise ValueError(
                f"ball_mask must have shape {(batch_size, n_cams, seq_len_in)}, "
                f"got {tuple(ball_mask.shape)}"
            )
        return court_kp, ball_vis, ball_mask, court_vis, batch_size, n_cams, seq_len_in
