"""Query-based multi-view BLCS model (2-stage)."""

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


class BLCSMultiViewModel(nn.Module):
    """Query-based multi-view BLCS model (2-stage).

    Stage 1:
    - Interleaved Ball->Court cross-attention, camera-wise temporal self-attention,
      and time-wise camera self-attention.

    Stage 2:
    - Shared readout query stream attends to per-frame all-camera ball states,
      then temporal self-attend.
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
        num_stage1_blocks: int = 4,
        num_stage2_layers: int = 2,
        predict_velocity: bool = False,
        max_seq_len: int = 120,
        max_num_cameras: int = 8,
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
        if max_num_cameras <= 0:
            raise ValueError(
                f"max_num_cameras must be positive, got {max_num_cameras}"
            )
        if num_stage1_blocks < 0:
            raise ValueError(
                f"num_stage1_blocks must be non-negative, got {num_stage1_blocks}"
            )
        if num_stage2_layers < 0:
            raise ValueError(
                f"num_stage2_layers must be non-negative, got {num_stage2_layers}"
            )

        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.max_num_cameras = int(max_num_cameras)
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
        self.cam_id_embed = nn.Embedding(self.max_num_cameras, hidden_dim)

        self.query_base = nn.Parameter(torch.randn(1, 1, hidden_dim) * query_init_std)

        self.stage1_cross_layers = nn.ModuleList(
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
                for _ in range(num_stage1_blocks)
            ]
        )
        self.stage1_temporal_layers = nn.ModuleList(
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
                for _ in range(num_stage1_blocks)
            ]
        )
        self.stage1_camera_layers = nn.ModuleList(
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
                for _ in range(num_stage1_blocks)
            ]
        )

        self.stage2_cross_layers = nn.ModuleList(
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
                for _ in range(num_stage2_layers)
            ]
        )
        self.stage2_temporal_layers = nn.ModuleList(
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
                for _ in range(num_stage2_layers)
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

        freqs_time_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=self.max_seq_len,
            base=rope_theta,
            yarn=yarn,
            device=None,
        )
        self.register_buffer("freqs_time_cis", freqs_time_cis, persistent=False)

        freqs_cam_cis = precompute_freqs_cis(
            dim=rope_dim,
            seqlen=self.max_num_cameras,
            base=rope_theta,
            yarn=yarn,
            device=None,
        )
        self.register_buffer("freqs_cam_cis", freqs_cam_cis, persistent=False)

    @classmethod
    def from_config(cls, config: DictConfig) -> BLCSMultiViewModel:
        """Create model from Hydra/OmegaConf config."""
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})

        yarn_cfg = model_cfg.get("yarn", None)
        yarn: YaRNConfig | None = None
        if yarn_cfg is not None:
            yarn_cfg = dict(yarn_cfg)
            if yarn_cfg.get("original_seq_len") is not None:
                yarn = YaRNConfig(**yarn_cfg)

        num_stage1_blocks = model_cfg.get("num_stage1_blocks", None)
        if num_stage1_blocks is None:
            num_stage1_blocks = model_cfg.get(
                "num_ball_layers",
                model_cfg.get(
                    "num_ball_temporal_layers",
                    model_cfg.get("num_ball2court_layers", model_cfg.get("num_layers", 4)),
                ),
            )

        return cls(
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            ffn_dim=model_cfg.get("ffn_dim", None),
            dropout=float(model_cfg.get("dropout", 0.1)),
            rope_dim=model_cfg.get("rope_dim", None),
            rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
            yarn=yarn,
            num_stage1_blocks=int(num_stage1_blocks),
            num_stage2_layers=int(model_cfg.get("num_stage2_layers", model_cfg.get("num_query2ball_layers", 2))),
            predict_velocity=bool(model_cfg.get("predict_velocity", False)),
            max_seq_len=int(model_cfg.get("max_seq_len", data_cfg.get("max_seq_len", 120))),
            max_num_cameras=int(model_cfg.get("max_num_cameras", model_cfg.get("max_views", 8))),
            num_court_tokens=int(model_cfg.get("num_court_tokens", NUM_COURT_KP)),
            invisible_init_std=float(model_cfg.get("invisible_init_std", 0.02)),
            query_init_std=float(model_cfg.get("query_init_std", 0.02)),
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
        attn_mask = valid_fixed[:, None, :].expand(valid_fixed.shape[0], valid_fixed.shape[1], valid_fixed.shape[1])
        return attn_mask, valid_fixed

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None = None,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        num_views: Tensor | None = None,
        seq_len: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D positions, shape (B, N, T, 2).
            court_kp: Court keypoints, shape (B, N, T, 20, 2) or (B, N, 20, 2).
            ball_vis: Ball visibility flags, shape (B, N, T). Optional.
            ball_mask: Ball validity mask, shape (B, N, T). Optional.
            court_vis: Court visibility mask, shape (B, N, T, 20) or (B, N, 20). Optional.
            num_views: Number of valid views per sample, shape (B,). Optional.
            seq_len: Valid sequence length per sample, shape (B,). Optional.
            camera_params: Camera parameters. Optional and currently unused.

        Returns:
            dict: Dictionary with 'position' (B, T, 3) and optionally 'velocity'.
        """
        del camera_params

        if ball_uv.dim() != 4:
            raise ValueError(f"ball_uv must have shape (B, N, T, 2), got {tuple(ball_uv.shape)}")

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
                "court_kp must have shape (B, N, T, 20, 2) or (B, N, 20, 2), "
                f"got {tuple(court_kp.shape)}"
            )

        if court_vis is not None:
            if court_vis.dim() == 3:
                court_vis = court_vis.unsqueeze(2).expand(-1, -1, seq_len_in, -1)
            if court_vis.dim() != 4:
                raise ValueError(
                    "court_vis must have shape (B, N, T, 20) or (B, N, 20), "
                    f"got {tuple(court_vis.shape)}"
                )

        if ball_vis is None and ball_mask is not None:
            ball_vis = ball_mask

        if ball_vis is not None and ball_vis.shape != (batch_size, n_cams, seq_len_in):
            raise ValueError(
                f"ball_vis must have shape {(batch_size, n_cams, seq_len_in)}, "
                f"got {tuple(ball_vis.shape)}"
            )
        if ball_mask is not None and ball_mask.shape != (batch_size, n_cams, seq_len_in):
            raise ValueError(
                f"ball_mask must have shape {(batch_size, n_cams, seq_len_in)}, "
                f"got {tuple(ball_mask.shape)}"
            )

        ball_uv_bn = ball_uv.reshape(batch_size * n_cams, seq_len_in, 2)
        ball_vis_bn = (
            ball_vis.reshape(batch_size * n_cams, seq_len_in)
            if ball_vis is not None
            else None
        )
        ball_tok = self.ball_embed(ball_uv_bn, ball_vis_bn).reshape(
            batch_size, n_cams, seq_len_in, self.hidden_dim
        )

        court_kp_flat = court_kp.reshape(batch_size * n_cams * seq_len_in, self.num_court_tokens, 2)
        court_vis_flat = (
            court_vis.reshape(batch_size * n_cams * seq_len_in, self.num_court_tokens)
            if court_vis is not None
            else None
        )
        court_tok = self.court_embed(court_kp_flat, court_vis_flat).reshape(
            batch_size, n_cams, seq_len_in, self.num_court_tokens, self.hidden_dim
        )

        device = ball_uv.device
        cam_ids = torch.arange(n_cams, device=device, dtype=torch.long)
        cam_emb = self.cam_id_embed(cam_ids).view(1, n_cams, 1, self.hidden_dim)

        court_ids = torch.arange(self.num_court_tokens, device=device, dtype=torch.long)
        court_id_emb = self.court_id_embed(court_ids).view(1, 1, 1, self.num_court_tokens, self.hidden_dim)

        # Apply camera/token identity before Stage 1.
        ball_tok = ball_tok + cam_emb
        court_tok = court_tok + court_id_emb + cam_emb.unsqueeze(3)

        valid = torch.ones(batch_size, n_cams, seq_len_in, device=device, dtype=torch.bool)
        if ball_mask is not None:
            valid = valid & (ball_mask > 0)
        elif ball_vis is not None:
            valid = valid & (ball_vis > 0)

        if num_views is not None:
            cam_idx = torch.arange(n_cams, device=device).view(1, n_cams, 1)
            valid = valid & (cam_idx < num_views.view(batch_size, 1, 1))

        if seq_len is not None:
            t_idx = torch.arange(seq_len_in, device=device).view(1, 1, seq_len_in)
            valid = valid & (t_idx < seq_len.view(batch_size, 1, 1))

        freqs_time = self.freqs_time_cis[:seq_len_in]
        if freqs_time.device != device:
            freqs_time = freqs_time.to(device)

        freqs_cam = self.freqs_cam_cis[:n_cams]
        if freqs_cam.device != device:
            freqs_cam = freqs_cam.to(device)

        # Stage 1: cross (per camera/frame) -> temporal self (per camera) -> camera self (per frame)
        ball_x = ball_tok
        for cross_layer, temporal_layer, camera_layer in zip(
            self.stage1_cross_layers,
            self.stage1_temporal_layers,
            self.stage1_camera_layers,
            strict=True,
        ):
            q = ball_x.reshape(batch_size * n_cams * seq_len_in, 1, self.hidden_dim)
            kv = court_tok.reshape(
                batch_size * n_cams * seq_len_in,
                self.num_court_tokens,
                self.hidden_dim,
            )
            court_valid = torch.ones(
                batch_size * n_cams * seq_len_in,
                self.num_court_tokens,
                dtype=torch.bool,
                device=device,
            )
            q = cross_layer(q, kv, key_valid=court_valid)
            ball_x = q.reshape(batch_size, n_cams, seq_len_in, self.hidden_dim)

            temporal_x = ball_x.reshape(batch_size * n_cams, seq_len_in, self.hidden_dim)
            temporal_valid = valid.reshape(batch_size * n_cams, seq_len_in)
            temporal_mask, temporal_valid_fixed = self._build_self_attn_mask(temporal_valid)
            temporal_x = temporal_x * temporal_valid_fixed.unsqueeze(-1).to(dtype=temporal_x.dtype)
            temporal_x, _ = temporal_layer(
                temporal_x,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_time,
                attn_mask=temporal_mask,
                is_causal=False,
            )
            ball_x = temporal_x.reshape(batch_size, n_cams, seq_len_in, self.hidden_dim)

            camera_x = ball_x.permute(0, 2, 1, 3).reshape(batch_size * seq_len_in, n_cams, self.hidden_dim)
            camera_valid = valid.permute(0, 2, 1).reshape(batch_size * seq_len_in, n_cams)
            camera_mask, camera_valid_fixed = self._build_self_attn_mask(camera_valid)
            camera_x = camera_x * camera_valid_fixed.unsqueeze(-1).to(dtype=camera_x.dtype)
            camera_x, _ = camera_layer(
                camera_x,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_cam,
                attn_mask=camera_mask,
                is_causal=False,
            )
            ball_x = camera_x.reshape(batch_size, seq_len_in, n_cams, self.hidden_dim).permute(0, 2, 1, 3)

        # Stage 2: query_t -> ball_all_cam_t cross-attn, then temporal self-attn on query
        query_x = self.query_base.expand(batch_size, seq_len_in, -1)
        frame_valid = valid.any(dim=1)
        query_mask, query_valid_fixed = self._build_self_attn_mask(frame_valid)

        for cross_layer, temporal_layer in zip(
            self.stage2_cross_layers,
            self.stage2_temporal_layers,
            strict=True,
        ):
            query_bt = query_x.reshape(batch_size * seq_len_in, 1, self.hidden_dim)
            ball_bt = ball_x.permute(0, 2, 1, 3).reshape(batch_size * seq_len_in, n_cams, self.hidden_dim)
            key_valid = valid.permute(0, 2, 1).reshape(batch_size * seq_len_in, n_cams)
            query_bt = cross_layer(query_bt, ball_bt, key_valid=key_valid)
            query_x = query_bt.reshape(batch_size, seq_len_in, self.hidden_dim)

            query_x = query_x * query_valid_fixed.unsqueeze(-1).to(dtype=query_x.dtype)
            query_x, _ = temporal_layer(
                query_x,
                residual=None,
                start_pos=0,
                freqs_cis=freqs_time,
                attn_mask=query_mask,
                is_causal=False,
            )

        query_x = self.final_norm(query_x)

        out: dict[str, Tensor] = {"position": self.position_head(query_x)}
        if self.predict_velocity and self.velocity_head is not None:
            out["velocity"] = self.velocity_head(query_x)
        return out
