"""Persistent track-query model for ID-ordered multi-view ball observations."""

from __future__ import annotations

import torch
from torch import nn

from src.tasks.blcs.configuration import TrackQueryModelConfig
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.models.components import CourtBallPointFusion
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    build_self_attn_mask,
    precompute_freqs_cis,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings import CourtBallGroupEmbedding, InvisibleTokenEmbedding


class BLCSTrackQueryModel(nn.Module):
    """Alternate spatial candidate aggregation and per-slot temporal attention."""

    def __init__(self, config: TrackQueryModelConfig) -> None:
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.num_queries = int(config.num_queries)
        self.num_stages = int(config.num_stages)
        self.role_rope_enabled = bool(config.role_rope_enabled)
        self.mask_invisible_observations = config.mask_invisible_observations
        self.observation_fusion = config.observation_fusion
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = config.rope_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be even and no larger than the attention head dim."
            )

        self.num_court_tokens = 14
        self.invisible_token: InvisibleTokenEmbedding | None = None
        self.group_embed: CourtBallGroupEmbedding | None = None
        self.point_fusion: CourtBallPointFusion | None = None
        invisible_init_std = config.invisible_init_std
        if self.observation_fusion == "linear":
            self.invisible_token = InvisibleTokenEmbedding(
                dim=self.hidden_dim,
                init_std=invisible_init_std,
            )
            self.group_embed = CourtBallGroupEmbedding(
                dim=self.hidden_dim,
                invisible_token=self.invisible_token,
                num_court_tokens=self.num_court_tokens,
            )
        elif self.observation_fusion == "point_attention":
            point_fusion_config = config.point_fusion
            if point_fusion_config is None:
                raise ValueError(
                    "model.point_fusion is required when observation_fusion="
                    "'point_attention'."
                )
            self.point_fusion = CourtBallPointFusion(
                output_dim=self.hidden_dim,
                num_court_points=self.num_court_tokens,
                config=point_fusion_config,
                invisible_init_std=invisible_init_std,
            )
        else:
            raise ValueError(
                "observation_fusion must be 'linear' or 'point_attention', got "
                f"{self.observation_fusion!r}."
            )
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.num_queries, self.hidden_dim) * 0.02
        )
        block_config = TransformerBlockConfig(
            dim=self.hidden_dim,
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
        self.spatial_blocks = nn.ModuleList(
            [TransformerBlock(block_config) for _ in range(self.num_stages)]
        )
        self.temporal_blocks = nn.ModuleList(
            [TransformerBlock(block_config) for _ in range(self.num_stages)]
        )
        self.output_norm = RMSNorm(self.hidden_dim)
        self.position_head = nn.Linear(self.hidden_dim, 3)
        self.presence_head = nn.Linear(self.hidden_dim, 1)

    @staticmethod
    def build_spatial_coordinates(
        *,
        batch_size: int,
        num_frames: int,
        num_views: int,
        num_detections: int,
        num_queries: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return ``(B*T,Q+V*D,3)`` time/camera/role coordinates."""
        time = torch.arange(num_frames, device=device).view(1, num_frames, 1)
        slot = torch.zeros(
            batch_size, num_frames, num_queries, 3, device=device, dtype=torch.long
        )
        slot[..., 0] = time
        camera_tokens = torch.zeros(
            batch_size,
            num_frames,
            num_views,
            num_detections,
            3,
            device=device,
            dtype=torch.long,
        )
        camera_tokens[..., 0] = time.view(1, num_frames, 1, 1)
        camera = torch.arange(1, num_views + 1, device=device).view(1, 1, num_views, 1)
        camera_tokens[..., 1] = camera
        camera_tokens[..., 2] = 1
        return torch.cat([slot, camera_tokens.flatten(2, 3)], dim=2).flatten(0, 1)

    def forward(
        self,
        ball_uv: torch.Tensor,
        ball_visible: torch.Tensor,
        court_kp: torch.Tensor,
        court_vis: torch.Tensor,
        frame_mask: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> BLCSTrackingPrediction:
        """Predict clip-local ball tracks.

        Args follow the multi-object contract in ``blcs/README.md``.
        """
        batch_size, num_views, num_frames, num_detections, _ = ball_uv.shape
        if ball_visible.shape != ball_uv.shape[:-1]:
            raise ValueError(
                "ball_visible must match ball_uv without its UV axis, got "
                f"ball_visible={tuple(ball_visible.shape)} and "
                f"ball_uv={tuple(ball_uv.shape)}."
            )
        observation_padding_valid = (
            view_mask[:, :, None, None] & frame_mask[:, None, :, None]
        ).expand(-1, -1, -1, num_detections)
        ball_visible = ball_visible.bool()
        if court_kp.shape[:3] != (batch_size, num_views, num_frames):
            raise ValueError("court_kp leading dimensions must match ball_uv (B,V,T).")
        if court_kp.shape[3:] != (self.num_court_tokens, 2):
            raise ValueError(
                f"court_kp must contain all {self.num_court_tokens} annotated UV "
                "points; "
                f"got shape {tuple(court_kp.shape)}."
            )
        if court_vis.shape != court_kp.shape[:-1]:
            raise ValueError(
                "court_vis must match court_kp without its UV axis, got "
                f"court_vis={tuple(court_vis.shape)} and "
                f"court_kp={tuple(court_kp.shape)}."
            )
        court_visible = court_vis if court_vis.dtype == torch.bool else court_vis > 0
        if self.observation_fusion == "linear":
            assert self.group_embed is not None
            masked_court = court_kp.masked_fill(~court_visible.unsqueeze(-1), 0.0)
            court_for_candidates = masked_court.unsqueeze(3).expand(
                -1, -1, -1, num_detections, -1, -1
            )
            ball_for_candidates = ball_uv.masked_fill(~ball_visible.unsqueeze(-1), 0.0)
            camera_tokens = self.group_embed(
                court_for_candidates,
                ball_for_candidates,
                ball_visible,
            ).permute(0, 2, 1, 3, 4)
        else:
            assert self.point_fusion is not None
            context_valid = view_mask[:, :, None] & frame_mask[:, None, :]
            camera_tokens = self.point_fusion(
                court_kp=court_kp,
                court_visible=court_visible,
                ball_uv=ball_uv,
                ball_visible=ball_visible,
                context_valid=context_valid,
                mask_invisible_ball=self.mask_invisible_observations,
            ).permute(0, 2, 1, 3, 4)
        camera_tokens = camera_tokens.reshape(
            batch_size,
            num_frames,
            num_views * num_detections,
            self.hidden_dim,
        )
        camera_padding_valid = observation_padding_valid.permute(0, 2, 1, 3)
        camera_attention_valid = camera_padding_valid.clone()
        if self.mask_invisible_observations:
            camera_attention_valid &= ball_visible.permute(0, 2, 1, 3)
        camera_state_valid = camera_attention_valid
        camera_tokens = camera_tokens * camera_state_valid.reshape(
            batch_size, num_frames, -1
        ).unsqueeze(-1)

        slots = self.slot_embeddings.view(
            1, 1, self.num_queries, self.hidden_dim
        ).expand(batch_size, num_frames, -1, -1)
        slots = slots * frame_mask[:, :, None, None]
        coordinates = self.build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_detections=num_detections,
            num_queries=self.num_queries,
            device=ball_uv.device,
        )
        rope_coordinates = coordinates
        if not self.role_rope_enabled:
            rope_coordinates = coordinates.clone()
            rope_coordinates[..., 2] = 0
        spatial_freqs = precompute_freqs_cis_nd(dim=self.rope_dim, pos=rope_coordinates)
        time_freqs = precompute_freqs_cis(
            dim=self.rope_dim, seqlen=num_frames, device=ball_uv.device
        )
        slot_padding_valid = frame_mask[:, :, None].expand(-1, -1, self.num_queries)
        spatial_valid = torch.cat(
            [
                slot_padding_valid,
                camera_attention_valid.reshape(batch_size, num_frames, -1),
            ],
            dim=2,
        ).flatten(0, 1)
        spatial_mask, _ = build_self_attn_mask(spatial_valid)
        temporal_valid = (
            frame_mask[:, None, :]
            .expand(-1, self.num_queries, -1)
            .reshape(batch_size * self.num_queries, num_frames)
        )
        temporal_mask, _ = build_self_attn_mask(temporal_valid)

        for spatial_block, temporal_block in zip(
            self.spatial_blocks, self.temporal_blocks, strict=True
        ):
            tokens = torch.cat([slots, camera_tokens], dim=2).flatten(0, 1)
            tokens = spatial_block(
                tokens, freqs_cis=spatial_freqs, attn_mask=spatial_mask
            ).view(batch_size, num_frames, -1, self.hidden_dim)
            slots = tokens[:, :, : self.num_queries] * frame_mask[:, :, None, None]
            camera_tokens = tokens[
                :, :, self.num_queries :
            ] * camera_state_valid.reshape(batch_size, num_frames, -1).unsqueeze(-1)
            temporal = slots.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_queries, num_frames, self.hidden_dim
            )
            temporal = temporal_block(
                temporal, freqs_cis=time_freqs, attn_mask=temporal_mask
            )
            slots = (
                temporal.view(
                    batch_size, self.num_queries, num_frames, self.hidden_dim
                ).permute(0, 2, 1, 3)
                * frame_mask[:, :, None, None]
            )

        slots = self.output_norm(slots)
        return {
            "position": self.position_head(slots),
            "presence_logits": self.presence_head(slots).squeeze(-1),
        }
