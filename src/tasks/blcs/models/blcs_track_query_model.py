"""Persistent track-query model for unordered multi-view ball candidates."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.utils.models import (
    CourtCameraEmbedding,
    InvisibleTokenEmbedding,
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    build_self_attn_mask,
    precompute_freqs_cis,
    precompute_freqs_cis_nd,
)
from src.utils.models.embeddings.projection import apply_visibility_mask


class BLCSTrackQueryModel(nn.Module):
    """Alternate spatial candidate aggregation and per-slot temporal attention."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.num_queries = int(config.num_queries)
        self.num_stages = int(config.num_stages)
        self.role_rope_enabled = bool(config.role_rope_enabled)
        self.mask_invisible_observations = bool(
            config.get("mask_invisible_observations", True)
        )
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = int(config.get("rope_dim", head_dim))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be even and no larger than the attention head dim."
            )

        self.candidate_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=float(config.get("invisible_init_std", 0.02)),
        )
        self.court_encoder = CourtCameraEmbedding(
            dim=self.hidden_dim,
            num_court_points=14,
        )
        self.slot_embeddings = nn.Parameter(
            torch.randn(self.num_queries, self.hidden_dim) * 0.02
        )
        block_config = TransformerBlockConfig(
            dim=self.hidden_dim,
            n_heads=self.num_heads,
            ffn_dim=int(config.ffn_dim),
            rope_dim=self.rope_dim,
            attn_dropout=float(config.dropout),
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
        """Return ``(B*T,Q+V*(D+1),3)`` time/camera/role coordinates."""
        time = torch.arange(num_frames, device=device).view(1, num_frames, 1)
        slot = torch.zeros(
            batch_size, num_frames, num_queries, 3, device=device, dtype=torch.long
        )
        slot[..., 0] = time
        camera_tokens = torch.zeros(
            batch_size,
            num_frames,
            num_views,
            num_detections + 1,
            3,
            device=device,
            dtype=torch.long,
        )
        camera_tokens[..., 0] = time.view(1, num_frames, 1, 1)
        camera = torch.arange(1, num_views + 1, device=device).view(1, 1, num_views, 1)
        camera_tokens[..., 1] = camera
        camera_tokens[..., :num_detections, 2] = 1
        camera_tokens[..., num_detections, 2] = 2
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
        encoded_candidates = self.candidate_encoder(
            ball_uv.masked_fill(~ball_visible.unsqueeze(-1), 0.0)
        )
        candidate_tokens = apply_visibility_mask(
            encoded_candidates,
            ball_visible,
            self.invisible_token,
        )
        if court_kp.shape[:3] != (batch_size, num_views, num_frames):
            raise ValueError("court_kp leading dimensions must match ball_uv (B,V,T).")
        if court_kp.shape[3:] != (14, 2):
            raise ValueError(
                "court_kp must contain all 14 annotated UV points; "
                f"got shape {tuple(court_kp.shape)}."
            )
        court_tokens = self.court_encoder(court_kp, court_vis).unsqueeze(3)
        camera_tokens = torch.cat([candidate_tokens, court_tokens], dim=3).permute(
            0, 2, 1, 3, 4
        )
        camera_tokens = camera_tokens.reshape(
            batch_size,
            num_frames,
            num_views * (num_detections + 1),
            self.hidden_dim,
        )
        camera_padding_valid = torch.cat(
            [
                observation_padding_valid,
                (view_mask[:, :, None] & frame_mask[:, None, :]).unsqueeze(-1),
            ],
            dim=3,
        ).permute(0, 2, 1, 3)
        camera_attention_valid = camera_padding_valid.clone()
        if self.mask_invisible_observations:
            camera_attention_valid[..., :num_detections] &= ball_visible.permute(
                0, 2, 1, 3
            )
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
