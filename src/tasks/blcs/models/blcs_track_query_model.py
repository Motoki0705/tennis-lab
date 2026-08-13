"""Persistent track-query model for ID-ordered multi-view ball observations."""

from __future__ import annotations

import torch
from torch import nn

from src.tasks.blcs.configuration import TrackQueryModelConfig
from src.tasks.blcs.data.tracking_types import BLCSTrackingPrediction
from src.tasks.blcs.models.components.observation_fusion import (
    TrackObservationFusion,
    build_track_observation_fusion,
)
from src.utils.models import (
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.embeddings import ReferenceViewConditioning


class BLCSTrackQueryModel(nn.Module):
    """Alternate spatial candidate aggregation and per-slot temporal attention."""

    def __init__(self, config: TrackQueryModelConfig) -> None:
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.num_queries = int(config.num_queries)
        self.num_stages = int(config.num_stages)
        self.role_rope_scale = int(config.role_rope_enabled)
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = config.rope_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be even and no larger than the attention head dim."
            )

        self.court_observation_profile = config.court_observation_profile
        self.kp7_camera_rope_enabled = config.kp7_camera_rope_enabled
        self.num_court_tokens = (
            14 if self.court_observation_profile == "kp14_reference_baseline" else None
        )
        invisible_init_std = config.invisible_init_std
        self.observation_encoder: TrackObservationFusion = (
            build_track_observation_fusion(
                profile=self.court_observation_profile,
                observation_fusion=config.observation_fusion,
                point_fusion=config.point_fusion,
                hidden_dim=self.hidden_dim,
                invisible_init_std=invisible_init_std,
            )
        )
        self.reference_conditioning = (
            ReferenceViewConditioning(self.hidden_dim)
            if self.court_observation_profile
            in {"kp14_reference_baseline", "kp7_reference"}
            else None
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
        self.spatial_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=10000.0,
            n_axes=3,
        )
        self.temporal_frequency_computer = RotaryFrequencyComputer(
            dim=self.rope_dim,
            base=10000.0,
            n_axes=1,
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
        frame_mask: torch.Tensor,
        observation_state_valid: torch.Tensor,
        spatial_attention_mask: torch.Tensor,
        temporal_attention_mask: torch.Tensor,
        reference_view_mask: torch.Tensor | None = None,
        ball_score: torch.Tensor | None = None,
        court_kp: torch.Tensor | None = None,
        court_vis: torch.Tensor | None = None,
        point_attention_mask: torch.Tensor | None = None,
        court_peak_uv: torch.Tensor | None = None,
        court_peak_score: torch.Tensor | None = None,
        court_peak_covariance: torch.Tensor | None = None,
        court_peak_valid: torch.Tensor | None = None,
    ) -> BLCSTrackingPrediction:
        """Predict clip-local ball tracks.

        Args follow the multi-object contract in ``blcs/README.md``.
        """
        batch_size, num_views, num_frames, num_detections, _ = ball_uv.shape
        camera_tokens, camera_state_valid = self.observation_encoder(
            ball_uv=ball_uv,
            ball_visible=ball_visible,
            state_valid=observation_state_valid,
            ball_score=ball_score,
            court_kp=court_kp,
            court_visible=court_vis,
            point_attention_mask=point_attention_mask,
            court_peak_uv=court_peak_uv,
            court_peak_score=court_peak_score,
            court_peak_covariance=court_peak_covariance,
            court_peak_valid=court_peak_valid,
        )
        if self.reference_conditioning is not None and reference_view_mask is not None:
            camera_tokens = self.reference_conditioning(
                camera_tokens.permute(0, 2, 1, 3, 4),
                reference_view_mask,
            ).permute(0, 2, 1, 3, 4)
        camera_tokens = camera_tokens.reshape(
            batch_size,
            num_frames,
            num_views * num_detections,
            self.hidden_dim,
        )
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
        if (
            self.court_observation_profile != "kp14_reference_baseline"
            and not self.kp7_camera_rope_enabled
        ):
            coordinates[..., 1] = 1
        rope_coordinates = coordinates.clone()
        rope_coordinates[..., 2] *= self.role_rope_scale
        spatial_freqs = self.spatial_frequency_computer(rope_coordinates)
        time_freqs = self.temporal_frequency_computer(
            torch.arange(num_frames, device=ball_uv.device).unsqueeze(-1)
        )
        for spatial_block, temporal_block in zip(
            self.spatial_blocks, self.temporal_blocks, strict=True
        ):
            tokens = torch.cat([slots, camera_tokens], dim=2).flatten(0, 1)
            tokens = spatial_block(
                tokens, freqs_cis=spatial_freqs, attn_mask=spatial_attention_mask
            ).view(batch_size, num_frames, -1, self.hidden_dim)
            slots = tokens[:, :, : self.num_queries] * frame_mask[:, :, None, None]
            camera_tokens = tokens[
                :, :, self.num_queries :
            ] * camera_state_valid.reshape(batch_size, num_frames, -1).unsqueeze(-1)
            temporal = slots.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_queries, num_frames, self.hidden_dim
            )
            temporal = temporal_block(
                temporal, freqs_cis=time_freqs, attn_mask=temporal_attention_mask
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
