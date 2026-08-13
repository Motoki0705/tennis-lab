"""Persistent track-query model for ID-ordered multi-view player observations."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.data.tracking_types import PLCSTrackingPrediction
from src.tasks.plcs.models.components.observation_fusion import (
    KP7PlayerObservationFusion,
    KP14PlayerObservationFusion,
    PlayerObservationFusion,
    build_player_observation_fusion,
)
from src.utils.models import (
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
)
from src.utils.models.embeddings import (
    ReferenceViewConditioning,
)


class PLCSTrackQueryModel(nn.Module):
    """Alternate unified spatial and per-slot temporal self-attention."""

    def __init__(self, config: PLCSModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.integer("hidden_dim")
        self.num_heads = config.integer("num_heads")
        self.num_queries = config.integer("num_queries")
        self.num_stages = config.integer("num_stages")
        self.num_joints = config.integer("num_joints")
        self.role_rope_enabled = config.boolean("role_rope_enabled")
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = config.integer("rope_dim")
        self._select_rope_coordinates = (
            self._role_rope_coordinates
            if self.role_rope_enabled
            else self._camera_time_rope_coordinates
        )
        self.court_observation_profile = config.string("court_observation_profile")
        self.kp7_camera_rope_enabled = config.boolean("kp7_camera_rope_enabled")
        self.num_court_tokens = (
            14 if self.court_observation_profile == "kp14_reference_baseline" else None
        )
        self.observation_encoder: PlayerObservationFusion = (
            build_player_observation_fusion(
                profile=self.court_observation_profile,
                hidden_dim=self.hidden_dim,
                player_feature_dim=self.num_joints * 3 + 1,
                invisible_init_std=config.number("invisible_init_std"),
            )
        )
        self.invisible_token = self.observation_encoder.invisible_token
        self.group_embed = (
            self.observation_encoder.group_embed
            if isinstance(self.observation_encoder, KP14PlayerObservationFusion)
            else None
        )
        self.kp7_observation_encoder = (
            self.observation_encoder
            if isinstance(self.observation_encoder, KP7PlayerObservationFusion)
            else None
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
            ffn_dim=config.integer("ffn_dim"),
            head_dim=head_dim,
            rope_dim=self.rope_dim,
            attn_dropout=config.number("dropout"),
            attention_type="mha",
            n_kv_heads=None,
            rope_base=config.number("rope_theta"),
            ffn_type=cast(Literal["swiglu", "mlp"], config.string("ffn_type")),
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
        self.rotation_head = nn.Linear(self.hidden_dim, 2)
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
        camera_tokens[..., 1] = torch.arange(1, num_views + 1, device=device).view(
            1, 1, num_views, 1
        )
        camera_tokens[..., 2] = 1
        return torch.cat([slot, camera_tokens.flatten(2, 3)], dim=2).flatten(0, 1)

    def forward(
        self,
        human_kp: torch.Tensor,
        detection_mask: torch.Tensor,
        frame_mask: torch.Tensor,
        camera_state_valid: torch.Tensor,
        spatial_attention_mask: torch.Tensor,
        temporal_attention_mask: torch.Tensor,
        reference_view_mask: torch.Tensor | None = None,
        court_kp: torch.Tensor | None = None,
        court_vis: torch.Tensor | None = None,
        court_peak_uv: torch.Tensor | None = None,
        court_peak_score: torch.Tensor | None = None,
        court_peak_covariance: torch.Tensor | None = None,
        court_peak_valid: torch.Tensor | None = None,
        player_anchor: torch.Tensor | None = None,
        player_features: torch.Tensor | None = None,
    ) -> PLCSTrackingPrediction:
        batch_size, num_views, num_frames, num_detections, num_joints, _ = (
            human_kp.shape
        )
        del num_joints
        camera_tokens, _ = self.observation_encoder(
            human_kp=human_kp,
            detection_mask=detection_mask,
            camera_state_valid=camera_state_valid,
            court_kp=court_kp,
            court_vis=court_vis,
            court_peak_uv=court_peak_uv,
            court_peak_score=court_peak_score,
            court_peak_covariance=court_peak_covariance,
            court_peak_valid=court_peak_valid,
            player_anchor=player_anchor,
            player_features=player_features,
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
            device=human_kp.device,
        )
        if (
            self.court_observation_profile != "kp14_reference_baseline"
            and not self.kp7_camera_rope_enabled
        ):
            coordinates[..., 1] = 1
        rope_coordinates = self._select_rope_coordinates(coordinates)
        spatial_freqs = self.spatial_frequency_computer(rope_coordinates)
        temporal_freqs = self.temporal_frequency_computer(
            torch.arange(num_frames, device=human_kp.device).unsqueeze(-1)
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
                temporal, freqs_cis=temporal_freqs, attn_mask=temporal_attention_mask
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
            "rotation": F.normalize(self.rotation_head(slots), dim=-1),
            "presence_logits": self.presence_head(slots).squeeze(-1),
        }

    @staticmethod
    def _role_rope_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates

    @staticmethod
    def _camera_time_rope_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        selected = coordinates.clone()
        selected[..., 2] = 0
        return selected
