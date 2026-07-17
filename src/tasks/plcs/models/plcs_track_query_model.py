"""Persistent track-query model for unordered multi-view player detections."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.plcs.data.tracking_types import PLCSTrackingPrediction
from src.utils.models import (
    RMSNorm,
    TransformerBlock,
    TransformerBlockConfig,
    precompute_freqs_cis,
    precompute_freqs_cis_nd,
)


class PLCSTrackQueryModel(nn.Module):
    """Alternate unified spatial and per-slot temporal self-attention."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.num_queries = int(config.num_queries)
        self.num_stages = int(config.num_stages)
        self.num_joints = int(config.num_joints)
        self.role_rope_enabled = bool(config.role_rope_enabled)
        head_dim = self.hidden_dim // self.num_heads
        self.rope_dim = int(config.get("rope_dim", head_dim))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.rope_dim > head_dim or self.rope_dim % 2:
            raise ValueError(
                "rope_dim must be even and no larger than the attention head dim."
            )
        input_dim = self.num_joints * 3 + 5
        self.detection_encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
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
        observation = torch.zeros(
            batch_size,
            num_frames,
            num_views,
            num_detections,
            3,
            device=device,
            dtype=torch.long,
        )
        observation[..., 0] = time.view(1, num_frames, 1, 1)
        observation[..., 1] = torch.arange(1, num_views + 1, device=device).view(
            1, 1, num_views, 1
        )
        observation[..., 2] = 1
        return torch.cat([slot, observation.flatten(2, 3)], dim=2).flatten(0, 1)

    @staticmethod
    def _attention_mask(valid: torch.Tensor) -> torch.Tensor:
        return valid[:, None, :].expand(-1, valid.size(1), -1)

    def forward(
        self,
        human_kp: torch.Tensor,
        human_vis: torch.Tensor,
        detection_mask: torch.Tensor,
        detection_score: torch.Tensor,
        bbox: torch.Tensor,
        frame_mask: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> PLCSTrackingPrediction:
        batch_size, num_views, num_frames, num_detections, num_joints, _ = (
            human_kp.shape
        )
        if num_joints != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {num_joints}.")
        keypoint_features = torch.cat(
            [human_kp, human_vis.float().unsqueeze(-1)], dim=-1
        ).flatten(-2)
        detection_features = torch.cat(
            [keypoint_features, detection_score.unsqueeze(-1), bbox], dim=-1
        )
        observations = self.detection_encoder(detection_features)
        observations = observations.permute(0, 2, 1, 3, 4).reshape(
            batch_size, num_frames, num_views * num_detections, self.hidden_dim
        )
        observation_valid = (
            (
                detection_mask
                & human_vis.any(-1)
                & view_mask[:, :, None, None]
                & frame_mask[:, None, :, None]
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, num_frames, -1)
        )
        observations = observations * observation_valid.unsqueeze(-1)
        slots = self.slot_embeddings.view(
            1, 1, self.num_queries, self.hidden_dim
        ).expand(batch_size, num_frames, -1, -1)
        coordinates = self.build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_detections=num_detections,
            num_queries=self.num_queries,
            device=human_kp.device,
        )
        rope_coordinates = coordinates
        if not self.role_rope_enabled:
            rope_coordinates = coordinates.clone()
            rope_coordinates[..., 2] = 0
        spatial_freqs = precompute_freqs_cis_nd(dim=self.rope_dim, pos=rope_coordinates)
        temporal_freqs = precompute_freqs_cis(
            dim=self.rope_dim, seqlen=num_frames, device=human_kp.device
        )
        slot_valid = torch.ones(
            batch_size,
            num_frames,
            self.num_queries,
            device=human_kp.device,
            dtype=torch.bool,
        )
        spatial_valid = torch.cat([slot_valid, observation_valid], dim=2).flatten(0, 1)
        spatial_mask = self._attention_mask(spatial_valid)
        temporal_valid = (
            frame_mask[:, None]
            .expand(-1, self.num_queries, -1)
            .reshape(batch_size * self.num_queries, num_frames)
        )
        temporal_mask = self._attention_mask(temporal_valid)
        for spatial_block, temporal_block in zip(
            self.spatial_blocks, self.temporal_blocks, strict=True
        ):
            tokens = torch.cat([slots, observations], dim=2).flatten(0, 1)
            tokens = spatial_block(
                tokens, freqs_cis=spatial_freqs, attn_mask=spatial_mask
            ).view(batch_size, num_frames, -1, self.hidden_dim)
            slots = tokens[:, :, : self.num_queries]
            observations = tokens[
                :, :, self.num_queries :
            ] * observation_valid.unsqueeze(-1)
            temporal = slots.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_queries, num_frames, self.hidden_dim
            )
            temporal = temporal_block(
                temporal, freqs_cis=temporal_freqs, attn_mask=temporal_mask
            )
            slots = temporal.view(
                batch_size, self.num_queries, num_frames, self.hidden_dim
            ).permute(0, 2, 1, 3)

        slots = self.output_norm(slots)
        return {
            "position": self.position_head(slots),
            "rotation": F.normalize(self.rotation_head(slots), dim=-1),
            "presence_logits": self.presence_head(slots).squeeze(-1),
        }
