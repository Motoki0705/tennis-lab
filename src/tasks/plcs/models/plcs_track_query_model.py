"""Persistent track-query model for ID-ordered multi-view player observations."""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.data.tracking_types import PLCSTrackingPrediction
from src.utils.models import (
    RMSNorm,
    RotaryFrequencyComputer,
    TransformerBlock,
    TransformerBlockConfig,
    build_fixed_query_padding_masks,
)
from src.utils.models.embeddings import (
    CourtPlayerGroupEmbedding,
    InvisibleTokenEmbedding,
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
        self.num_court_tokens = 14
        self.invisible_token = InvisibleTokenEmbedding(
            dim=self.hidden_dim,
            init_std=config.number("invisible_init_std"),
        )
        self.group_embed = CourtPlayerGroupEmbedding(
            dim=self.hidden_dim,
            invisible_token=self.invisible_token,
            num_court_tokens=self.num_court_tokens,
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
        human_vis: torch.Tensor,
        court_kp: torch.Tensor,
        court_vis: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> PLCSTrackingPrediction:
        batch_size, num_views, num_frames, num_queries, num_joints, _ = (
            human_kp.shape
        )
        del num_joints
        if num_queries != self.num_queries:
            raise ValueError(
                "human_kp query axis must equal model.num_queries: "
                f"got {num_queries} and {self.num_queries}."
            )
        masks = build_fixed_query_padding_masks(
            padding_mask,
            num_queries=self.num_queries,
        )
        context_valid = masks.context_valid
        effective_human_vis = human_vis & context_valid[..., None, None]
        effective_court_vis = court_vis & context_valid[..., None]
        observation_visible = effective_human_vis.any(dim=-1)
        masked_court = court_kp.masked_fill(~effective_court_vis.unsqueeze(-1), 0.0)
        court_for_queries = masked_court.unsqueeze(3).expand(
            -1, -1, -1, num_queries, -1, -1
        )
        masked_human = human_kp.masked_fill(~effective_human_vis.unsqueeze(-1), 0.0)
        camera_tokens = self.group_embed(
            court_for_queries,
            masked_human,
            observation_visible,
        ).permute(
            0,
            2,
            1,
            3,
            4,
        )
        camera_tokens = camera_tokens.reshape(
            batch_size,
            num_frames,
            num_views * num_queries,
            self.hidden_dim,
        )
        camera_tokens = camera_tokens * masks.camera_state_valid.permute(
            0, 2, 1, 3
        ).reshape(
            batch_size, num_frames, -1
        ).unsqueeze(-1)

        slots = self.slot_embeddings.view(
            1, 1, self.num_queries, self.hidden_dim
        ).expand(batch_size, num_frames, -1, -1)
        slots = slots * masks.frame_valid[:, :, None, None]
        coordinates = self.build_spatial_coordinates(
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=num_views,
            num_detections=num_queries,
            num_queries=self.num_queries,
            device=human_kp.device,
        )
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
                tokens,
                freqs_cis=spatial_freqs,
                attn_mask=masks.spatial_attention_keep_mask,
            ).view(batch_size, num_frames, -1, self.hidden_dim)
            slots = (
                tokens[:, :, : self.num_queries]
                * masks.frame_valid[:, :, None, None]
            )
            camera_tokens = tokens[
                :, :, self.num_queries :
            ] * masks.camera_state_valid.permute(0, 2, 1, 3).reshape(
                batch_size, num_frames, -1
            ).unsqueeze(-1)
            temporal = slots.permute(0, 2, 1, 3).reshape(
                batch_size * self.num_queries, num_frames, self.hidden_dim
            )
            temporal = temporal_block(
                temporal,
                freqs_cis=temporal_freqs,
                attn_mask=masks.query_temporal_attention_keep_mask,
            )
            slots = (
                temporal.view(
                    batch_size, self.num_queries, num_frames, self.hidden_dim
                ).permute(0, 2, 1, 3)
                * masks.frame_valid[:, :, None, None]
            )

        slots = self.output_norm(slots)
        output_valid = masks.frame_valid[:, :, None]
        return {
            "position": self.position_head(slots) * output_valid.unsqueeze(-1),
            "rotation": F.normalize(self.rotation_head(slots), dim=-1)
            * output_valid.unsqueeze(-1),
            "presence_logits": self.presence_head(slots).squeeze(-1) * output_valid,
        }

    @staticmethod
    def _role_rope_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        return coordinates

    @staticmethod
    def _camera_time_rope_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
        selected = coordinates.clone()
        selected[..., 2] = 0
        return selected
