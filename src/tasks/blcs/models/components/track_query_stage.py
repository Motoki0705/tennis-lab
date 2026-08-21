"""One fixed-mode mHC/spatial/temporal BLCS track-query stage."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.utils.models import TransformerBlock
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection


class BLCSTrackQueryStage(nn.Module):
    """Run object temporal, global spatial, then same-mode query temporal work."""

    def __init__(
        self,
        *,
        stage_index: int,
        mhc: ManifoldConstrainedHyperConnection,
        object_temporal_block: TransformerBlock,
        spatial_block: TransformerBlock,
        query_temporal_block: TransformerBlock,
        hidden_dim: int,
        num_queries: int,
    ) -> None:
        super().__init__()
        if stage_index < 0:
            raise ValueError("stage_index must be non-negative.")
        if hidden_dim <= 0 or num_queries <= 0:
            raise ValueError("hidden_dim and num_queries must be positive.")
        self.stage_index = stage_index
        self.is_global = stage_index % 4 == 3
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.mhc = mhc
        self.object_temporal_block = object_temporal_block
        self.spatial_block = spatial_block
        self.query_temporal_block = query_temporal_block

        expected_temporal = "mha" if self.is_global else "cswa"
        if object_temporal_block.cfg.attention_type != expected_temporal:
            raise ValueError(
                "object temporal block does not match the fixed stage mode."
            )
        if query_temporal_block.cfg.attention_type != expected_temporal:
            raise ValueError(
                "query temporal block does not match the fixed stage mode."
            )
        if spatial_block.cfg.attention_type != "mha":
            raise ValueError("spatial attention must always use global MHA.")
        self.register_forward_pre_hook(
            self._validate_forward_inputs,
            with_kwargs=True,
        )

    def _validate_forward_inputs(
        self,
        _module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        camera_tokens = cast(
            Tensor,
            args[0] if args else kwargs["camera_tokens"],
        )
        slots = cast(
            Tensor,
            args[1] if len(args) > 1 else kwargs["slots"],
        )
        camera_state_valid = cast(Tensor, kwargs["camera_state_valid"])
        frame_valid = cast(Tensor, kwargs["frame_valid"])
        if camera_tokens.ndim != 5:
            raise ValueError("camera_tokens must have shape (B,V,T,Q,D).")
        batch_size, _, num_frames, num_queries, hidden_dim = camera_tokens.shape
        if (num_queries, hidden_dim) != (self.num_queries, self.hidden_dim):
            raise ValueError("camera_tokens do not match the constructed Q/D widths.")
        if slots.shape != (batch_size, num_frames, num_queries, hidden_dim):
            raise ValueError("slots must have shape (B,T,Q,D).")
        if camera_state_valid.shape != camera_tokens.shape[:-1]:
            raise ValueError("camera_state_valid must match camera token axes.")
        if frame_valid.shape != (batch_size, num_frames):
            raise ValueError("frame_valid must have shape (B,T).")

    def forward(
        self,
        camera_tokens: Tensor,
        slots: Tensor,
        *,
        camera_state_valid: Tensor,
        frame_valid: Tensor,
        spatial_attention_keep_mask: Tensor,
        object_temporal_state_valid: Tensor,
        object_temporal_attention_keep_mask: Tensor,
        query_temporal_state_valid: Tensor,
        query_temporal_attention_keep_mask: Tensor,
        spatial_freqs: Tensor,
        time_freqs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return canonical ``[B,V,T,Q,D]`` camera and ``[B,T,Q,D]`` slots."""
        batch_size, num_views, num_frames, num_queries, hidden_dim = camera_tokens.shape

        projected, mhc_state = self.mhc.pre(camera_tokens, camera_state_valid)
        object_values = projected.squeeze(-2).reshape(
            batch_size * num_views, num_frames, hidden_dim
        )
        if self.is_global:
            object_update = self.object_temporal_block.forward_update(
                object_values,
                freqs_cis=time_freqs,
                attn_mask=object_temporal_attention_keep_mask,
            )
        else:
            object_update = self.object_temporal_block.forward_update(
                object_values,
                freqs_cis=time_freqs,
                state_valid=object_temporal_state_valid,
            )
        object_update = object_update.reshape(
            batch_size, num_views, num_frames, 1, hidden_dim
        )
        camera_object = self.mhc.post(
            object_update,
            residual=camera_tokens,
            state=mhc_state,
        )
        camera_object = camera_object * camera_state_valid.unsqueeze(-1)

        time_major_camera = camera_object.permute(0, 2, 1, 3, 4)
        spatial_values = torch.cat(
            (slots, time_major_camera.flatten(2, 3)),
            dim=2,
        ).flatten(0, 1)
        spatial_values = self.spatial_block(
            spatial_values,
            freqs_cis=spatial_freqs,
            attn_mask=spatial_attention_keep_mask,
        ).reshape(batch_size, num_frames, -1, hidden_dim)
        spatial_slots = spatial_values[:, :, :num_queries]
        spatial_slots = spatial_slots * frame_valid[:, :, None, None]
        camera_output = spatial_values[:, :, num_queries:].reshape(
            batch_size, num_frames, num_views, num_queries, hidden_dim
        )
        camera_output = camera_output.permute(0, 2, 1, 3, 4)
        camera_output = camera_output * camera_state_valid.unsqueeze(-1)

        query_values = spatial_slots.permute(0, 2, 1, 3).reshape(
            batch_size * num_queries, num_frames, hidden_dim
        )
        if self.is_global:
            query_values = self.query_temporal_block(
                query_values,
                freqs_cis=time_freqs,
                attn_mask=query_temporal_attention_keep_mask,
            )
        else:
            query_values = self.query_temporal_block(
                query_values,
                freqs_cis=time_freqs,
                state_valid=query_temporal_state_valid,
            )
        query_output = query_values.reshape(
            batch_size, num_queries, num_frames, hidden_dim
        ).permute(0, 2, 1, 3)
        query_output = query_output * frame_valid[:, :, None, None]
        return camera_output, query_output


__all__ = ["BLCSTrackQueryStage"]
