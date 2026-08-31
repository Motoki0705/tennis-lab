"""Fixed-query tracking stage with compressed spatial tokens and late writeback."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from src.utils.models.components.block import TransformerBlock
from src.utils.models.components.ffn_layers import build_ffn
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection, MHCState
from src.utils.models.components.norm import RMSNorm


class FixedQueryTrackCompressedStage(nn.Module):
    """Run one fixed-query stage with a compressed object path.

    The stage projects each view's object streams to one temporal token, exposes
    only ``Q + V`` tokens to spatial attention, runs query-temporal attention,
    applies one shared stage-end FFN to query and compressed object tokens, and
    writes the object update back through mHC exactly once at the layer end.
    Attention blocks are deliberately FFN-free.
    """

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
        if type(stage_index) is not int or stage_index < 0:
            raise ValueError("stage_index must be a non-negative int.")
        if type(hidden_dim) is not int or type(num_queries) is not int:
            raise TypeError("hidden_dim and num_queries must be exactly int.")
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

        blocks = (
            object_temporal_block,
            spatial_block,
            query_temporal_block,
        )
        if any(block.cfg.dim != hidden_dim for block in blocks):
            raise ValueError("all block dimensions must equal hidden_dim.")
        ffn_types = {block.cfg.ffn_type for block in blocks}
        if len(ffn_types) != 1:
            raise ValueError("all block FFN types must match.")
        ffn_dims = {block.cfg.ffn_dim for block in blocks}
        if len(ffn_dims) != 1:
            raise ValueError("all block FFN dimensions must match.")
        if any(block.cfg.ffn_enabled for block in blocks):
            raise ValueError("all attention blocks must be FFN-free.")

        self.shared_ffn_norm = RMSNorm(hidden_dim)
        self.shared_ffn = build_ffn(
            ffn_type=ffn_types.pop(),
            dim=hidden_dim,
            ffn_dim=ffn_dims.pop(),
        )
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
        object_tokens = cast(
            Tensor,
            args[0] if args else kwargs["object_tokens"],
        )
        query_tokens = cast(
            Tensor,
            args[1] if len(args) > 1 else kwargs["query_tokens"],
        )
        object_state_valid = cast(Tensor, kwargs["object_state_valid"])
        frame_valid = cast(Tensor, kwargs["frame_valid"])
        spatial_attention_keep_mask = cast(
            Tensor, kwargs["spatial_attention_keep_mask"]
        )
        spatial_freqs = cast(Tensor, kwargs["spatial_freqs"])
        if object_tokens.ndim != 5:
            raise ValueError("object_tokens must have shape (B,V,T,P,D).")
        batch_size, num_views, num_frames, num_streams, hidden_dim = (
            object_tokens.shape
        )
        if (num_streams, hidden_dim) != (self.num_queries, self.hidden_dim):
            raise ValueError("object_tokens do not match the constructed P/D widths.")
        if query_tokens.shape != (
            batch_size,
            num_frames,
            self.num_queries,
            hidden_dim,
        ):
            raise ValueError("query_tokens must have shape (B,T,Q,D).")
        if object_state_valid.shape != object_tokens.shape[:-1]:
            raise ValueError("object_state_valid must match object token axes.")
        if object_state_valid.dtype is not torch.bool:
            raise TypeError("object_state_valid must have dtype torch.bool.")
        if frame_valid.shape != (batch_size, num_frames):
            raise ValueError("frame_valid must have shape (B,T).")
        if frame_valid.dtype is not torch.bool:
            raise TypeError("frame_valid must have dtype torch.bool.")
        spatial_width = self.num_queries + num_views
        expected_spatial_shape = (
            batch_size * num_frames,
            spatial_width,
            spatial_width,
        )
        if spatial_attention_keep_mask.shape != expected_spatial_shape:
            raise ValueError(
                "spatial_attention_keep_mask must have exact shape "
                f"{expected_spatial_shape}, got "
                f"{tuple(spatial_attention_keep_mask.shape)}."
            )
        if spatial_freqs.shape[0:2] != (
            batch_size * num_frames,
            spatial_width,
        ):
            raise ValueError(
                "spatial_freqs must align with the compressed Q+V width."
            )

    def _object_temporal_update(
        self,
        object_values: Tensor,
        *,
        object_temporal_state_valid: Tensor,
        object_temporal_attention_keep_mask: Tensor,
        time_freqs: Tensor,
    ) -> Tensor:
        if self.is_global:
            return self.object_temporal_block.forward_update(
                object_values,
                freqs_cis=time_freqs,
                attn_mask=object_temporal_attention_keep_mask,
            )
        return self.object_temporal_block.forward_update(
            object_values,
            freqs_cis=time_freqs,
            state_valid=object_temporal_state_valid,
        )

    def _query_temporal(
        self,
        query_values: Tensor,
        *,
        query_temporal_state_valid: Tensor,
        query_temporal_attention_keep_mask: Tensor,
        time_freqs: Tensor,
    ) -> Tensor:
        if self.is_global:
            return cast(
                Tensor,
                self.query_temporal_block(
                    query_values,
                    freqs_cis=time_freqs,
                    attn_mask=query_temporal_attention_keep_mask,
                ),
            )
        return cast(
            Tensor,
            self.query_temporal_block(
                query_values,
                freqs_cis=time_freqs,
                state_valid=query_temporal_state_valid,
            ),
        )

    def _write_mhc_update(
        self,
        update: Tensor,
        *,
        residual: Tensor,
        state: MHCState,
    ) -> Tensor:
        """Restore autocast updates to the strict residual-stream dtype."""
        return self.mhc.post(
            update.to(dtype=residual.dtype),
            residual=residual,
            state=state,
        )

    def forward(
        self,
        object_tokens: Tensor,
        query_tokens: Tensor,
        *,
        object_state_valid: Tensor,
        frame_valid: Tensor,
        spatial_attention_keep_mask: Tensor,
        object_temporal_state_valid: Tensor,
        object_temporal_attention_keep_mask: Tensor,
        query_temporal_state_valid: Tensor,
        query_temporal_attention_keep_mask: Tensor,
        spatial_freqs: Tensor,
        time_freqs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return canonical ``(B,V,T,P,D)`` objects and ``(B,T,Q,D)`` queries."""
        batch_size, num_views, num_frames, _, hidden_dim = object_tokens.shape
        compressed_valid = object_state_valid.any(dim=-1)

        projected, mhc_state = self.mhc.pre(object_tokens, object_state_valid)
        object_values = projected.squeeze(-2).reshape(
            batch_size * num_views,
            num_frames,
            hidden_dim,
        )
        object_update = self._object_temporal_update(
            object_values,
            object_temporal_state_valid=object_temporal_state_valid,
            object_temporal_attention_keep_mask=(
                object_temporal_attention_keep_mask
            ),
            time_freqs=time_freqs,
        ).reshape(batch_size, num_views, num_frames, 1, hidden_dim)
        object_update = object_update * compressed_valid.unsqueeze(-1).unsqueeze(-1)

        time_major_objects = object_update.permute(0, 2, 1, 3, 4)
        spatial_values = torch.cat(
            (query_tokens, time_major_objects.flatten(2, 3)),
            dim=2,
        ).flatten(0, 1)
        spatial_values = self.spatial_block(
            spatial_values,
            freqs_cis=spatial_freqs,
            attn_mask=spatial_attention_keep_mask,
        ).reshape(batch_size, num_frames, -1, hidden_dim)
        spatial_queries = spatial_values[:, :, : self.num_queries]
        spatial_queries = spatial_queries * frame_valid[:, :, None, None]
        current_objects = spatial_values[:, :, self.num_queries :].reshape(
            batch_size,
            num_frames,
            num_views,
            1,
            hidden_dim,
        )
        current_objects = current_objects.permute(0, 2, 1, 3, 4)
        current_objects = current_objects * compressed_valid.unsqueeze(-1).unsqueeze(-1)

        query_values = spatial_queries.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_queries,
            num_frames,
            hidden_dim,
        )
        query_values = self._query_temporal(
            query_values,
            query_temporal_state_valid=query_temporal_state_valid,
            query_temporal_attention_keep_mask=query_temporal_attention_keep_mask,
            time_freqs=time_freqs,
        )
        query_output = query_values.reshape(
            batch_size,
            self.num_queries,
            num_frames,
            hidden_dim,
        ).permute(0, 2, 1, 3)
        query_output = query_output * frame_valid[:, :, None, None]

        time_major_objects = current_objects.permute(0, 2, 1, 3, 4)
        shared_values = torch.cat(
            (query_output, time_major_objects.flatten(2, 3)),
            dim=2,
        )
        shared_values = shared_values + self.shared_ffn(
            self.shared_ffn_norm(shared_values)
        )
        query_output = shared_values[:, :, : self.num_queries]
        query_output = query_output * frame_valid[:, :, None, None]
        current_objects = shared_values[:, :, self.num_queries :].reshape(
            batch_size,
            num_frames,
            num_views,
            1,
            hidden_dim,
        )
        current_objects = current_objects.permute(0, 2, 1, 3, 4)
        current_objects = current_objects * compressed_valid.unsqueeze(-1).unsqueeze(-1)

        object_output = self._write_mhc_update(
            current_objects,
            residual=object_tokens,
            state=mhc_state,
        )
        object_output = object_output * object_state_valid.unsqueeze(-1)
        return object_output, query_output


__all__ = ["FixedQueryTrackCompressedStage"]
