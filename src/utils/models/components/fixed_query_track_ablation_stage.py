"""Fixed-query tracking stage with strict FFN and mHC ablation axes."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor, nn

from src.utils.models.components.block import TransformerBlock
from src.utils.models.components.ffn_layers import SwiGLU
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection, MHCState
from src.utils.models.components.norm import RMSNorm

FFNMode: TypeAlias = Literal["per_attention", "shared"]
MHCWriteback: TypeAlias = Literal["after_object_temporal", "layer_end"]

_FFN_MODES = frozenset({"per_attention", "shared"})
_MHC_WRITEBACKS = frozenset({"after_object_temporal", "layer_end"})


class FixedQueryTrackAblationStage(nn.Module):
    """Run one strict fixed-query FFN/writeback ablation stage.

    All variants keep the ``C,C,C,G`` temporal cycle. Per-attention
    stages use three normal Transformer blocks. Shared stages use three
    attention-only blocks and exactly one stage-owned pre-norm SwiGLU residual
    over the latest query and object tokens after all attention operations.
    Variant E additionally inserts a separate query-only pre-norm SwiGLU
    residual between spatial and query-temporal attention.
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
        ffn_mode: FFNMode,
        mhc_writeback: MHCWriteback,
        query_ffn_after_spatial: bool,
    ) -> None:
        super().__init__()
        if type(stage_index) is not int or stage_index < 0:
            raise ValueError("stage_index must be a non-negative int.")
        if type(hidden_dim) is not int or type(num_queries) is not int:
            raise TypeError("hidden_dim and num_queries must be exactly int.")
        if hidden_dim <= 0 or num_queries <= 0:
            raise ValueError("hidden_dim and num_queries must be positive.")
        if ffn_mode not in _FFN_MODES:
            raise ValueError(
                "ffn_mode must be exactly 'per_attention' or 'shared'."
            )
        if mhc_writeback not in _MHC_WRITEBACKS:
            raise ValueError(
                "mhc_writeback must be exactly 'after_object_temporal' or "
                "'layer_end'."
            )
        if type(query_ffn_after_spatial) is not bool:
            raise TypeError("query_ffn_after_spatial must be exactly bool.")
        if query_ffn_after_spatial and (
            ffn_mode != "shared" or mhc_writeback != "layer_end"
        ):
            raise ValueError(
                "query_ffn_after_spatial requires shared FFN mode and "
                "layer-end mHC writeback."
            )

        self.stage_index = stage_index
        self.is_global = stage_index % 4 == 3
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.ffn_mode = ffn_mode
        self.mhc_writeback = mhc_writeback
        self.query_ffn_after_spatial_enabled = query_ffn_after_spatial
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
        if any(block.cfg.ffn_type != "swiglu" for block in blocks):
            raise ValueError("ablation stages require SwiGLU FFN configuration.")
        ffn_dims = {block.cfg.ffn_dim for block in blocks}
        if len(ffn_dims) != 1:
            raise ValueError("all block FFN dimensions must match.")
        ffn_dim = ffn_dims.pop()
        expected_block_ffn = self.ffn_mode == "per_attention"
        if any(block.cfg.ffn_enabled is not expected_block_ffn for block in blocks):
            raise ValueError(
                "per_attention requires three FFN-enabled blocks and shared "
                "requires three FFN-disabled blocks."
            )

        self.query_ffn_after_spatial_norm: RMSNorm | None
        self.query_ffn_after_spatial: SwiGLU | None
        if self.query_ffn_after_spatial_enabled:
            self.query_ffn_after_spatial_norm = RMSNorm(hidden_dim)
            self.query_ffn_after_spatial = SwiGLU(hidden_dim, ffn_dim)
        else:
            self.query_ffn_after_spatial_norm = None
            self.query_ffn_after_spatial = None

        self.shared_ffn_norm: RMSNorm | None
        self.shared_ffn: SwiGLU | None
        if self.ffn_mode == "shared":
            self.shared_ffn_norm = RMSNorm(hidden_dim)
            self.shared_ffn = SwiGLU(hidden_dim, ffn_dim)
        else:
            self.shared_ffn_norm = None
            self.shared_ffn = None

        self.register_forward_pre_hook(
            self._validate_forward_inputs,
            with_kwargs=True,
        )

    @property
    def spatial_object_width_per_view(self) -> int:
        """Return the object width contributed by each view to spatial attention."""
        if self.mhc_writeback == "after_object_temporal":
            return self.num_queries
        return 1

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
        spatial_width = self.num_queries + (
            num_views * self.spatial_object_width_per_view
        )
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
                "spatial_freqs must align with the selected Q+object width."
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

        if self.mhc_writeback == "after_object_temporal":
            spatial_objects = self._write_mhc_update(
                object_update,
                residual=object_tokens,
                state=mhc_state,
            )
            spatial_objects = spatial_objects * object_state_valid.unsqueeze(-1)
        else:
            spatial_objects = object_update

        spatial_object_width = spatial_objects.shape[3]
        time_major_objects = spatial_objects.permute(0, 2, 1, 3, 4)
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
        if self.query_ffn_after_spatial_enabled:
            if (
                self.query_ffn_after_spatial is None
                or self.query_ffn_after_spatial_norm is None
            ):
                raise RuntimeError(
                    "query-only post-spatial FFN modules were not constructed"
                )
            spatial_queries = spatial_queries + self.query_ffn_after_spatial(
                self.query_ffn_after_spatial_norm(spatial_queries)
            )
            spatial_queries = spatial_queries * frame_valid[:, :, None, None]
        current_objects = spatial_values[:, :, self.num_queries :].reshape(
            batch_size,
            num_frames,
            num_views,
            spatial_object_width,
            hidden_dim,
        )
        current_objects = current_objects.permute(0, 2, 1, 3, 4)
        if spatial_object_width == self.num_queries:
            current_objects = current_objects * object_state_valid.unsqueeze(-1)
        else:
            current_objects = current_objects * compressed_valid.unsqueeze(-1).unsqueeze(
                -1
            )

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

        if self.ffn_mode == "shared":
            if self.shared_ffn is None or self.shared_ffn_norm is None:
                raise RuntimeError("shared FFN modules were not constructed")
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
                spatial_object_width,
                hidden_dim,
            )
            current_objects = current_objects.permute(0, 2, 1, 3, 4)
            if spatial_object_width == self.num_queries:
                current_objects = current_objects * object_state_valid.unsqueeze(-1)
            else:
                current_objects = current_objects * compressed_valid.unsqueeze(
                    -1
                ).unsqueeze(-1)

        if self.mhc_writeback == "layer_end":
            object_output = self._write_mhc_update(
                current_objects,
                residual=object_tokens,
                state=mhc_state,
            )
            object_output = object_output * object_state_valid.unsqueeze(-1)
        else:
            object_output = current_objects
        return object_output, query_output


__all__ = ["FFNMode", "FixedQueryTrackAblationStage", "MHCWriteback"]
