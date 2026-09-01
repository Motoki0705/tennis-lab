"""Structural tests for the canonical fixed-query tracking stage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from src.utils.models.components.block import TransformerBlock
from src.utils.models.components.ffn_layers import SwiGLU
from src.utils.models.components.fixed_query_track_stage import FixedQueryTrackStage
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection


class _RecordingBlock(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        attention_type: str,
        events: list[str],
        ffn_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.cfg = SimpleNamespace(
            attention_type=attention_type,
            dim=8,
            ffn_type="swiglu",
            ffn_dim=16,
            ffn_enabled=ffn_enabled,
        )
        self.ffn = nn.Identity() if ffn_enabled else None
        self.calls: list[dict[str, Tensor | None]] = []
        self.output_dtype: torch.dtype | None = None

    def _run(
        self,
        values: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None,
        state_valid: Tensor | None,
    ) -> Tensor:
        self.events.append(f"{self.name}.attention")
        self.calls.append(
            {
                "values": values,
                "freqs_cis": freqs_cis,
                "attn_mask": attn_mask,
                "state_valid": state_valid,
            }
        )
        return values.to(dtype=self.output_dtype) if self.output_dtype else values

    def forward_update(
        self,
        values: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None = None,
        state_valid: Tensor | None = None,
    ) -> Tensor:
        return self._run(
            values,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )

    def forward(
        self,
        values: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None = None,
        state_valid: Tensor | None = None,
    ) -> Tensor:
        return self._run(
            values,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )


class _RecordingMHC(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.post_shapes: list[tuple[int, ...]] = []
        self.post_dtypes: list[tuple[torch.dtype, torch.dtype]] = []

    def pre(self, streams: Tensor, valid_mask: Tensor) -> tuple[Tensor, object]:
        del valid_mask
        self.events.append("mhc.pre")
        return streams.mean(dim=-2, keepdim=True), object()

    def post(
        self,
        update: Tensor,
        *,
        residual: Tensor,
        state: object,
    ) -> Tensor:
        del state
        self.events.append("mhc.post")
        self.post_shapes.append(tuple(update.shape))
        self.post_dtypes.append((update.dtype, residual.dtype))
        return residual + update


def _stage(
    *,
    stage_index: int = 0,
    ffn_enabled: bool = False,
) -> tuple[
    FixedQueryTrackStage,
    _RecordingMHC,
    tuple[_RecordingBlock, _RecordingBlock, _RecordingBlock],
    list[str],
]:
    events: list[str] = []
    temporal_type = "mha" if stage_index % 4 == 3 else "cswa"
    mhc = _RecordingMHC(events)
    blocks = (
        _RecordingBlock(
            name="object",
            attention_type=temporal_type,
            events=events,
            ffn_enabled=ffn_enabled,
        ),
        _RecordingBlock(
            name="spatial",
            attention_type="mha",
            events=events,
            ffn_enabled=ffn_enabled,
        ),
        _RecordingBlock(
            name="query",
            attention_type=temporal_type,
            events=events,
            ffn_enabled=ffn_enabled,
        ),
    )
    stage = FixedQueryTrackStage(
        stage_index=stage_index,
        mhc=cast(ManifoldConstrainedHyperConnection, mhc),
        object_temporal_block=cast(TransformerBlock, blocks[0]),
        spatial_block=cast(TransformerBlock, blocks[1]),
        query_temporal_block=cast(TransformerBlock, blocks[2]),
        hidden_dim=8,
        num_queries=4,
    )
    stage.shared_ffn.register_forward_pre_hook(
        lambda _module, _args: events.append("shared.ffn")
    )
    return stage, mhc, blocks, events


def _inputs() -> dict[str, Any]:
    object_state_valid = torch.tensor(
        [
            [
                [[True] * 4, [False] * 4],
                [[False] * 4, [True] * 4],
                [[True] * 4, [False] * 4],
            ]
        ]
    )
    frame_valid = object_state_valid.any(dim=1).any(dim=-1)
    return {
        "object_tokens": torch.randn(1, 3, 2, 4, 8),
        "query_tokens": torch.randn(1, 2, 4, 8),
        "object_state_valid": object_state_valid,
        "frame_valid": frame_valid,
        "spatial_attention_keep_mask": torch.ones(2, 7, 7, dtype=torch.bool),
        "object_temporal_state_valid": object_state_valid[..., 0].reshape(3, 2),
        "object_temporal_attention_keep_mask": torch.ones(3, 2, 2, dtype=torch.bool),
        "query_temporal_state_valid": frame_valid[:, None, :]
        .expand(1, 4, 2)
        .reshape(4, 2),
        "query_temporal_attention_keep_mask": torch.ones(4, 2, 2, dtype=torch.bool),
        "spatial_freqs": torch.ones(2, 7, 2, dtype=torch.complex64),
        "time_freqs": torch.ones(2, 2, dtype=torch.complex64),
    }


def test_stage_has_fixed_compressed_width_shared_ffn_and_late_writeback() -> None:
    stage, mhc, blocks, events = _stage()
    inputs = _inputs()

    object_output, query_output = stage(**inputs)

    assert events == [
        "mhc.pre",
        "object.attention",
        "spatial.attention",
        "query.attention",
        "shared.ffn",
        "mhc.post",
    ]
    assert tuple(cast(Tensor, blocks[1].calls[0]["values"]).shape) == (2, 7, 8)
    assert blocks[1].calls[0]["attn_mask"] is inputs["spatial_attention_keep_mask"]
    assert mhc.post_shapes == [(1, 3, 2, 1, 8)]
    assert all(block.ffn is None for block in blocks)
    assert isinstance(stage.shared_ffn, SwiGLU)
    assert object_output.shape == (1, 3, 2, 4, 8)
    assert query_output.shape == (1, 2, 4, 8)
    assert torch.isfinite(object_output).all()
    assert torch.isfinite(query_output).all()


def test_late_writeback_restores_update_to_residual_dtype() -> None:
    stage, mhc, blocks, _ = _stage()
    inputs = _inputs()
    inputs["object_tokens"] = inputs["object_tokens"].to(dtype=torch.bfloat16)
    inputs["query_tokens"] = inputs["query_tokens"].to(dtype=torch.bfloat16)
    blocks[1].output_dtype = torch.float32

    object_output, _ = stage(**inputs)

    assert mhc.post_dtypes == [(torch.bfloat16, torch.bfloat16)]
    assert object_output.dtype == torch.bfloat16


def test_global_stage_uses_dense_temporal_masks() -> None:
    stage, _, blocks, events = _stage(stage_index=3)
    inputs = _inputs()

    stage(**inputs)

    assert events == [
        "mhc.pre",
        "object.attention",
        "spatial.attention",
        "query.attention",
        "shared.ffn",
        "mhc.post",
    ]
    assert (
        blocks[0].calls[0]["attn_mask"] is inputs["object_temporal_attention_keep_mask"]
    )
    assert blocks[0].calls[0]["state_valid"] is None
    assert (
        blocks[2].calls[0]["attn_mask"] is inputs["query_temporal_attention_keep_mask"]
    )
    assert blocks[2].calls[0]["state_valid"] is None


def test_stage_rejects_attention_blocks_with_local_ffns() -> None:
    with pytest.raises(ValueError, match="attention blocks must be FFN-free"):
        _stage(ffn_enabled=True)


def test_stage_rejects_non_compressed_spatial_width() -> None:
    stage, _, _, _ = _stage()
    inputs = _inputs()
    inputs["spatial_attention_keep_mask"] = torch.ones(2, 16, 16, dtype=torch.bool)

    with pytest.raises(ValueError, match="exact shape"):
        stage(**inputs)
