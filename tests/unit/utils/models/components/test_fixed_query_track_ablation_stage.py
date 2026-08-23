"""Structural tests for the fixed-query FFN/writeback ablation stage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from src.utils.models.components.block import TransformerBlock
from src.utils.models.components.ffn_layers import SwiGLU
from src.utils.models.components.fixed_query_track_ablation_stage import (
    FFNMode,
    FixedQueryTrackAblationStage,
    MHCWriteback,
)
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection

_CONDITIONS: tuple[tuple[str, FFNMode, MHCWriteback, int], ...] = (
    ("A", "per_attention", "after_object_temporal", 16),
    ("B", "shared", "after_object_temporal", 16),
    ("C", "per_attention", "layer_end", 7),
    ("D", "shared", "layer_end", 7),
)


class _RecordingFFN(nn.Module):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.scale = nn.Parameter(torch.ones(()))
        self.call_count = 0

    def forward(self, values: Tensor) -> Tensor:
        self.call_count += 1
        self.events.append(f"{self.name}.ffn")
        return torch.zeros_like(values) * self.scale


class _RecordingBlock(nn.Module):
    def __init__(
        self,
        *,
        name: str,
        attention_type: str,
        ffn_enabled: bool,
        events: list[str],
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
        self.ffn = _RecordingFFN(name, events) if ffn_enabled else None
        self.calls: list[dict[str, Tensor | None]] = []
        self.output_dtype: torch.dtype | None = None

    def _record(
        self,
        values: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None,
        state_valid: Tensor | None,
    ) -> None:
        self.events.append(f"{self.name}.attention")
        self.calls.append(
            {
                "values": values,
                "freqs_cis": freqs_cis,
                "attn_mask": attn_mask,
                "state_valid": state_valid,
            }
        )
        if self.ffn is not None:
            self.ffn(values)

    def forward_update(
        self,
        values: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None = None,
        state_valid: Tensor | None = None,
    ) -> Tensor:
        self._record(
            values,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        return values.to(dtype=self.output_dtype) if self.output_dtype else values

    def forward(
        self,
        values: Tensor,
        *,
        freqs_cis: Tensor,
        attn_mask: Tensor | None = None,
        state_valid: Tensor | None = None,
    ) -> Tensor:
        self._record(
            values,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        return values.to(dtype=self.output_dtype) if self.output_dtype else values


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
        if update.dtype != residual.dtype:
            raise TypeError("mHC update and residual dtypes must match")
        return residual + update


def _stage(
    *,
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    stage_index: int = 0,
) -> tuple[
    FixedQueryTrackAblationStage,
    _RecordingMHC,
    tuple[_RecordingBlock, _RecordingBlock, _RecordingBlock],
    list[str],
]:
    events: list[str] = []
    temporal_type = "mha" if stage_index % 4 == 3 else "cswa"
    ffn_enabled = ffn_mode == "per_attention"
    mhc = _RecordingMHC(events)
    blocks = (
        _RecordingBlock(
            name="object",
            attention_type=temporal_type,
            ffn_enabled=ffn_enabled,
            events=events,
        ),
        _RecordingBlock(
            name="spatial",
            attention_type="mha",
            ffn_enabled=ffn_enabled,
            events=events,
        ),
        _RecordingBlock(
            name="query",
            attention_type=temporal_type,
            ffn_enabled=ffn_enabled,
            events=events,
        ),
    )
    stage = FixedQueryTrackAblationStage(
        stage_index=stage_index,
        mhc=cast(ManifoldConstrainedHyperConnection, mhc),
        object_temporal_block=cast(TransformerBlock, blocks[0]),
        spatial_block=cast(TransformerBlock, blocks[1]),
        query_temporal_block=cast(TransformerBlock, blocks[2]),
        hidden_dim=8,
        num_queries=4,
        ffn_mode=ffn_mode,
        mhc_writeback=mhc_writeback,
    )
    if stage.shared_ffn is not None:
        stage.shared_ffn.register_forward_pre_hook(
            lambda _module, _args: events.append("shared.ffn")
        )
    return stage, mhc, blocks, events


def _inputs(*, spatial_width: int) -> dict[str, Any]:
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
        "spatial_attention_keep_mask": torch.ones(
            2, spatial_width, spatial_width, dtype=torch.bool
        ),
        "object_temporal_state_valid": object_state_valid[..., 0].reshape(3, 2),
        "object_temporal_attention_keep_mask": torch.ones(
            3, 2, 2, dtype=torch.bool
        ),
        "query_temporal_state_valid": frame_valid[:, None, :]
        .expand(1, 4, 2)
        .reshape(4, 2),
        "query_temporal_attention_keep_mask": torch.ones(
            4, 2, 2, dtype=torch.bool
        ),
        "spatial_freqs": torch.ones(2, spatial_width, 2, dtype=torch.complex64),
        "time_freqs": torch.ones(2, 2, dtype=torch.complex64),
    }


@pytest.mark.parametrize(
    ("condition", "ffn_mode", "mhc_writeback", "spatial_width"),
    _CONDITIONS,
)
def test_four_conditions_have_exact_event_order_width_and_single_writeback(
    condition: str,
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    spatial_width: int,
) -> None:
    del condition
    stage, mhc, blocks, events = _stage(
        ffn_mode=ffn_mode,
        mhc_writeback=mhc_writeback,
    )
    inputs = _inputs(spatial_width=spatial_width)

    object_output, query_output = stage(**inputs)

    per_attention_events = [
        "mhc.pre",
        "object.attention",
        "object.ffn",
        "spatial.attention",
        "spatial.ffn",
        "query.attention",
        "query.ffn",
    ]
    shared_events = [
        "mhc.pre",
        "object.attention",
        "spatial.attention",
        "query.attention",
        "shared.ffn",
    ]
    expected = per_attention_events if ffn_mode == "per_attention" else shared_events
    object_end = 3 if ffn_mode == "per_attention" else 2
    writeback_index = (
        object_end if mhc_writeback == "after_object_temporal" else len(expected)
    )
    expected.insert(writeback_index, "mhc.post")
    assert events == expected
    assert tuple(cast(Tensor, blocks[1].calls[0]["values"]).shape) == (
        2,
        spatial_width,
        8,
    )
    assert blocks[1].calls[0]["attn_mask"] is inputs[
        "spatial_attention_keep_mask"
    ]
    assert blocks[1].calls[0]["freqs_cis"] is inputs["spatial_freqs"]
    assert mhc.post_shapes == [(1, 3, 2, 1, 8)]
    assert object_output.shape == (1, 3, 2, 4, 8)
    assert query_output.shape == (1, 2, 4, 8)
    assert torch.isfinite(object_output).all()
    assert torch.isfinite(query_output).all()


@pytest.mark.parametrize(
    ("condition", "ffn_mode", "mhc_writeback", "spatial_width"),
    _CONDITIONS,
)
def test_writeback_restores_mixed_precision_update_to_residual_dtype(
    condition: str,
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
    spatial_width: int,
) -> None:
    del condition
    stage, mhc, blocks, _ = _stage(
        ffn_mode=ffn_mode,
        mhc_writeback=mhc_writeback,
    )
    inputs = _inputs(spatial_width=spatial_width)
    residual_dtype = (
        torch.float32
        if mhc_writeback == "after_object_temporal"
        else torch.bfloat16
    )
    inputs["object_tokens"] = inputs["object_tokens"].to(dtype=residual_dtype)
    inputs["query_tokens"] = inputs["query_tokens"].to(dtype=residual_dtype)
    if mhc_writeback == "after_object_temporal":
        blocks[0].output_dtype = torch.bfloat16
    else:
        blocks[1].output_dtype = torch.float32

    object_output, _ = stage(**inputs)

    assert mhc.post_dtypes == [(residual_dtype, residual_dtype)]
    assert object_output.dtype == residual_dtype


@pytest.mark.parametrize("ffn_mode", ["per_attention", "shared"])
def test_ffn_module_identity_and_parameter_inventory_are_exact(
    ffn_mode: FFNMode,
) -> None:
    stage, _, blocks, _ = _stage(
        ffn_mode=ffn_mode,
        mhc_writeback="layer_end",
    )

    block_ffns = [block.ffn for block in blocks]
    if ffn_mode == "per_attention":
        assert stage.shared_ffn is None
        assert stage.shared_ffn_norm is None
        assert all(isinstance(ffn, _RecordingFFN) for ffn in block_ffns)
        assert len({id(ffn) for ffn in block_ffns}) == 3
        parameter_ids = [
            {id(parameter) for parameter in cast(nn.Module, ffn).parameters()}
            for ffn in block_ffns
        ]
        assert all(parameter_ids)
        assert parameter_ids[0].isdisjoint(parameter_ids[1])
        assert parameter_ids[0].isdisjoint(parameter_ids[2])
        assert parameter_ids[1].isdisjoint(parameter_ids[2])
    else:
        assert block_ffns == [None, None, None]
        assert isinstance(stage.shared_ffn, SwiGLU)
        assert stage.shared_ffn_norm is not None
        assert any(name.startswith("shared_ffn.") for name, _ in stage.named_parameters())
        assert not any("_block.ffn." in name for name, _ in stage.named_parameters())


def test_global_stage_uses_dense_temporal_masks_without_changing_schedule() -> None:
    stage, _, blocks, events = _stage(
        ffn_mode="shared",
        mhc_writeback="layer_end",
        stage_index=3,
    )
    inputs = _inputs(spatial_width=7)

    stage(**inputs)

    assert events == [
        "mhc.pre",
        "object.attention",
        "spatial.attention",
        "query.attention",
        "shared.ffn",
        "mhc.post",
    ]
    assert blocks[0].calls[0]["attn_mask"] is inputs[
        "object_temporal_attention_keep_mask"
    ]
    assert blocks[0].calls[0]["state_valid"] is None
    assert blocks[2].calls[0]["attn_mask"] is inputs[
        "query_temporal_attention_keep_mask"
    ]
    assert blocks[2].calls[0]["state_valid"] is None


@pytest.mark.parametrize(
    ("axis", "value", "message"),
    [
        ("ffn_mode", "legacy", "ffn_mode must be exactly"),
        ("mhc_writeback", "before_attention", "mhc_writeback must be exactly"),
    ],
)
def test_stage_rejects_unknown_ablation_axis_values(
    axis: str,
    value: str,
    message: str,
) -> None:
    kwargs: dict[str, Any] = {
        "ffn_mode": "per_attention",
        "mhc_writeback": "layer_end",
    }
    kwargs[axis] = value
    events: list[str] = []
    blocks = [
        _RecordingBlock(
            name=name,
            attention_type="mha" if name == "spatial" else "cswa",
            ffn_enabled=True,
            events=events,
        )
        for name in ("object", "spatial", "query")
    ]

    with pytest.raises(ValueError, match=message):
        FixedQueryTrackAblationStage(
            stage_index=0,
            mhc=cast(
                ManifoldConstrainedHyperConnection,
                _RecordingMHC(events),
            ),
            object_temporal_block=cast(TransformerBlock, blocks[0]),
            spatial_block=cast(TransformerBlock, blocks[1]),
            query_temporal_block=cast(TransformerBlock, blocks[2]),
            hidden_dim=8,
            num_queries=4,
            **kwargs,
        )
