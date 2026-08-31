"""Tests for the shared fixed-query multi-view tracking stage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from src.utils.models import FixedQueryTrackStage
from src.utils.models.components import TransformerBlock
from src.utils.models.components.mhc import ManifoldConstrainedHyperConnection


class _RecordingMHC(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        self.valid_mask: torch.Tensor | None = None
        self.post_dtypes: list[tuple[torch.dtype, torch.dtype]] = []

    def pre(
        self,
        streams: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, object]:
        self.events.append("mhc.pre")
        self.valid_mask = valid_mask
        return streams.mean(dim=-2, keepdim=True), object()

    def post(
        self,
        update: torch.Tensor,
        *,
        residual: torch.Tensor,
        state: object,
    ) -> torch.Tensor:
        del state
        self.events.append("mhc.post")
        self.post_dtypes.append((update.dtype, residual.dtype))
        return residual


class _RecordingBlock(nn.Module):
    def __init__(
        self,
        attention_type: str,
        name: str,
        events: list[str],
    ) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(attention_type=attention_type)
        self.name = name
        self.events = events
        self.calls: list[dict[str, torch.Tensor | None]] = []
        self.output_dtype: torch.dtype | None = None
        self.forward_update_output_dtypes: list[torch.dtype] = []

    def _record(
        self,
        operation: str,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None,
        state_valid: torch.Tensor | None,
    ) -> None:
        self.events.append(f"{self.name}.{operation}")
        self.calls.append(
            {
                "freqs_cis": freqs_cis,
                "attn_mask": attn_mask,
                "state_valid": state_valid,
            }
        )

    def forward_update(
        self,
        values: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._record(
            "forward_update",
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        update = torch.zeros_like(values, dtype=self.output_dtype)
        self.forward_update_output_dtypes.append(update.dtype)
        return update

    def forward(
        self,
        values: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._record(
            "forward",
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            state_valid=state_valid,
        )
        return values


def _stage(
    stage_index: int,
) -> tuple[
    FixedQueryTrackStage,
    _RecordingMHC,
    _RecordingBlock,
    _RecordingBlock,
    _RecordingBlock,
    list[str],
]:
    events: list[str] = []
    temporal_type = "mha" if stage_index % 4 == 3 else "cswa"
    mhc = _RecordingMHC(events)
    object_temporal = _RecordingBlock(temporal_type, "object", events)
    spatial = _RecordingBlock("mha", "spatial", events)
    query_temporal = _RecordingBlock(temporal_type, "query", events)
    stage = FixedQueryTrackStage(
        stage_index=stage_index,
        mhc=cast(ManifoldConstrainedHyperConnection, mhc),
        object_temporal_block=cast(TransformerBlock, object_temporal),
        spatial_block=cast(TransformerBlock, spatial),
        query_temporal_block=cast(TransformerBlock, query_temporal),
        hidden_dim=4,
        num_queries=2,
    )
    return stage, mhc, object_temporal, spatial, query_temporal, events


def _inputs() -> dict[str, Any]:
    object_state_valid = torch.tensor(
        [[[[True, True], [False, False], [False, False]],
          [[False, False], [True, True], [False, False]]]]
    )
    frame_valid = object_state_valid.any(dim=1).any(dim=-1)
    return {
        "object_tokens": torch.randn(1, 2, 3, 2, 4),
        "query_tokens": torch.randn(1, 3, 2, 4),
        "object_state_valid": object_state_valid,
        "frame_valid": frame_valid,
        "spatial_attention_keep_mask": torch.ones(3, 6, 6, dtype=torch.bool),
        "object_temporal_state_valid": object_state_valid[..., 0].reshape(2, 3),
        "object_temporal_attention_keep_mask": torch.ones(
            2, 3, 3, dtype=torch.bool
        ),
        "query_temporal_state_valid": frame_valid[:, None, :]
        .expand(1, 2, 3)
        .reshape(2, 3),
        "query_temporal_attention_keep_mask": torch.ones(
            2, 3, 3, dtype=torch.bool
        ),
        "spatial_freqs": torch.empty(0),
        "time_freqs": torch.empty(0),
    }


@pytest.mark.parametrize("stage_index", [0, 1, 2])
def test_cswa_stages_use_raw_temporal_state_validity(stage_index: int) -> None:
    stage, mhc, object_temporal, spatial, query_temporal, events = _stage(stage_index)
    inputs = _inputs()

    object_output, query_output = stage(**inputs)

    assert events == [
        "mhc.pre",
        "object.forward_update",
        "mhc.post",
        "spatial.forward",
        "query.forward",
    ]
    assert mhc.valid_mask is inputs["object_state_valid"]
    assert object_temporal.calls[0]["state_valid"] is inputs[
        "object_temporal_state_valid"
    ]
    assert object_temporal.calls[0]["attn_mask"] is None
    assert spatial.calls[0]["attn_mask"] is inputs[
        "spatial_attention_keep_mask"
    ]
    assert query_temporal.calls[0]["state_valid"] is inputs[
        "query_temporal_state_valid"
    ]
    assert query_temporal.calls[0]["attn_mask"] is None
    assert not object_output[:, :, -1].any()
    assert not query_output[:, -1].any()
    assert torch.isfinite(object_output).all()
    assert torch.isfinite(query_output).all()


def test_global_stage_uses_dense_temporal_keep_masks() -> None:
    stage, _, object_temporal, spatial, query_temporal, events = _stage(3)
    inputs = _inputs()

    stage(**inputs)

    assert events == [
        "mhc.pre",
        "object.forward_update",
        "mhc.post",
        "spatial.forward",
        "query.forward",
    ]
    assert object_temporal.calls[0]["attn_mask"] is inputs[
        "object_temporal_attention_keep_mask"
    ]
    assert object_temporal.calls[0]["state_valid"] is None
    assert spatial.calls[0]["attn_mask"] is inputs[
        "spatial_attention_keep_mask"
    ]
    assert query_temporal.calls[0]["attn_mask"] is inputs[
        "query_temporal_attention_keep_mask"
    ]
    assert query_temporal.calls[0]["state_valid"] is None


@pytest.mark.parametrize("temporal_dtype", [torch.bfloat16, torch.float16])
def test_mhc_writeback_casts_temporal_update_to_object_token_dtype(
    temporal_dtype: torch.dtype,
) -> None:
    stage, mhc, object_temporal, _, _, _ = _stage(0)
    inputs = _inputs()
    object_tokens = cast(torch.Tensor, inputs["object_tokens"])
    object_temporal.output_dtype = temporal_dtype

    stage(**inputs)

    assert object_tokens.dtype is torch.float32
    assert object_temporal.forward_update_output_dtypes == [temporal_dtype]
    assert mhc.post_dtypes == [(torch.float32, torch.float32)]


@pytest.mark.parametrize(
    ("stage_index", "temporal_type"),
    [(0, "mha"), (3, "cswa")],
)
def test_rejects_temporal_block_that_breaks_fixed_cycle(
    stage_index: int,
    temporal_type: str,
) -> None:
    events: list[str] = []
    with pytest.raises(ValueError, match="fixed stage mode"):
        FixedQueryTrackStage(
            stage_index=stage_index,
            mhc=cast(ManifoldConstrainedHyperConnection, _RecordingMHC(events)),
            object_temporal_block=cast(
                TransformerBlock,
                _RecordingBlock(temporal_type, "object", events),
            ),
            spatial_block=cast(
                TransformerBlock,
                _RecordingBlock("mha", "spatial", events),
            ),
            query_temporal_block=cast(
                TransformerBlock,
                _RecordingBlock(temporal_type, "query", events),
            ),
            hidden_dim=4,
            num_queries=2,
        )
