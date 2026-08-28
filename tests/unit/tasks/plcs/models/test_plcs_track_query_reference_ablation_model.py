"""PLCS reference-conditioned generic ablation contract tests."""

from __future__ import annotations

import inspect
from typing import TypedDict, cast

import pytest
import torch
from torch import Tensor

from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.models.plcs_track_query_ablation_model import (
    PLCSTrackQueryAblationModel,
)
from src.tasks.plcs.models.plcs_track_query_reference_ablation_model import (
    PLCSTrackQueryReferenceAblationModel,
)
from src.utils.models.components.fixed_query_track_ablation_stage import (
    FFNMode,
    MHCWriteback,
)


class _SpatialCoordinateArgs(TypedDict):
    num_frames: int
    num_views: int
    num_detections: int
    num_queries: int
    mhc_writeback: MHCWriteback


def _raw_config(
    *,
    selector_mode: str = "reference",
    ffn_mode: FFNMode = "shared",
    mhc_writeback: MHCWriteback = "layer_end",
    rope_dim: int = 6,
) -> dict[str, object]:
    return {
        "name": "plcs_track_query_reference_ablation",
        "hidden_dim": 24,
        "num_heads": 4,
        "ffn_dim": 48,
        "num_queries": 2,
        "num_stages": 4,
        "num_joints": 17,
        "rope_dim": rope_dim,
        "rope_theta": 10_000.0,
        "ffn_type": "swiglu",
        "dropout": 0.0,
        "invisible_init_std": 0.02,
        "target_frame_contract": "reference_camera_court_rzpi_v1",
        "track_query_rope_contract": "time_camera_reference_selector_v1",
        "reference_selector_mode": selector_mode,
        "ffn_mode": ffn_mode,
        "mhc_writeback": mhc_writeback,
        "mhc": {
            "coefficient_dim": 8,
            "sinkhorn_iters": 5,
            "eps": 1.0e-6,
            "residual_identity_bias": 4.0,
            "update_scale_init": 0.0,
        },
        "cswa": {
            "compression_ratio": 2,
            "window_radius": 1,
            "backend": "reference",
        },
    }


def _model(
    *,
    selector_mode: str = "reference",
    ffn_mode: FFNMode = "shared",
    mhc_writeback: MHCWriteback = "layer_end",
) -> PLCSTrackQueryReferenceAblationModel:
    model = PLCSTrackQueryReferenceAblationModel(
        PLCSModelConfig.from_mapping(
            _raw_config(
                selector_mode=selector_mode,
                ffn_mode=ffn_mode,
                mhc_writeback=mhc_writeback,
            )
        )
    )
    model.eval()
    return model


def _inputs() -> dict[str, Tensor]:
    prefix = (2, 3, 2)
    return {
        "human_kp": torch.rand(*prefix, 2, 17, 2),
        "human_vis": torch.zeros(*prefix, 2, 17, dtype=torch.bool),
        "court_kp": torch.rand(*prefix, 14, 2),
        "court_vis": torch.zeros(*prefix, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
        "reference_view_index": torch.tensor([0, 2], dtype=torch.int64),
    }


def test_public_forward_has_exact_required_sixth_tensor() -> None:
    assert list(
        inspect.signature(
            PLCSTrackQueryReferenceAblationModel.forward
        ).parameters
    ) == [
        "self",
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "reference_view_index",
    ]
    assert list(
        inspect.signature(PLCSTrackQueryAblationModel.forward).parameters
    ) == [
        "self",
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]


@pytest.mark.parametrize(
    ("writeback", "width"),
    [("after_object_temporal", 8), ("layer_end", 5)],
)
def test_reference_coordinates_preserve_full_and_compressed_token_order(
    writeback: MHCWriteback,
    width: int,
) -> None:
    coordinates = (
        PLCSTrackQueryReferenceAblationModel.build_spatial_coordinates(
            torch.tensor([1, 2], dtype=torch.int64),
            num_frames=2,
            num_views=3,
            num_detections=2,
            num_queries=2,
            mhc_writeback=writeback,
            selector_mode=ReferenceSelectorMode.REFERENCE,
        )
    )
    assert coordinates.shape == (4, width, 3)
    structured = coordinates.reshape(2, 2, width, 3)
    assert torch.equal(structured[:, 0, :, 0], torch.zeros(2, width))
    assert torch.equal(structured[:, 1, :, 0], torch.ones(2, width))
    object_width = 2 if writeback == "after_object_temporal" else 1
    expected_camera = torch.cat(
        (
            torch.zeros(2, dtype=torch.int64),
            torch.arange(1, 4).repeat_interleave(object_width),
        )
    )
    assert torch.equal(
        structured[:, :, :, 1], expected_camera.expand(2, 2, -1)
    )


@pytest.mark.parametrize("writeback", ["after_object_temporal", "layer_end"])
def test_selector_zero_changes_only_the_third_axis(
    writeback: MHCWriteback,
) -> None:
    reference = torch.tensor([1, 2], dtype=torch.int64)
    shared: _SpatialCoordinateArgs = {
        "num_frames": 2,
        "num_views": 3,
        "num_detections": 2,
        "num_queries": 2,
        "mhc_writeback": writeback,
    }
    selected = PLCSTrackQueryReferenceAblationModel.build_spatial_coordinates(
        reference,
        selector_mode=ReferenceSelectorMode.REFERENCE,
        **shared,
    )
    zero = PLCSTrackQueryReferenceAblationModel.build_spatial_coordinates(
        reference,
        selector_mode=ReferenceSelectorMode.SELECTOR_ZERO,
        **shared,
    )

    assert torch.equal(selected[..., :2], zero[..., :2])
    assert torch.count_nonzero(zero[..., 2]) == 0
    assert torch.count_nonzero(selected[..., 2]) > 0


@pytest.mark.parametrize(
    ("ffn_mode", "writeback", "selector_mode"),
    [
        ("per_attention", "after_object_temporal", "reference"),
        ("shared", "after_object_temporal", "reference"),
        ("per_attention", "layer_end", "reference"),
        ("shared", "layer_end", "reference"),
        ("shared", "layer_end", "selector_zero"),
    ],
)
def test_generic_ablation_forward_backward_is_finite(
    ffn_mode: FFNMode,
    writeback: MHCWriteback,
    selector_mode: str,
) -> None:
    model = _model(
        ffn_mode=ffn_mode,
        mhc_writeback=writeback,
        selector_mode=selector_mode,
    ).train()
    inputs = _inputs()
    inputs["human_kp"].requires_grad_()
    inputs["court_kp"].requires_grad_()
    output = cast("dict[str, Tensor]", model(**inputs))
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert inputs["human_kp"].grad is not None
    assert inputs["court_kp"].grad is not None


def test_selector_and_selector_zero_states_are_strictly_incompatible() -> None:
    selector = _model(selector_mode="reference")
    selector_zero = _model(selector_mode="selector_zero")

    with pytest.raises(RuntimeError):
        selector.load_state_dict(selector_zero.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        selector_zero.load_state_dict(selector.state_dict(), strict=True)
