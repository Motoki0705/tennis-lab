"""BLCS reference-v2 generic ablation model contracts."""

from __future__ import annotations

import inspect
from typing import cast

import pytest
import torch
from torch import Tensor

from src.tasks.base.models import ReferenceSelectorMode
from src.tasks.blcs.configuration import (
    TrackQueryReferenceAblationModelConfig,
    parse_model_config,
)
from src.tasks.blcs.models import BLCSTrackQueryReferenceAblationModel
from src.utils.configuration import SemanticConfigurationError


def _config(
    *,
    selector_mode: str = "reference",
    mhc_writeback: str = "layer_end",
    rope_dim: int = 6,
) -> TrackQueryReferenceAblationModelConfig:
    parsed = parse_model_config(
        {
            "model": {
                "name": "blcs_track_query_reference_ablation",
                "hidden_dim": 24,
                "num_heads": 4,
                "num_stages": 4,
                "ffn_dim": 48,
                "num_queries": 2,
                "rope_dim": rope_dim,
                "dropout": 0.0,
                "invisible_init_std": 0.02,
                "target_frame_contract": "reference_camera_court_rzpi_v1",
                "track_query_rope_contract": "time_camera_reference_selector_v1",
                "reference_selector_mode": selector_mode,
                "ffn_mode": "shared",
                "mhc_writeback": mhc_writeback,
                "query_ffn_after_spatial": False,
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
        }
    )
    assert isinstance(parsed, TrackQueryReferenceAblationModelConfig)
    return parsed


def _inputs() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.rand(2, 3, 2, 2, 2, requires_grad=True),
        torch.zeros(2, 3, 2, 2, dtype=torch.bool),
        torch.rand(2, 3, 2, 14, 2, requires_grad=True),
        torch.zeros(2, 3, 2, 14, dtype=torch.bool),
        torch.zeros(2, 3, 2, dtype=torch.bool),
        torch.tensor([1, 2], dtype=torch.int64),
    )


def test_ablation_forward_has_exactly_six_required_tensors() -> None:
    signature = inspect.signature(BLCSTrackQueryReferenceAblationModel.forward)
    assert list(signature.parameters)[-1] == "reference_view_index"
    assert len(signature.parameters) == 7
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in list(signature.parameters.values())[1:]
    )


@pytest.mark.parametrize(
    ("writeback", "width"),
    [("after_object_temporal", 8), ("layer_end", 5)],
)
def test_full_and_compressed_positions_preserve_width_and_per_sample_selector(
    writeback: str,
    width: int,
) -> None:
    model = BLCSTrackQueryReferenceAblationModel(
        _config(mhc_writeback=writeback)
    )
    coordinates = model.build_spatial_coordinates(
        torch.tensor([1, 2], dtype=torch.int64),
        num_frames=2,
        num_views=3,
        num_detections=2,
        num_queries=2,
        mhc_writeback=model.mhc_writeback,
        selector_mode=ReferenceSelectorMode.REFERENCE,
    )
    assert coordinates.shape == (4, width, 3)
    object_width = 2 if writeback == "after_object_temporal" else 1
    assert coordinates[0, 2:, 2].tolist() == (
        [1] * object_width + [0] * object_width + [1] * object_width
    )
    assert coordinates[2, 2:, 2].tolist() == (
        [1] * object_width + [1] * object_width + [0] * object_width
    )


def test_selector_zero_keeps_sixth_input_and_zeroes_only_third_axis() -> None:
    reference = BLCSTrackQueryReferenceAblationModel(_config(selector_mode="reference"))
    zero = BLCSTrackQueryReferenceAblationModel(
        _config(selector_mode="selector_zero")
    )
    index = torch.tensor([1, 2], dtype=torch.int64)
    reference_coordinates = reference.build_spatial_coordinates(
        index,
        num_frames=2,
        num_views=3,
        num_detections=2,
        num_queries=2,
        mhc_writeback="layer_end",
        selector_mode=reference.reference_selector_mode,
    )
    zero_coordinates = zero.build_spatial_coordinates(
        index,
        num_frames=2,
        num_views=3,
        num_detections=2,
        num_queries=2,
        mhc_writeback="layer_end",
        selector_mode=zero.reference_selector_mode,
    )
    torch.testing.assert_close(reference_coordinates[..., :2], zero_coordinates[..., :2])
    assert not zero_coordinates[..., 2].any()
    with pytest.raises(RuntimeError):
        reference.load_state_dict(zero.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        zero.load_state_dict(reference.state_dict(), strict=True)


@pytest.mark.parametrize("selector_mode", ["reference", "selector_zero"])
def test_cpu_forward_backward_is_finite(selector_mode: str) -> None:
    model = BLCSTrackQueryReferenceAblationModel(
        _config(selector_mode=selector_mode)
    ).train()
    inputs = _inputs()
    output = cast("dict[str, Tensor]", model(*inputs))
    loss = output["position"].square().mean() + output["presence_logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert inputs[0].grad is not None and torch.isfinite(inputs[0].grad).all()
    assert inputs[2].grad is not None and torch.isfinite(inputs[2].grad).all()


def test_ablation_rejects_dim4_and_role_flag_in_v2_schema() -> None:
    with pytest.raises(SemanticConfigurationError, match="at least 6"):
        _config(rope_dim=4)
    assert not hasattr(_config(), "role_rope_enabled")
