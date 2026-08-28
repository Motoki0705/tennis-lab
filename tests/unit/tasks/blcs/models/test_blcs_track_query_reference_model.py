"""BLCS reference-v2 normal model contracts."""

from __future__ import annotations

import inspect
from typing import cast

import pytest
import torch
from torch import Tensor

from src.tasks.blcs.configuration import (
    TrackQueryModelConfig,
    TrackQueryReferenceModelConfig,
    parse_model_config,
)
from src.tasks.blcs.models import (
    BLCSTrackQueryModel,
    BLCSTrackQueryReferenceModel,
)
from src.utils.configuration import SemanticConfigurationError


def _raw_model(*, rope_dim: int = 6) -> dict[str, object]:
    return {
        "name": "blcs_track_query_reference",
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
        "reference_selector_mode": "reference",
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


def _config(*, rope_dim: int = 6) -> TrackQueryReferenceModelConfig:
    parsed = parse_model_config({"model": _raw_model(rope_dim=rope_dim)})
    assert isinstance(parsed, TrackQueryReferenceModelConfig)
    return parsed


def _legacy_config() -> TrackQueryModelConfig:
    raw = _raw_model()
    raw["name"] = "blcs_track_query"
    raw["role_rope_enabled"] = True
    del raw["target_frame_contract"]
    del raw["track_query_rope_contract"]
    del raw["reference_selector_mode"]
    parsed = parse_model_config({"model": raw})
    assert isinstance(parsed, TrackQueryModelConfig)
    return parsed


def _inputs() -> dict[str, Tensor]:
    return {
        "ball_uv": torch.rand(2, 3, 2, 2, 2),
        "ball_vis": torch.zeros(2, 3, 2, 2, dtype=torch.bool),
        "court_kp": torch.rand(2, 3, 2, 14, 2),
        "court_vis": torch.zeros(2, 3, 2, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(2, 3, 2, dtype=torch.bool),
        "reference_view_index": torch.tensor([1, 2], dtype=torch.int64),
    }


def test_forward_public_contract_has_exactly_six_required_tensors() -> None:
    signature = inspect.signature(BLCSTrackQueryReferenceModel.forward)
    assert list(signature.parameters) == [
        "self",
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "reference_view_index",
    ]
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in list(signature.parameters.values())[1:]
    )


def test_coordinates_and_frequencies_expand_distinct_references_over_time() -> None:
    model = BLCSTrackQueryReferenceModel(_config()).eval()
    inputs = _inputs()
    expected = model.build_spatial_coordinates(
        inputs["reference_view_index"],
        num_frames=2,
        num_views=3,
        num_detections=2,
        num_queries=2,
    )
    assert expected.shape == (4, 8, 3)
    assert torch.equal(expected[:, :2, 2], torch.zeros(4, 2, dtype=torch.int64))
    assert expected[0, 2:, 2].tolist() == [1, 1, 0, 0, 1, 1]
    assert expected[1, 2:, 2].tolist() == [1, 1, 0, 0, 1, 1]
    assert expected[2, 2:, 2].tolist() == [1, 1, 1, 1, 0, 0]
    assert expected[3, 2:, 2].tolist() == [1, 1, 1, 1, 0, 0]
    captured: dict[str, Tensor] = {}

    def capture(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        value = kwargs["spatial_freqs"]
        assert isinstance(value, Tensor)
        captured["spatial_freqs"] = value

    hook = model.stages[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        hook.remove()
    torch.testing.assert_close(
        captured["spatial_freqs"],
        model.spatial_frequency_computer(expected),
    )


def test_forward_backward_keeps_reference_context_when_observations_are_empty() -> None:
    model = BLCSTrackQueryReferenceModel(_config()).train()
    inputs = _inputs()
    inputs["padding_mask"][0, 0] = True
    inputs["padding_mask"][1, 0] = True
    inputs["padding_mask"][1, 1] = True
    output = cast("dict[str, Tensor]", model(**inputs))
    loss = output["position"].square().mean() + output["presence_logits"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().any())
        for parameter in model.parameters()
    )


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        (torch.tensor([1, 2], dtype=torch.int32), ValueError),
        (torch.tensor([[1], [2]], dtype=torch.int64), ValueError),
        (torch.tensor([1, 3], dtype=torch.int64), ValueError),
    ],
)
def test_forward_rejects_nonexact_reference_index(
    replacement: Tensor,
    error: type[Exception],
) -> None:
    inputs = _inputs()
    inputs["reference_view_index"] = replacement
    with pytest.raises(error):
        BLCSTrackQueryReferenceModel(_config())(**inputs)


def test_forward_rejects_masked_reference_at_a_supervised_time() -> None:
    inputs = _inputs()
    inputs["padding_mask"][0, 1, 0] = True
    with pytest.raises(ValueError, match="unmasked reference context"):
        BLCSTrackQueryReferenceModel(_config())(**inputs)


def test_v2_rejects_dim4_and_is_state_incompatible_with_same_shape_v1() -> None:
    with pytest.raises(SemanticConfigurationError, match="at least 6"):
        _config(rope_dim=4)
    reference = BLCSTrackQueryReferenceModel(_config())
    legacy = BLCSTrackQueryModel(_legacy_config())
    with pytest.raises(RuntimeError):
        reference.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(reference.state_dict(), strict=True)
