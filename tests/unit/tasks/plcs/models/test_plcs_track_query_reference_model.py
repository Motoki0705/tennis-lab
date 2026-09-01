"""PLCS reference-conditioned normal track-query contract tests."""

from __future__ import annotations

import inspect
from typing import cast

import pytest
import torch
from torch import Tensor

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.models.plcs_track_query_model import PLCSTrackQueryModel
from src.tasks.plcs.models.plcs_track_query_reference_model import (
    PLCSTrackQueryReferenceModel,
)


def _raw_config(*, rope_dim: int = 6) -> dict[str, object]:
    return {
        "name": "plcs_track_query_reference",
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


def _model() -> PLCSTrackQueryReferenceModel:
    model = PLCSTrackQueryReferenceModel(PLCSModelConfig.from_mapping(_raw_config()))
    model.eval()
    return model


def _inputs() -> dict[str, Tensor]:
    prefix = (2, 3, 2)
    return {
        "human_kp": torch.rand(*prefix, 2, 17, 2),
        "human_vis": torch.ones(*prefix, 2, 17, dtype=torch.bool),
        "court_kp": torch.rand(*prefix, 14, 2),
        "court_vis": torch.ones(*prefix, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(*prefix, dtype=torch.bool),
        "reference_view_index": torch.tensor([1, 2], dtype=torch.int64),
    }


def test_public_forward_has_exact_required_sixth_tensor() -> None:
    assert list(inspect.signature(PLCSTrackQueryReferenceModel.forward).parameters) == [
        "self",
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
        "reference_view_index",
    ]
    assert list(inspect.signature(PLCSTrackQueryModel.forward).parameters) == [
        "self",
        "human_kp",
        "human_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]


def test_coordinates_expand_distinct_batch_references_across_all_time() -> None:
    reference = torch.tensor([1, 2], dtype=torch.int64)
    coordinates = PLCSTrackQueryReferenceModel.build_spatial_coordinates(
        reference,
        num_frames=2,
        num_views=3,
        num_detections=2,
        num_queries=2,
    ).reshape(2, 2, 5, 3)

    expected_camera = torch.tensor([0, 0, 1, 2, 3])
    assert torch.equal(coordinates[:, :, :, 1], expected_camera.expand(2, 2, -1))
    assert torch.equal(coordinates[:, 0, :, 0], torch.zeros(2, 5, dtype=torch.int64))
    assert torch.equal(coordinates[:, 1, :, 0], torch.ones(2, 5, dtype=torch.int64))
    assert torch.equal(
        coordinates[0, :, :, 2],
        torch.tensor([0, 0, 1, 0, 1]).expand(2, -1),
    )
    assert torch.equal(
        coordinates[1, :, :, 2],
        torch.tensor([0, 0, 1, 1, 0]).expand(2, -1),
    )


def test_forward_uses_exact_shared_coordinates_and_frequencies() -> None:
    model = _model()
    inputs = _inputs()
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
            output = cast("dict[str, Tensor]", model(**inputs))
    finally:
        hook.remove()

    coordinates = model.build_spatial_coordinates(
        inputs["reference_view_index"],
        num_frames=2,
        num_views=3,
        num_detections=2,
        num_queries=2,
    )
    torch.testing.assert_close(
        captured["spatial_freqs"],
        model.spatial_frequency_computer(coordinates),
    )
    assert output["position"].shape == (2, 2, 2, 3)
    assert output["rotation"].shape == (2, 2, 2, 2)
    assert output["presence_logits"].shape == (2, 2, 2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda inputs: inputs.__setitem__(
                "reference_view_index",
                inputs["reference_view_index"].to(torch.int32),
            ),
            "torch.int64",
        ),
        (
            lambda inputs: inputs.__setitem__(
                "reference_view_index", torch.tensor([0, 3])
            ),
            "\[0, 3\)",
        ),
        (
            lambda inputs: inputs["padding_mask"].__setitem__((0, 1, 0), True),
            "unmasked reference context",
        ),
    ],
)
def test_forward_rejects_invalid_or_padded_reference(
    mutation: object,
    message: str,
) -> None:
    inputs = _inputs()
    assert callable(mutation)
    mutation(inputs)
    with pytest.raises((TypeError, ValueError), match=message):
        _model()(**inputs)


def test_empty_visibility_keeps_reference_context_and_backward_is_finite() -> None:
    model = _model().train()
    inputs = _inputs()
    inputs["human_vis"].zero_()
    inputs["court_vis"].zero_()
    output = cast("dict[str, Tensor]", model(**inputs))
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_v1_and_v2_state_dicts_are_strictly_incompatible() -> None:
    raw_v1 = _raw_config()
    raw_v1["name"] = "plcs_track_query"
    del raw_v1["target_frame_contract"]
    del raw_v1["track_query_rope_contract"]
    del raw_v1["reference_selector_mode"]
    v1 = PLCSTrackQueryModel(PLCSModelConfig.from_mapping(raw_v1))
    v2 = _model()

    with pytest.raises(RuntimeError):
        v2.load_state_dict(v1.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        v1.load_state_dict(v2.state_dict(), strict=True)
