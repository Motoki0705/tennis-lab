"""Public padding-mask contract tests for standard BLCS model families."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from src.tasks.blcs.models.blcs_model import BLCSModel
from src.tasks.blcs.models.blcs_multiview_axial_model import BLCSMultiViewAxialModel
from src.tasks.blcs.models.blcs_multiview_model import BLCSMultiViewModel


def _single_model() -> BLCSModel:
    return BLCSModel(
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        rope_dim=4,
        rope_theta=10_000.0,
        rope_theta_time=10_000.0,
        rope_theta_camera=1_000.0,
        rope_theta_type=1_000.0,
        ffn_type="swiglu",
        predict_velocity=False,
        max_seq_len=3,
        invisible_init_std=0.02,
        num_court_tokens=2,
    )


def _multiview_model() -> BLCSMultiViewModel:
    return BLCSMultiViewModel(
        hidden_dim=8,
        num_heads=2,
        ffn_dim=16,
        ffn_type="swiglu",
        dropout=0.0,
        rope_dim=4,
        rope_theta=10_000.0,
        rope_theta_time=10_000.0,
        rope_theta_camera=1_000.0,
        rope_theta_type=1_000.0,
        num_layers=1,
        predict_velocity=False,
        max_seq_len=3,
        max_num_cameras=2,
        num_court_tokens=2,
        invisible_init_std=0.02,
        query_init_std=0.02,
    )


def _axial_model() -> BLCSMultiViewAxialModel:
    return BLCSMultiViewAxialModel(
        hidden_dim=8,
        num_heads=2,
        attention_type="mha",
        num_kv_heads=None,
        ffn_dim=16,
        ffn_type="swiglu",
        dropout=0.0,
        rope_dim=4,
        rope_theta_time=10_000.0,
        rope_theta_camera=1_000.0,
        num_layers=1,
        predict_velocity=False,
        max_seq_len=3,
        max_num_cameras=2,
        invisible_init_std=0.02,
        num_court_tokens=2,
        camera_layers_per_stage=[1],
        time_layers_per_stage=[1],
    )


def _single_inputs() -> dict[str, Tensor]:
    return {
        "ball_uv": torch.rand(1, 3, 2),
        "ball_vis": torch.ones(1, 3, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 2),
        "court_vis": torch.ones(1, 2, dtype=torch.bool),
        "padding_mask": torch.tensor([[False, False, True]]),
    }


def _multiview_inputs() -> dict[str, Tensor]:
    return {
        "ball_uv": torch.rand(1, 2, 3, 2),
        "ball_vis": torch.ones(1, 2, 3, dtype=torch.bool),
        "court_kp": torch.rand(1, 2, 3, 2, 2),
        "court_vis": torch.ones(1, 2, 3, 2, dtype=torch.bool),
        "padding_mask": torch.tensor([[[False, False, True], [False, True, True]]]),
    }


@pytest.mark.parametrize(
    ("build_model", "build_inputs"),
    [
        (_single_model, _single_inputs),
        (_multiview_model, _multiview_inputs),
        (_axial_model, _multiview_inputs),
    ],
)
def test_standard_model_uses_exact_five_tensor_padding_contract(
    build_model: Callable[[], nn.Module],
    build_inputs: Callable[[], dict[str, Tensor]],
) -> None:
    model = build_model()
    assert list(inspect.signature(model.forward).parameters) == [
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "padding_mask",
    ]

    inputs = build_inputs()
    model.eval()
    with torch.no_grad():
        output: dict[str, Tensor] = model(**inputs)

    frame_valid = ~inputs["padding_mask"]
    if frame_valid.ndim == 3:
        frame_valid = frame_valid.any(dim=1)
    assert torch.isfinite(output["position"]).all()
    assert not output["position"][~frame_valid].any()


@pytest.mark.parametrize(
    ("build_model", "build_inputs"),
    [
        (_single_model, _single_inputs),
        (_multiview_model, _multiview_inputs),
        (_axial_model, _multiview_inputs),
    ],
)
def test_padding_values_cannot_change_valid_standard_outputs(
    build_model: Callable[[], nn.Module],
    build_inputs: Callable[[], dict[str, Tensor]],
) -> None:
    torch.manual_seed(4)
    model = build_model().eval()
    inputs = build_inputs()
    changed: dict[str, Any] = {name: value.clone() for name, value in inputs.items()}
    padding_mask = inputs["padding_mask"]
    changed["ball_uv"][padding_mask] = 0.99
    if padding_mask.ndim == 3:
        court_padding = padding_mask.unsqueeze(-1).expand_as(inputs["court_vis"])
        changed["court_kp"][court_padding] = 0.01

    with torch.no_grad():
        baseline: dict[str, Tensor] = model(**inputs)
        modified: dict[str, Tensor] = model(**changed)

    frame_valid = ~padding_mask
    if frame_valid.ndim == 3:
        frame_valid = frame_valid.any(dim=1)
    torch.testing.assert_close(
        baseline["position"][frame_valid],
        modified["position"][frame_valid],
    )
