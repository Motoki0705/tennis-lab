"""Small CPU forward/backward smoke tests for every BLCS ablation condition."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import Tensor

from src.tasks.blcs.configuration import (
    TrackQueryAblationModelConfig,
    parse_model_config,
)
from src.tasks.blcs.models.blcs_track_query_ablation_model import (
    BLCSTrackQueryAblationModel,
)
from src.utils.models.components.fixed_query_track_ablation_stage import (
    FFNMode,
    MHCWriteback,
)


def _config(
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
) -> TrackQueryAblationModelConfig:
    parsed = parse_model_config(
        {
            "model": {
                "name": "blcs_track_query_ablation",
                "hidden_dim": 16,
                "num_heads": 4,
                "num_stages": 4,
                "ffn_dim": 32,
                "num_queries": 4,
                "rope_dim": 4,
                "dropout": 0.0,
                "role_rope_enabled": True,
                "invisible_init_std": 0.02,
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
        }
    )
    assert isinstance(parsed, TrackQueryAblationModelConfig)
    return parsed


@pytest.mark.parametrize(
    ("ffn_mode", "mhc_writeback"),
    [
        ("per_attention", "after_object_temporal"),
        ("shared", "after_object_temporal"),
        ("per_attention", "layer_end"),
        ("shared", "layer_end"),
    ],
)
def test_cpu_forward_backward_has_finite_outputs_and_gradients(
    ffn_mode: FFNMode,
    mhc_writeback: MHCWriteback,
) -> None:
    torch.manual_seed(777)
    model = BLCSTrackQueryAblationModel(
        _config(ffn_mode, mhc_writeback)
    ).train()
    ball_uv = torch.rand(1, 2, 2, 4, 2, requires_grad=True)
    court_kp = torch.rand(1, 2, 2, 14, 2, requires_grad=True)
    output = cast(
        "dict[str, Tensor]",
        model(
            ball_uv,
            torch.ones(1, 2, 2, 4, dtype=torch.bool),
            court_kp,
            torch.ones(1, 2, 2, 14, dtype=torch.bool),
            torch.tensor([[[False, True], [True, False]]]),
        ),
    )

    loss = output["position"].square().mean() + output[
        "presence_logits"
    ].square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert ball_uv.grad is not None and torch.isfinite(ball_uv.grad).all()
    assert court_kp.grad is not None and torch.isfinite(court_kp.grad).all()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(gradient.abs().any()) for gradient in gradients)
