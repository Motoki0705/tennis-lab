"""CPU smoke test for the fixed PLCS track-query experiment architecture."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from src.tasks.plcs.configuration import PLCSModelConfig
from src.tasks.plcs.models.plcs_track_query_ablation_model import (
    PLCSTrackQueryAblationModel,
)


def _config() -> PLCSModelConfig:
    return PLCSModelConfig.from_mapping(
        {
            "name": "plcs_track_query_ablation",
            "hidden_dim": 16,
            "num_heads": 4,
            "ffn_dim": 32,
            "num_queries": 4,
            "num_stages": 4,
            "num_joints": 17,
            "rope_dim": 4,
            "rope_theta": 10_000.0,
            "ffn_type": "swiglu",
            "dropout": 0.0,
            "role_rope_enabled": True,
            "invisible_init_std": 0.02,
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
    )


def test_cpu_forward_backward_has_finite_outputs_and_gradients() -> None:
    torch.manual_seed(777)
    model = PLCSTrackQueryAblationModel(_config()).train()
    human_kp = torch.rand(1, 2, 2, 4, 17, 2, requires_grad=True)
    court_kp = torch.rand(1, 2, 2, 14, 2, requires_grad=True)
    output = cast(
        "dict[str, Tensor]",
        model(
            human_kp,
            torch.ones(1, 2, 2, 4, 17, dtype=torch.bool),
            court_kp,
            torch.ones(1, 2, 2, 14, dtype=torch.bool),
            torch.tensor([[[False, True], [True, False]]]),
        ),
    )

    loss = (
        output["position"].square().mean()
        + output["rotation"][..., 0].square().mean()
        + output["presence_logits"].square().mean()
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in output.values())
    assert human_kp.grad is not None and torch.isfinite(human_kp.grad).all()
    assert court_kp.grad is not None and torch.isfinite(court_kp.grad).all()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(gradient.abs().any()) for gradient in gradients)
