"""Shape tests for PLCS axial multiview models."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.tasks.plcs.models import build_plcs_model


def test_multiview_axial_model_forward_accepts_single_view() -> None:
    torch.manual_seed(0)
    model = build_plcs_model(
        OmegaConf.create(
            {
                "model": {
                    "name": "plcs_multiview_axial",
                    "io": {"input_profile": "multiview"},
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "max_views": 1,
                    "max_seq_len": 4,
                    "dropout": 0.0,
                },
                "data": {"num_court_kp": 20},
            }
        )
    ).eval()

    with torch.no_grad():
        out = model(
            human_kp=torch.randn(2, 1, 4, 17, 2),
            court_kp=torch.randn(2, 1, 4, 20, 2),
            human_vis=torch.ones(2, 1, 4, 17),
            human_mask=torch.ones(2, 1, 4),
            court_vis=torch.ones(2, 1, 4, 20),
        )

    assert out["position"].shape == (2, 4, 3)
    assert out["rotation"].shape == (2, 4, 2)
