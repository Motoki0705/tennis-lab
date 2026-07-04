"""Shape tests for BLCS axial multiview models."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.tasks.blcs.models import build_blcs_model


def test_multiview_axial_model_forward_accepts_single_view() -> None:
    torch.manual_seed(0)
    model = build_blcs_model(
        OmegaConf.create(
            {
                "model": {
                    "name": "blcs_multiview_axial",
                    "io": {"input_profile": "multiview"},
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "camera_layers_per_stage": [1],
                    "time_layers_per_stage": [1],
                    "time_global_stage_mask": [False],
                    "max_num_cameras": 1,
                    "max_seq_len": 4,
                    "dropout": 0.0,
                    "time_window_radius": 2,
                },
                "data": {"num_court_kp": 20},
            }
        )
    ).eval()

    with torch.no_grad():
        out = model(
            ball_uv=torch.randn(2, 1, 4, 2),
            court_kp=torch.randn(2, 1, 4, 20, 2),
            ball_vis=torch.ones(2, 1, 4),
            ball_mask=torch.ones(2, 1, 4),
            court_vis=torch.ones(2, 1, 4, 20),
        )

    assert out["position"].shape == (2, 4, 3)
