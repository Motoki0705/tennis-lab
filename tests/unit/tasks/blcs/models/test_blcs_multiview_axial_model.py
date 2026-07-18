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


def test_multiview_axial_model_masks_invisible_court_coordinates() -> None:
    torch.manual_seed(1)
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
                    "max_num_cameras": 2,
                    "max_seq_len": 3,
                    "dropout": 0.0,
                    "time_window_radius": 2,
                },
                "data": {"num_court_kp": 20},
            }
        )
    ).eval()
    inputs = {
        "ball_uv": torch.rand(1, 2, 3, 2),
        "court_kp": torch.rand(1, 2, 3, 20, 2),
        "ball_vis": torch.ones(1, 2, 3),
        "ball_mask": torch.ones(1, 2, 3),
        "court_vis": torch.ones(1, 2, 3, 20),
    }
    inputs["court_vis"][:, 1, :, 4] = 0
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["court_kp"][:, 1, :, 4] = torch.nan

    with torch.no_grad():
        output = model(**inputs)
        changed_output = model(**changed)

    torch.testing.assert_close(output["position"], changed_output["position"])
