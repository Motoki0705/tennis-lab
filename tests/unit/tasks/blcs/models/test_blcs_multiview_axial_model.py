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


def test_multiview_axial_line_model_uses_two_tokens_per_camera() -> None:
    model = build_blcs_model(
        OmegaConf.create(
            {
                "model": {
                    "name": "blcs_multiview_axial",
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "camera_layers_per_stage": [1],
                    "time_layers_per_stage": [1],
                    "time_global_stage_mask": [False],
                    "max_num_cameras": 2,
                    "max_seq_len": 4,
                    "dropout": 0.0,
                    "time_window_radius": 2,
                    "court_input_type": "line",
                    "max_court_lines": 6,
                }
            }
        )
    )
    captured: list[int] = []
    handle = model.camera_layers[0][0].register_forward_pre_hook(
        lambda _module, args: captured.append(int(args[0].shape[1]))
    )
    out = model(
        ball_uv=torch.randn(2, 2, 4, 2),
        court_lines=torch.zeros(2, 2, 4, 6, 4),
        ball_vis=torch.ones(2, 2, 4),
        ball_mask=torch.ones(2, 2, 4),
    )
    out["position"].sum().backward()
    handle.remove()

    assert captured == [4]
    assert out["position"].shape == (2, 4, 3)
    assert torch.isfinite(out["position"]).all()


def test_multiview_axial_line_model_output_depends_on_court_lines() -> None:
    torch.manual_seed(4)
    model = build_blcs_model(
        OmegaConf.create(
            {
                "model": {
                    "name": "blcs_multiview_axial",
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
                    "court_input_type": "line",
                    "max_court_lines": 6,
                }
            }
        )
    ).eval()
    ball_uv = torch.randn(1, 1, 4, 2)
    common = {
        "ball_uv": ball_uv,
        "ball_vis": torch.ones(1, 1, 4),
        "ball_mask": torch.ones(1, 1, 4),
    }

    with torch.no_grad():
        no_line = model(court_lines=torch.zeros(1, 1, 4, 6, 4), **common)
        with_lines = model(court_lines=torch.rand(1, 1, 4, 6, 4), **common)

    assert not torch.allclose(no_line["position"], with_lines["position"])


def test_kp_default_state_dict_remains_strictly_compatible() -> None:
    config = OmegaConf.create(
        {
            "model": {
                "name": "blcs_multiview_axial",
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "camera_layers_per_stage": [1],
                "time_layers_per_stage": [1],
                "time_global_stage_mask": [False],
                "max_num_cameras": 1,
                "max_seq_len": 4,
            },
            "data": {"num_court_kp": 20},
        }
    )
    original = build_blcs_model(config)
    restored = build_blcs_model(config)
    restored.load_state_dict(original.state_dict(), strict=True)
