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


def test_multiview_axial_line_model_uses_two_tokens_per_camera() -> None:
    model = build_plcs_model(
        OmegaConf.create(
            {
                "model": {
                    "name": "plcs_multiview_axial",
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "max_views": 2,
                    "max_seq_len": 4,
                    "dropout": 0.0,
                    "court_input_type": "line",
                    "max_court_lines": 6,
                }
            }
        )
    )
    captured: list[int] = []
    handle = model.camera_layers[0].register_forward_pre_hook(
        lambda _module, args: captured.append(int(args[0].shape[1]))
    )
    out = model(
        human_kp=torch.randn(2, 2, 4, 17, 2),
        court_lines=torch.zeros(2, 2, 4, 6, 4),
        human_vis=torch.ones(2, 2, 4, 17),
        human_mask=torch.ones(2, 2, 4),
    )
    (out["position"].sum() + out["rotation"].sum()).backward()
    handle.remove()

    assert captured == [4]
    assert out["position"].shape == (2, 4, 3)
    assert out["rotation"].shape == (2, 4, 2)
    assert torch.isfinite(out["position"]).all()


def test_kp_default_state_dict_remains_strictly_compatible() -> None:
    config = OmegaConf.create(
        {
            "model": {
                "name": "plcs_multiview_axial",
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "max_views": 1,
                "max_seq_len": 4,
            },
            "data": {"num_court_kp": 20},
        }
    )
    original = build_plcs_model(config)
    restored = build_plcs_model(config)
    restored.load_state_dict(original.state_dict(), strict=True)
