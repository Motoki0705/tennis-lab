"""Shape tests for PLCS axial multiview models."""

from __future__ import annotations

import torch

from src.tasks.plcs.model_io.attention_masks import prepare_axial_attention_masks
from src.tasks.plcs.models.plcs_multiview_axial_model import PLCSMultiViewAxialModel


def test_multiview_axial_model_forward_accepts_single_view() -> None:
    torch.manual_seed(0)
    model = PLCSMultiViewAxialModel(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        rope_dim=4,
        rope_theta_time=10000.0,
        rope_theta_camera=10000.0,
        ffn_type="swiglu",
        predict_canonical_pose=False,
        max_views=1,
        max_seq_len=4,
        invisible_init_std=0.02,
        num_court_tokens=20,
    ).eval()

    padding_mask = torch.zeros(2, 1, 4, dtype=torch.bool)
    camera_mask, time_mask = prepare_axial_attention_masks(padding_mask)
    with torch.no_grad():
        out = model(
            human_kp=torch.randn(2, 1, 4, 17, 2),
            court_kp=torch.randn(2, 1, 4, 20, 2),
            human_vis=torch.ones(2, 1, 4, 17),
            padding_mask=padding_mask,
            court_vis=torch.ones(2, 1, 4, 20),
            camera_attention_mask=camera_mask,
            time_attention_mask=time_mask,
        )

    assert out["position"].shape == (2, 4, 3)
    assert out["rotation"].shape == (2, 4, 2)
