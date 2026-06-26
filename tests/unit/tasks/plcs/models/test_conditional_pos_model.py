"""Unit tests for PLCS conditional-position axial model variants."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.plcs.models import (
    PLCSMultiViewAxialConditionalPosModel,
    build_plcs_model,
)


@pytest.mark.parametrize(
    ("architecture", "source", "num_layers", "num_task_layers"),
    [
        ("reference_gated_cross_attn", "rotation", 0, 1),
        ("conditional_pos_decoder", "both", 1, 1),
        ("head_specific_conditional_bands", "rotation", 0, 1),
    ],
)
def test_conditional_pos_variants_forward_shapes(
    architecture: str,
    source: str,
    num_layers: int,
    num_task_layers: int,
) -> None:
    config = OmegaConf.create(
        {
            "model": {
                "name": "plcs_multiview_axial_conditional_pos",
                "hidden_dim": 32,
                "num_layers": num_layers,
                "num_task_layers": num_task_layers,
                "num_heads": 4,
                "rope_dim": 8,
                "dropout": 0.0,
                "predict_canonical_pose": True,
                "canonical_on_rotation_branch": True,
                "aux_position_on_rotation_branch": True,
                "max_views": 3,
                "max_seq_len": 8,
                "num_court_tokens": 20,
                "pos_context": {
                    "architecture": architecture,
                    "source": source,
                    "decoder_layers": 1,
                },
            },
            "data": {"max_seq_len": 8, "num_court_kp": 20},
        }
    )
    model = build_plcs_model(config)

    assert isinstance(model, PLCSMultiViewAxialConditionalPosModel)
    outputs = model(
        human_kp=torch.randn(2, 3, 4, 17, 2),
        court_kp=torch.randn(2, 3, 4, 20, 2),
        human_vis=torch.ones(2, 3, 4, 17),
        human_mask=torch.ones(2, 3, 4),
        court_vis=torch.ones(2, 3, 4, 20),
    )

    assert outputs["position"].shape == (2, 4, 3)
    assert outputs["rotation"].shape == (2, 4, 2)
    assert outputs["canonical_pose"].shape == (2, 4, 17, 3)
    assert outputs["aux_position"].shape == (2, 4, 3)
