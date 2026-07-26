from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.tasks.blcs.models.components import CourtBallPointFusion


def _fusion() -> CourtBallPointFusion:
    return CourtBallPointFusion(
        output_dim=64,
        num_court_points=14,
        config=OmegaConf.create(
            {
                "token_dim": 32,
                "num_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "rope_dim": 8,
                "dropout": 0.0,
            }
        ),
        invisible_init_std=0.02,
    )


def test_rope_coordinates_use_independent_court_and_ball_axes() -> None:
    coordinates = CourtBallPointFusion.build_rope_coordinates(
        num_court_points=3,
        num_ball_points=2,
        device=torch.device("cpu"),
    )

    assert torch.equal(
        coordinates,
        torch.tensor([[0, 0], [1, 0], [2, 0], [0, 0], [0, 1]]),
    )


def test_fusion_uses_32_dim_tokens_and_projects_only_ball_outputs() -> None:
    fusion = _fusion()
    output = fusion(
        court_kp=torch.rand(2, 3, 14, 2),
        court_visible=torch.ones(2, 3, 14, dtype=torch.bool),
        ball_uv=torch.rand(2, 3, 5, 2),
        ball_visible=torch.ones(2, 3, 5, dtype=torch.bool),
        context_valid=torch.ones(2, 3, dtype=torch.bool),
        mask_invisible_ball=True,
    )

    assert fusion.token_dim == 32
    assert fusion.coordinate_projection.layers[3].out_features == 32
    assert fusion.output_projection.in_features == 32
    assert fusion.output_projection.out_features == 64
    assert output.shape == (2, 3, 5, 64)


def test_ball_output_receives_gradient_from_visible_court_tokens() -> None:
    torch.manual_seed(4)
    fusion = _fusion()
    court_kp = torch.rand(1, 14, 2, requires_grad=True)
    output = fusion(
        court_kp=court_kp,
        court_visible=torch.ones(1, 14, dtype=torch.bool),
        ball_uv=torch.rand(1, 2, 2),
        ball_visible=torch.ones(1, 2, dtype=torch.bool),
        context_valid=torch.ones(1, dtype=torch.bool),
        mask_invisible_ball=True,
    )

    output.square().sum().backward()

    assert court_kp.grad is not None
    assert bool(court_kp.grad.abs().sum() > 0)


def test_masked_coordinates_cannot_change_outputs_and_invalid_ball_is_zero() -> None:
    torch.manual_seed(5)
    fusion = _fusion().eval()
    inputs = {
        "court_kp": torch.rand(1, 14, 2),
        "court_visible": torch.ones(1, 14, dtype=torch.bool),
        "ball_uv": torch.rand(1, 2, 2),
        "ball_visible": torch.tensor([[True, False]]),
        "context_valid": torch.ones(1, dtype=torch.bool),
        "mask_invisible_ball": True,
    }
    inputs["court_visible"][0, 3] = False
    changed = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    changed["court_kp"][0, 3] = torch.nan
    changed["ball_uv"][0, 1] = torch.nan

    with torch.no_grad():
        output = fusion(**inputs)
        changed_output = fusion(**changed)

    torch.testing.assert_close(output, changed_output)
    torch.testing.assert_close(output[0, 1], torch.zeros_like(output[0, 1]))


def test_unmasked_invisible_ball_remains_learnable_context_memory() -> None:
    torch.manual_seed(6)
    fusion = _fusion()
    output = fusion(
        court_kp=torch.rand(1, 14, 2),
        court_visible=torch.ones(1, 14, dtype=torch.bool),
        ball_uv=torch.rand(1, 2, 2),
        ball_visible=torch.tensor([[True, False]]),
        context_valid=torch.ones(1, dtype=torch.bool),
        mask_invisible_ball=False,
    )

    output.square().sum().backward()

    gradient = fusion.invisible_ball_token.token.grad
    assert gradient is not None
    assert bool(gradient.abs().sum() > 0)
