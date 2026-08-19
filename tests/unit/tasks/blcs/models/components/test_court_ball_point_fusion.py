from __future__ import annotations

from typing import TypedDict, cast

import torch

from src.tasks.blcs.configuration import PointFusionConfig
from src.tasks.blcs.model_io.attention_masks import prepare_point_attention_mask
from src.tasks.blcs.models.components.court_ball_point_fusion import (
    CourtBallPointFusion,
)


def _fusion() -> CourtBallPointFusion:
    return CourtBallPointFusion(
        output_dim=64,
        num_court_points=14,
        config=PointFusionConfig(
            token_dim=32,
            num_heads=4,
            num_layers=1,
            ffn_dim=64,
            rope_dim=8,
            dropout=0.0,
        ),
        invisible_init_std=0.02,
    )


def _forward(
    fusion: CourtBallPointFusion,
    *,
    court_kp: torch.Tensor,
    court_visible: torch.Tensor,
    ball_uv: torch.Tensor,
    ball_visible: torch.Tensor,
    context_valid: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    mask_invisible_ball: bool = True,
) -> torch.Tensor:
    ball_state_valid, attention_mask = prepare_point_attention_mask(
        ball_visible=ball_visible,
        candidate_mask=(
            torch.ones_like(ball_visible)
            if candidate_mask is None
            else candidate_mask
        ),
        court_visible=court_visible,
        context_valid=context_valid,
        mask_invisible_observations=mask_invisible_ball,
    )
    return cast(
        torch.Tensor,
        fusion(
            court_kp=court_kp,
            court_visible=court_visible,
            ball_uv=ball_uv,
            ball_visible=ball_visible,
            ball_state_valid=ball_state_valid,
            attention_mask=attention_mask,
        ),
    )


class _FusionInputs(TypedDict):
    court_kp: torch.Tensor
    court_visible: torch.Tensor
    ball_uv: torch.Tensor
    ball_visible: torch.Tensor
    context_valid: torch.Tensor


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
    output = _forward(
        fusion,
        court_kp=torch.rand(2, 3, 14, 2),
        court_visible=torch.ones(2, 3, 14, dtype=torch.bool),
        ball_uv=torch.rand(2, 3, 5, 2),
        ball_visible=torch.ones(2, 3, 5, dtype=torch.bool),
        context_valid=torch.ones(2, 3, dtype=torch.bool),
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
    output = _forward(
        fusion,
        court_kp=court_kp,
        court_visible=torch.ones(1, 14, dtype=torch.bool),
        ball_uv=torch.rand(1, 2, 2),
        ball_visible=torch.ones(1, 2, dtype=torch.bool),
        context_valid=torch.ones(1, dtype=torch.bool),
    )

    output.square().sum().backward()

    assert court_kp.grad is not None
    assert bool(court_kp.grad.abs().sum() > 0)


def test_masked_coordinates_cannot_change_outputs_and_invalid_ball_is_zero() -> None:
    torch.manual_seed(5)
    fusion = _fusion().eval()
    inputs: _FusionInputs = {
        "court_kp": torch.rand(1, 14, 2),
        "court_visible": torch.ones(1, 14, dtype=torch.bool),
        "ball_uv": torch.rand(1, 2, 2),
        "ball_visible": torch.tensor([[True, False]]),
        "context_valid": torch.ones(1, dtype=torch.bool),
    }
    inputs["court_visible"][0, 3] = False
    changed: _FusionInputs = {
        "court_kp": inputs["court_kp"].clone(),
        "court_visible": inputs["court_visible"].clone(),
        "ball_uv": inputs["ball_uv"].clone(),
        "ball_visible": inputs["ball_visible"].clone(),
        "context_valid": inputs["context_valid"].clone(),
    }
    changed["court_kp"][0, 3] = torch.nan
    changed["ball_uv"][0, 1] = torch.nan

    with torch.no_grad():
        output = _forward(fusion, **inputs)
        changed_output = _forward(fusion, **changed)

    torch.testing.assert_close(output, changed_output)
    torch.testing.assert_close(output[0, 1], torch.zeros_like(output[0, 1]))


def test_unmasked_invisible_ball_remains_learnable_context_memory() -> None:
    torch.manual_seed(6)
    fusion = _fusion()
    output = _forward(
        fusion,
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


def test_candidate_mask_excludes_padding_when_invisible_memory_is_enabled() -> None:
    torch.manual_seed(8)
    fusion = _fusion().eval()
    ball_uv = torch.rand(1, 2, 2)
    changed_uv = ball_uv.clone()
    changed_uv[0, 1] = torch.nan
    kwargs = {
        "court_kp": torch.rand(1, 14, 2),
        "court_visible": torch.ones(1, 14, dtype=torch.bool),
        "ball_visible": torch.tensor([[True, False]]),
        "candidate_mask": torch.tensor([[True, False]]),
        "context_valid": torch.ones(1, dtype=torch.bool),
        "mask_invisible_ball": False,
    }

    with torch.no_grad():
        output = _forward(fusion, ball_uv=ball_uv, **kwargs)
        changed = _forward(fusion, ball_uv=changed_uv, **kwargs)

    torch.testing.assert_close(output, changed)
    assert not output[0, 1].any()
