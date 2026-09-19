"""Physical velocity supervision, masking, confidence, and temporal contracts."""

from dataclasses import replace

import pytest
import torch

from src.tasks.slcs.training.losses import (
    SLCSLoss,
    SLCSLossConfig,
    SLCSLossInputs,
    make_ball_velocity_term,
)
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _inputs(frames: int = 3, fps: float = 30.0) -> SLCSLossInputs:
    time = torch.arange(frames, dtype=torch.float32)[None] / fps
    target = (
        time.unsqueeze(-1)
        * torch.tensor([30.0, 0.0, 0.0])
        / torch.tensor(COURT_COORD_SCALE_XYZ)
    )
    return SLCSLossInputs(
        pred_player_position=torch.zeros(1, 1, frames, 3),
        pred_player_rotation=torch.ones(1, 1, frames, 2),
        pred_ball_position=target.clone().requires_grad_(),
        pred_player_position_log_b=torch.zeros(1, 1, frames),
        pred_player_rotation_log_b=torch.zeros(1, 1, frames),
        pred_ball_position_log_b=torch.zeros(1, frames),
        target_player_position=torch.zeros(1, 1, frames, 3),
        target_player_rotation=torch.ones(1, 1, frames, 2),
        target_ball_position=target,
        player_mask=torch.ones(1, 1, frames, dtype=torch.bool),
        player_weight=torch.ones(1, 1, frames),
        ball_mask=torch.ones(1, frames, dtype=torch.bool),
        ball_weight=torch.ones(1, frames),
        padding_mask=torch.zeros(1, frames, dtype=torch.bool),
        frame_idx=torch.arange(frames)[None],
        timestamp=time,
    )


def test_correct_fast_motion_and_bounce_are_unpenalized() -> None:
    inputs = _inputs()
    assert make_ball_velocity_term(1.0)(inputs).item() == 0
    target = inputs.target_ball_position.clone()
    target[:, -1] = -target[:, -1]
    assert (
        make_ball_velocity_term(1.0)(
            replace(inputs, pred_ball_position=target, target_ball_position=target)
        ).item()
        == 0
    )


@pytest.mark.parametrize("fps", [30.0, 60.0])
def test_direction_physical_units_scale_and_gradient(fps: float) -> None:
    inputs = _inputs(fps=fps)
    pred = (-inputs.target_ball_position).requires_grad_()
    # Opposite 30 m/s velocity: 60 m/s residual, scale=30 => |x|=2.
    value = make_ball_velocity_term(30.0)(replace(inputs, pred_ball_position=pred))
    torch.testing.assert_close(value, torch.tensor(1.5 / 3))
    value.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_confidence_min_and_fractional_total_normalization() -> None:
    inputs = _inputs(fps=1.0)
    # x residual velocities 1 and 3 m/s give SmoothL1 XYZ means 1/6 and 5/6.
    pred = inputs.target_ball_position + torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]
    ) / torch.tensor(COURT_COORD_SCALE_XYZ)
    inputs = replace(
        inputs, pred_ball_position=pred, ball_weight=torch.tensor([[0.1, 0.4, 0.2]])
    )
    expected = ((1 / 6) * 0.1 + (5 / 6) * 0.2) / 0.3
    assert make_ball_velocity_term(1.0)(inputs).item() == pytest.approx(
        expected, abs=1e-6
    )


@pytest.mark.parametrize("exclusion", ["padding", "invalid", "gap"])
def test_excluded_pairs_do_not_read_nan_coordinates_or_time(exclusion: str) -> None:
    inputs = _inputs(4)
    pred = inputs.pred_ball_position.detach().clone()
    pred[:, -1] = float("nan")
    pred.requires_grad_()
    target = pred.detach().clone()
    assert inputs.timestamp is not None and inputs.frame_idx is not None
    inputs.timestamp[:, -1] = float("nan")
    if exclusion == "padding":
        inputs.padding_mask[:, -1] = True
    elif exclusion == "invalid":
        inputs.ball_mask[:, -1] = False
    else:
        inputs.frame_idx[:, -1] += 1
    value = make_ball_velocity_term(1.0)(
        replace(inputs, pred_ball_position=pred, target_ball_position=target)
    )
    assert value.item() == 0
    value.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("mode", ["short", "all_masked", "zero_weight"])
def test_empty_supervision_has_finite_connected_zero(mode: str) -> None:
    inputs = _inputs(1 if mode == "short" else 3)
    if mode == "all_masked":
        inputs.ball_mask.zero_()
    if mode == "zero_weight":
        inputs.ball_weight.zero_()
    value = make_ball_velocity_term(1.0)(inputs)
    assert value.item() == 0
    value.backward()
    assert inputs.pred_ball_position.grad is not None
    assert torch.isfinite(inputs.pred_ball_position.grad).all()


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_valid_time_difference_fails(dt: float) -> None:
    inputs = _inputs(2)
    with pytest.raises(ValueError, match="finite positive dt"):
        make_ball_velocity_term(1.0)(
            replace(inputs, timestamp=torch.tensor([[0.0, dt]]))
        )


@pytest.mark.parametrize("key", ["frame_idx", "timestamp"])
@pytest.mark.parametrize("problem", ["missing", "dtype", "shape"])
def test_time_metadata_contract(key: str, problem: str) -> None:
    inputs = _inputs()
    original = inputs.frame_idx if key == "frame_idx" else inputs.timestamp
    assert original is not None
    invalid = (
        None
        if problem == "missing"
        else original.double()
        if problem == "dtype"
        else original[:, :1]
    )
    with pytest.raises(ValueError, match="metadata|shape"):
        invalid_inputs = (
            replace(inputs, frame_idx=invalid)
            if key == "frame_idx"
            else replace(inputs, timestamp=invalid)
        )
        make_ball_velocity_term(1.0)(invalid_inputs)


def test_disabled_loss_preserves_all_legacy_values_without_metadata() -> None:
    config = SLCSLossConfig(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3)
    inputs = _inputs()
    baseline = SLCSLoss(config)(inputs)
    legacy = SLCSLoss(config)(replace(inputs, frame_idx=None, timestamp=None))
    assert baseline.keys() == legacy.keys() and "ball_velocity" not in legacy
    for key in baseline:
        torch.testing.assert_close(baseline[key], legacy[key], rtol=0, atol=0)
    enabled = SLCSLoss(replace(config, ball_velocity_weight=0.2))
    imperfect = replace(inputs, pred_ball_position=-inputs.pred_ball_position)
    original = SLCSLoss(config)(imperfect)
    supervised = enabled(imperfect)
    assert supervised["ball_velocity"] > 0
    torch.testing.assert_close(
        supervised["total"], original["total"] + 0.2 * supervised["ball_velocity"]
    )
    for key in original.keys() - {"total"}:
        torch.testing.assert_close(original[key], supervised[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="metadata"):
        enabled(replace(inputs, timestamp=None))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_fails_even_when_disabled(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        SLCSLossConfig(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            3,
            ball_velocity_scale_mps=scale,
        )
