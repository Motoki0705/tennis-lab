"""Recorded-clock correctness, event masking, scales and gradient reachability."""

from dataclasses import replace

import pytest
import torch

from src.tasks.blcs.model_io.contracts import BLCSTrajectoryPrediction
from src.tasks.blcs.physics.rollout import rollout_residuals
from src.tasks.blcs.training.losses import BLCSLoss, robust_mean
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    denormalize_court_velocity,
)
from tests.support.tasks.blcs_physics import (
    loss_config,
    physics_sample,
    prepared_samples,
)


@pytest.mark.parametrize(
    "fps,sim_fps,surface", [(30, 120, "hard"), (60, 240, "clay"), (25, 100, "grass")]
)
def test_saved_gt_replays_with_crop_and_collision_boundaries(fps, sim_fps, surface):
    _, target = prepared_samples(
        [
            physics_sample(
                start=3,
                output_fps=fps,
                sim_fps=sim_fps,
                surface=surface,
                hits=(20, 29, 40),
                kinds=("bounce", "net", "shot"),
            )
        ]
    )
    residual = rollout_residuals(
        denormalize_court_position(target.position),
        denormalize_court_velocity(target.velocity),
        target.loss_mask,
        target.physics,
        (1, 3),
    )
    assert len(residual.position)
    assert residual.position.abs().max() < 2e-6
    assert residual.velocity.abs().max() < 3e-6
    assert len(residual.position) < residual.possible
    assert torch.all(residual.simulation_steps == residual.horizon * (sim_fps // fps))


def test_nonzero_finite_physics_gradients_reach_both_heads_and_both_endpoints():
    _, target = prepared_samples([physics_sample()])
    p = target.position.clone()
    p[:, 4, 0] += 0.2
    p.requires_grad_()
    v = target.velocity.clone()
    v[:, 2, 0] += 0.3
    v.requires_grad_()
    cfg = loss_config(
        position_weight=0.0,
        reprojection_weight=0.0,
        velocity_weight=0.0,
        physics_start_step=0,
        physics_warmup_steps=0,
    )
    loss = BLCSLoss(cfg)(BLCSTrajectoryPrediction(p, v), target)["total"]
    assert loss > 0
    loss.backward()
    for value in (p, v):
        assert torch.isfinite(value.grad).all()
        assert value.grad.norm() > 0
        assert value.grad[:, 0].norm() > 0 or value.grad[:, -1].norm() > 0
    disabled = BLCSLoss(
        {**cfg, "rollout_position_weight": 0.0, "rollout_velocity_weight": 0.0}
    )(BLCSTrajectoryPrediction(p, v), target)["total"]
    assert disabled == 0


def test_missing_observations_remain_supervised_and_padding_is_excluded():
    sample = physics_sample()
    sample["ball_vis"].zero_()
    _, target = prepared_samples([sample, physics_sample(3, name="short")])
    assert target.loss_mask.sum() == 15
    assert target.physics.segment[1, 3:].eq(-1).all()
    p = target.position.clone()
    p[0, 3, 0] += 1
    p[1, 3:] = 999
    result = BLCSLoss(loss_config(reprojection_weight=0.0))(
        BLCSTrajectoryPrediction(p, target.velocity), target, step=1000
    )
    assert result["position"] > 0
    p[1, 3:] = -999
    repeat = BLCSLoss(loss_config(reprojection_weight=0.0))(
        BLCSTrajectoryPrediction(p, target.velocity), target, step=1000
    )
    assert result["total"] == repeat["total"]


def test_short_and_all_padding_have_differentiable_empty_losses():
    _, target = prepared_samples([physics_sample(1)])
    target = replace(
        target,
        loss_mask=torch.zeros_like(target.loss_mask),
        physics=replace(
            target.physics, segment=torch.full_like(target.physics.segment, -1)
        ),
    )
    p = target.position.clone().requires_grad_()
    v = target.velocity.clone().requires_grad_()
    losses = BLCSLoss(loss_config())(BLCSTrajectoryPrediction(p, v), target, step=1000)
    assert losses["total"] == 0
    losses["total"].backward()
    assert torch.isfinite(p.grad).all()


def test_physical_position_scale_reproduces_the_historical_control():
    error = torch.tensor([[2.0, 0.0, -1.0], [12.0, 3.0, 2.0]])
    expected = torch.nn.functional.smooth_l1_loss(
        error / 11.885, torch.zeros_like(error)
    )
    assert robust_mean(error, scale=11.885, beta=11.885) == expected


def test_physics_warmup_is_explicit():
    loss = BLCSLoss(loss_config())
    assert [loss.physics_factor(s) for s in (0, 200, 400, 600, 4000)] == [
        0,
        0,
        0.5,
        1,
        1,
    ]


@pytest.mark.parametrize("head", ["velocity", "physics"])
def test_active_terms_reject_missing_heads_in_standalone_loss(head):
    _, target = prepared_samples([physics_sample()])
    cfg = loss_config(parameters_weight=0.01 if head == "physics" else 0.0)
    prediction = BLCSTrajectoryPrediction(
        target.position, target.velocity if head == "physics" else None
    )
    with pytest.raises(ValueError, match="require"):
        BLCSLoss(cfg)(prediction, target, step=1000)
