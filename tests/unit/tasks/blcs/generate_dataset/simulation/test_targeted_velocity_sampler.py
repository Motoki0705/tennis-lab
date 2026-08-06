from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.ball_physics import BallPhysics
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)


def test_net_height_uses_only_the_canonical_court_utility_path() -> None:
    assert not hasattr(BallPhysics, "_net_height_at_x")
    assert not hasattr(TargetedVelocitySampler, "_net_height_at_x")


def _config(*, max_ballistic_apex_height_m: float) -> TargetedVelocityConfig:
    return TargetedVelocityConfig(
        drive_elevation_range_deg=(5.0, 25.0),
        lob_elevation_range_deg=(35.0, 70.0),
        lob_probability=0.0,
        max_ballistic_apex_height_m=max_ballistic_apex_height_m,
        gravity=9.81,
        net_elevation_step_deg=2.0,
        landing_refine_enabled=True,
        landing_refine_max_iters=14,
        landing_refine_tolerance_m=0.25,
        landing_sim_max_frames=1200,
        target_margin_m=0.35,
    )


def _sampler(config: TargetedVelocityConfig) -> TargetedVelocitySampler:
    return TargetedVelocitySampler(
        cell_manager=Mock(),
        config=config,
        device="cpu",
    )


def test_high_lob_is_resolved_at_apex_limit_without_changing_target() -> None:
    config = _config(max_ballistic_apex_height_m=8.0)
    sampler = _sampler(config)
    start = torch.tensor([0.0, -12.0, 1.0])
    target = torch.tensor([2.0, 12.0, 0.0])

    velocity = sampler.compute_velocity_to_target(
        start,
        target,
        from_side="near",
        elevation_deg=70.0,
    )

    gravity = config.gravity
    apex = float(start[2]) + float(velocity[2]) ** 2 / (2.0 * gravity)
    flight_time = (
        float(velocity[2])
        + math.sqrt(float(velocity[2]) ** 2 + 2.0 * gravity * float(start[2]))
    ) / gravity
    landing_xy = start[:2] + velocity[:2] * flight_time
    assert apex == pytest.approx(config.max_ballistic_apex_height_m, abs=1e-5)
    torch.testing.assert_close(landing_xy, target[:2], atol=1e-5, rtol=1e-5)


def test_apex_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _sampler(_config(max_ballistic_apex_height_m=0.0))


def test_full_physics_refinement_fails_when_no_landing_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _config(max_ballistic_apex_height_m=8.0),
        landing_refine_max_iters=1,
    )
    sampler = _sampler(config)
    monkeypatch.setattr(
        sampler,
        "_simulate_landing",
        Mock(
            return_value=SimpleNamespace(
                hit_net=True,
                net_pos=torch.tensor([0.0, 0.0, 0.5]),
                bounce_pos=None,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="no gravity-only retry fallback"):
        sampler.compute_velocity_to_target(
            torch.tensor([0.0, -12.0, 1.0]),
            torch.tensor([0.0, 12.0, 0.0]),
            from_side="near",
            elevation_deg=20.0,
            physics=Mock(),
        )
