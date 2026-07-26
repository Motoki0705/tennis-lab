from __future__ import annotations

import math

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)


def test_high_lob_is_resolved_at_apex_limit_without_changing_target() -> None:
    config = TargetedVelocityConfig(max_ballistic_apex_height_m=8.0)
    sampler = TargetedVelocitySampler(config=config)
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
        TargetedVelocitySampler(
            config=TargetedVelocityConfig(max_ballistic_apex_height_m=0.0)
        )
