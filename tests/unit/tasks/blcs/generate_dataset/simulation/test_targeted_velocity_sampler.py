from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch

from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
    TargetedVelocitySampler,
)


def _config(*, max_ballistic_apex_height_m: float) -> TargetedVelocityConfig:
    return TargetedVelocityConfig(
        drive_elevation_range_deg=(5.0, 25.0),
        lob_elevation_range_deg=(35.0, 70.0),
        lob_probability=0.0,
        max_ballistic_apex_height_m=max_ballistic_apex_height_m,
        gravity=9.81,
        net_retry_max_attempts=12,
        net_check_max_frames=600,
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
