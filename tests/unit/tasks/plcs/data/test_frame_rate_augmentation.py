"""Tests for synchronized PLCS train-time frame-rate sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tasks.plcs.data.frame_rate_augmentation import PLCSFrameRateSampler


def _config(*, choices: list[float], enabled: bool = True) -> dict[str, object]:
    return {
        "enabled": True,
        "frame_rate": {
            "enabled": enabled,
            "prob": 1.0,
            "choices_hz": choices,
        },
    }


def test_plan_downsamples_all_signals_at_identical_source_times() -> None:
    sampler = PLCSFrameRateSampler(_config(choices=[30.0]))
    plan = sampler.plan(
        source_fps=60.0,
        full_len=20,
        seq_len_range=(5, 5),
        crop_mode="center",
        augment=True,
        rng=np.random.default_rng(4),
    )
    source = torch.arange(plan.source_window.seq_len, dtype=torch.float32)
    visibility = torch.arange(plan.source_window.seq_len) % 2 == 0

    assert plan.output_fps == 30.0
    assert plan.source_positions == (0.0, 2.0, 4.0, 6.0, 8.0)
    torch.testing.assert_close(
        plan.linear(source, axis=0), torch.arange(0, 10, 2).float()
    )
    torch.testing.assert_close(plan.nearest(visibility, axis=0), visibility[::2])


def test_plan_upsamples_heading_on_shortest_wrapped_arc() -> None:
    sampler = PLCSFrameRateSampler(_config(choices=[60.0]))
    plan = sampler.plan(
        source_fps=30.0,
        full_len=4,
        seq_len_range=(3, 3),
        crop_mode="center",
        augment=True,
        rng=np.random.default_rng(3),
    )
    angles = torch.deg2rad(torch.tensor([170.0, -170.0], dtype=torch.float32))
    heading = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    output = plan.heading(heading, axis=0)

    assert plan.source_positions == (0.0, 0.5, 1.0)
    assert float(output[1, 0]) == pytest.approx(-1.0, abs=1e-6)
    assert float(output[1, 1]) == pytest.approx(0.0, abs=1e-6)


def test_masked_linear_never_blends_visible_coordinates_with_invalid_zero() -> None:
    sampler = PLCSFrameRateSampler(_config(choices=[60.0]))
    plan = sampler.plan(
        source_fps=30.0,
        full_len=2,
        seq_len_range=(3, 3),
        crop_mode="center",
        augment=True,
        rng=np.random.default_rng(2),
    )
    coordinates = torch.tensor([[[10.0, 20.0]], [[0.0, 0.0]]])
    visible = torch.tensor([[True], [False]])

    output, output_visible = plan.masked_linear(coordinates, visible, axis=0)

    torch.testing.assert_close(output[1], coordinates[0])
    torch.testing.assert_close(output[2], torch.zeros_like(output[2]))
    torch.testing.assert_close(
        output_visible,
        torch.tensor([[True], [True], [False]]),
    )


def test_masked_heading_uses_identity_outside_lifecycle() -> None:
    sampler = PLCSFrameRateSampler(_config(choices=[60.0]))
    plan = sampler.plan(
        source_fps=30.0,
        full_len=2,
        seq_len_range=(3, 3),
        crop_mode="center",
        augment=True,
        rng=np.random.default_rng(2),
    )
    heading = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    present = torch.tensor([True, False])

    output, output_present = plan.masked_heading(heading, present, axis=0)

    torch.testing.assert_close(output[1], heading[0])
    torch.testing.assert_close(output[2], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(output_present, torch.tensor([True, True, False]))


def test_infeasible_configured_rate_fails_explicitly() -> None:
    sampler = PLCSFrameRateSampler(_config(choices=[10.0]))
    with pytest.raises(ValueError, match="too short for every configured"):
        sampler.plan(
            source_fps=120.0,
            full_len=20,
            seq_len_range=(10, 10),
            crop_mode="random",
            augment=True,
            rng=np.random.default_rng(0),
        )
