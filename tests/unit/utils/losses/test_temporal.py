"""Unit tests for :mod:`src.utils.losses.temporal`."""

from __future__ import annotations

import torch

from src.utils.losses.temporal import (
    ballistic_gravity_penalty,
    ballistic_second_difference,
    finite_difference,
    smoothness_penalty,
)


class TestFiniteDifference:
    def test_first_difference_of_ramp_is_constant(self) -> None:
        x = torch.arange(6.0).view(1, 6, 1) * 3.0  # slope 3
        d = finite_difference(x, 1)
        assert d.shape == (1, 5, 1)
        torch.testing.assert_close(d, torch.full((1, 5, 1), 3.0))

    def test_second_difference_of_ramp_is_zero(self) -> None:
        x = torch.arange(6.0).view(1, 6, 1) * 3.0
        torch.testing.assert_close(finite_difference(x, 2), torch.zeros(1, 4, 1))

    def test_second_difference_of_parabola_is_two_a(self) -> None:
        t = torch.arange(7.0)
        a = 0.5
        x = (a * t * t).view(1, 7, 1)
        d = finite_difference(x, 2)
        torch.testing.assert_close(d, torch.full((1, 5, 1), 2 * a))

    def test_multichannel_shapes(self) -> None:
        x = torch.randn(2, 10, 3)
        assert finite_difference(x, 3).shape == (2, 7, 3)


class TestSmoothnessPenalty:
    def test_constant_velocity_has_zero_jerk_loss(self) -> None:
        # Linear motion: 3rd difference is exactly zero.
        t = torch.arange(20.0).view(1, 20, 1)
        traj = torch.cat([t, 2 * t, -0.5 * t], dim=-1)  # (1, 20, 3)
        assert smoothness_penalty(traj, order=3).item() == 0.0

    def test_constant_acceleration_has_zero_jerk_loss(self) -> None:
        # Parabolic (ballistic) motion: acceleration constant => jerk zero.
        t = torch.arange(30.0)
        z = (-0.5 * 9.81 * (t / 30) ** 2).view(1, 30, 1)
        # Not bit-exact zero in float32, but jerk is ~1e-9 vs a jittery ~1e-2.
        assert smoothness_penalty(z, order=3).item() < 1e-8

    def test_jitter_is_penalized(self) -> None:
        torch.manual_seed(0)
        smooth = torch.linspace(0, 1, 40).view(1, 40, 1).repeat(1, 1, 3)
        jittery = smooth + 0.05 * torch.randn(1, 40, 3)
        assert smoothness_penalty(jittery, order=3) > 10 * smoothness_penalty(
            smooth, order=3
        )

    def test_mask_excludes_padded_frames(self) -> None:
        torch.manual_seed(1)
        real = torch.linspace(0, 1, 12).view(1, 12, 1)  # smooth
        pad = 100.0 * torch.randn(1, 6, 1)  # garbage in padding
        seq = torch.cat([real, pad], dim=1)
        mask = torch.zeros(1, 18)
        mask[:, :12] = 1.0
        masked = smoothness_penalty(seq, mask, order=3)
        reference = smoothness_penalty(real, order=3)
        torch.testing.assert_close(masked, reference)

    def test_axis_weights_scale_contribution(self) -> None:
        torch.manual_seed(2)
        traj = torch.randn(1, 20, 3)
        base = smoothness_penalty(traj, order=2, axis_weights=[1.0, 1.0, 1.0])
        zeroed = smoothness_penalty(traj, order=2, axis_weights=[1.0, 1.0, 0.0])
        assert zeroed < base

    def test_short_sequence_returns_zero(self) -> None:
        assert smoothness_penalty(torch.randn(2, 3, 3), order=3).item() == 0.0

    def test_gradient_flows(self) -> None:
        traj = torch.randn(1, 15, 3, requires_grad=True)
        smoothness_penalty(traj, order=3).backward()
        assert traj.grad is not None and torch.isfinite(traj.grad).all()


class TestBallisticGravity:
    def test_second_difference_value(self) -> None:
        val = ballistic_second_difference(9.81, 1 / 30, 1.07)
        assert val < 0
        torch.testing.assert_close(
            torch.tensor(val), torch.tensor(-9.81 * (1 / 30) ** 2 / 1.07)
        )

    def test_free_fall_trajectory_has_zero_penalty(self) -> None:
        # Build normalized height with exactly the ballistic second difference.
        target = ballistic_second_difference(9.81, 1 / 30, 1.07)
        t = torch.arange(40.0)
        # z_norm[t] = z0 + v*t + 0.5*target*t^2 has 2nd difference == target.
        z = (1.0 + 0.02 * t + 0.5 * target * t * t).view(1, 40)
        loss = ballistic_gravity_penalty(z, target_second_difference=target)
        assert loss.item() < 1e-10

    def test_wrong_curvature_is_penalized(self) -> None:
        target = ballistic_second_difference(9.81, 1 / 30, 1.07)
        t = torch.arange(40.0)
        # Half the curvature (as if depth/scale were wrong) -> nonzero penalty.
        z_wrong = (1.0 + 0.02 * t + 0.5 * (0.5 * target) * t * t).view(1, 40)
        loss = ballistic_gravity_penalty(z_wrong, target_second_difference=target)
        assert loss.item() > 1e-4

    def test_mask_excludes_padding(self) -> None:
        target = ballistic_second_difference(9.81, 1 / 30, 1.07)
        t = torch.arange(20.0)
        z_real = (1.0 + 0.5 * target * t * t).view(1, 20)
        z = torch.cat([z_real, 50.0 * torch.randn(1, 8)], dim=1)
        mask = torch.zeros(1, 28)
        mask[:, :20] = 1.0
        loss = ballistic_gravity_penalty(z, mask, target_second_difference=target)
        assert loss.item() < 1e-9

    def test_short_sequence_returns_zero(self) -> None:
        assert (
            ballistic_gravity_penalty(
                torch.randn(1, 2), target_second_difference=-0.01
            ).item()
            == 0.0
        )
