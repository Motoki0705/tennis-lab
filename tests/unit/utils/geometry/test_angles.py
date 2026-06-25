"""Unit tests for :mod:`src.utils.geometry.angles`."""

from __future__ import annotations

import math

import pytest
import torch

from src.utils.geometry.angles import (
    angular_error,
    normalize_vector,
    signed_angle_around_axis,
    wrapped_angle_diff,
)


class TestNormalizeVector:
    def test_unit_norm_result(self) -> None:
        v = torch.tensor([3.0, 4.0])
        out = normalize_vector(v)
        assert out.norm().item() == pytest.approx(1.0, abs=1e-6)
        assert torch.allclose(out, torch.tensor([0.6, 0.8]), atol=1e-6)

    def test_preserves_direction_batched(self) -> None:
        v = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        out = normalize_vector(v)
        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_zero_vector_is_safe(self) -> None:
        v = torch.zeros(3)
        out = normalize_vector(v)
        # eps clamp avoids division by zero; result stays finite and near-zero.
        assert torch.isfinite(out).all()
        assert out.norm().item() < 1e-3


class TestWrappedAngleDiff:
    def test_zero_difference(self) -> None:
        a = torch.tensor([0.0, 1.0, -2.0])
        assert torch.allclose(wrapped_angle_diff(a, a), torch.zeros(3), atol=1e-6)

    def test_result_wrapped_into_pi_range(self) -> None:
        pred = torch.tensor([0.1])
        target = torch.tensor([2 * math.pi - 0.1])
        diff = wrapped_angle_diff(pred, target)
        assert diff.item() == pytest.approx(0.2, abs=1e-5)
        assert -math.pi <= diff.item() <= math.pi

    def test_sign_is_preserved(self) -> None:
        diff = wrapped_angle_diff(torch.tensor([1.0]), torch.tensor([0.0]))
        assert diff.item() == pytest.approx(1.0, abs=1e-6)


class TestAngularError:
    def test_orthogonal_cos_sin_pairs(self) -> None:
        pred = torch.tensor([1.0, 0.0])  # angle 0
        target = torch.tensor([0.0, 1.0])  # angle pi/2
        assert angular_error(pred, target).item() == pytest.approx(
            math.pi / 2, abs=1e-6
        )

    def test_error_is_non_negative_and_wrapped(self) -> None:
        pred = torch.tensor([math.cos(0.1), math.sin(0.1)])
        target = torch.tensor([math.cos(-0.1), math.sin(-0.1)])
        err = angular_error(pred, target)
        assert err.item() == pytest.approx(0.2, abs=1e-5)
        assert err.item() >= 0.0

    def test_batched_shape(self) -> None:
        pred = torch.randn(4, 5, 2)
        target = torch.randn(4, 5, 2)
        assert angular_error(pred, target).shape == (4, 5)


class TestSignedAngleAroundAxis:
    def test_quarter_turn_around_z(self) -> None:
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        axis = torch.tensor([0.0, 0.0, 1.0])
        assert signed_angle_around_axis(v1, v2, axis).item() == pytest.approx(
            math.pi / 2, abs=1e-6
        )

    def test_sign_flips_with_direction(self) -> None:
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, -1.0, 0.0])
        axis = torch.tensor([0.0, 0.0, 1.0])
        assert signed_angle_around_axis(v1, v2, axis).item() == pytest.approx(
            -math.pi / 2, abs=1e-6
        )

    def test_component_along_axis_is_ignored(self) -> None:
        # Adding an axis-parallel component to either vector must not change
        # the signed angle measured in the perpendicular plane.
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        axis = torch.tensor([0.0, 0.0, 1.0])
        base = signed_angle_around_axis(v1, v2, axis)
        shifted = signed_angle_around_axis(
            v1 + torch.tensor([0.0, 0.0, 3.0]),
            v2 - torch.tensor([0.0, 0.0, 2.0]),
            axis,
        )
        assert shifted.item() == pytest.approx(base.item(), abs=1e-6)
