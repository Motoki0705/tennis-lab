"""Tests for src/utils/geometry/rotation_conversions.py."""

import math
from typing import cast

import pytest
import torch

from src.utils.geometry.rotation_conversions import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    standardize_quaternion,
)


def random_rotation_matrices(batch: int, seed: int = 0) -> torch.Tensor:
    """Build valid random rotation matrices via QR decomposition."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, 3, 3, generator=gen)
    q, r = torch.linalg.qr(a)
    # Make the decomposition unique (positive diagonal) and det = +1
    sign = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * sign[:, None, :]
    det = torch.linalg.det(q)
    q[:, :, 0] = q[:, :, 0] * det[:, None]
    return cast(torch.Tensor, q)


class TestKnownValues:
    def test_identity_axis_angle(self) -> None:
        aa = torch.zeros(1, 3)
        torch.testing.assert_close(axis_angle_to_matrix(aa), torch.eye(3)[None])

    def test_z_rotation_90deg(self) -> None:
        aa = torch.tensor([[0.0, 0.0, math.pi / 2]])
        expected = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        torch.testing.assert_close(
            axis_angle_to_matrix(aa), expected, atol=1e-6, rtol=0
        )

    def test_quaternion_identity(self) -> None:
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(quaternion_to_matrix(q), torch.eye(3)[None])

    def test_euler_z_matches_axis_angle(self) -> None:
        angle = torch.tensor([0.3])
        euler = torch.stack([angle, torch.zeros(1), torch.zeros(1)], dim=-1)
        m_euler = euler_angles_to_matrix(euler, "ZYX")
        m_aa = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, 0.3]]))
        torch.testing.assert_close(m_euler, m_aa, atol=1e-6, rtol=0)


class TestRoundTrips:
    @pytest.fixture()
    def matrices(self) -> torch.Tensor:
        return random_rotation_matrices(64)

    def test_matrix_quaternion_roundtrip(self, matrices: torch.Tensor) -> None:
        q = matrix_to_quaternion(matrices)
        torch.testing.assert_close(
            quaternion_to_matrix(q), matrices, atol=1e-5, rtol=0
        )

    def test_matrix_axis_angle_roundtrip(self, matrices: torch.Tensor) -> None:
        aa = matrix_to_axis_angle(matrices)
        torch.testing.assert_close(
            axis_angle_to_matrix(aa), matrices, atol=1e-5, rtol=0
        )

    def test_matrix_6d_roundtrip(self, matrices: torch.Tensor) -> None:
        d6 = matrix_to_rotation_6d(matrices)
        torch.testing.assert_close(
            rotation_6d_to_matrix(d6), matrices, atol=1e-5, rtol=0
        )

    def test_axis_angle_quaternion_roundtrip(self) -> None:
        gen = torch.Generator().manual_seed(1)
        aa = torch.randn(64, 3, generator=gen)
        q = axis_angle_to_quaternion(aa)
        torch.testing.assert_close(
            quaternion_to_axis_angle(q), aa, atol=1e-5, rtol=0
        )

    def test_small_angle_stability(self) -> None:
        aa = torch.tensor([[1e-8, 0.0, 0.0], [0.0, 0.0, 0.0]])
        q = axis_angle_to_quaternion(aa)
        assert torch.isfinite(q).all()
        m = axis_angle_to_matrix(aa)
        torch.testing.assert_close(m, torch.eye(3).expand(2, 3, 3), atol=1e-6, rtol=0)


class TestProperties:
    def test_rotation_6d_produces_valid_rotations(self) -> None:
        gen = torch.Generator().manual_seed(2)
        d6 = torch.randn(32, 6, generator=gen)
        m = rotation_6d_to_matrix(d6)
        eye = torch.eye(3).expand(32, 3, 3)
        torch.testing.assert_close(m @ m.transpose(-1, -2), eye, atol=1e-5, rtol=0)
        torch.testing.assert_close(
            torch.linalg.det(m), torch.ones(32), atol=1e-5, rtol=0
        )

    def test_standardize_quaternion_nonnegative_real(self) -> None:
        gen = torch.Generator().manual_seed(3)
        q = torch.randn(32, 4, generator=gen)
        std = standardize_quaternion(q)
        assert (std[..., 0] >= 0).all()
        torch.testing.assert_close(
            quaternion_to_matrix(torch.nn.functional.normalize(q, dim=-1)),
            quaternion_to_matrix(torch.nn.functional.normalize(std, dim=-1)),
            atol=1e-5,
            rtol=0,
        )

    def test_matrix_to_quaternion_unit_norm(self) -> None:
        matrices = random_rotation_matrices(32, seed=4)
        q = matrix_to_quaternion(matrices)
        torch.testing.assert_close(
            q.norm(dim=-1), torch.ones(32), atol=1e-5, rtol=0
        )

    def test_invalid_matrix_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid rotation matrix shape"):
            matrix_to_quaternion(torch.zeros(2, 3, 4))

    def test_invalid_euler_convention_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid convention"):
            euler_angles_to_matrix(torch.zeros(1, 3), "XXY")
