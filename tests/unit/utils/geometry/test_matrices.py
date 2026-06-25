"""Unit tests for :mod:`src.utils.geometry.matrices`."""

from __future__ import annotations

import math

import numpy as np

from src.utils.geometry.matrices import (
    apply_plcs_transform,
    apply_plcs_transform_batch,
    axis_angle_to_rotation_matrix,
    rotation_matrix_y,
    rotation_matrix_z,
)


def _is_rotation(mat: np.ndarray) -> bool:
    identity = np.eye(3, dtype=mat.dtype)
    orthonormal = np.allclose(mat @ mat.T, identity, atol=1e-5)
    proper = math.isclose(float(np.linalg.det(mat)), 1.0, abs_tol=1e-5)
    return orthonormal and proper


class TestRotationMatrixY:
    def test_zero_yaw_is_identity(self) -> None:
        np.testing.assert_allclose(rotation_matrix_y(0.0), np.eye(3), atol=1e-7)

    def test_is_a_proper_rotation(self) -> None:
        assert _is_rotation(rotation_matrix_y(0.9))

    def test_rotates_x_into_minus_z_for_quarter_turn(self) -> None:
        # vertices @ R.T convention: a quarter Y-turn maps +x to -z.
        rot = rotation_matrix_y(math.pi / 2)
        point = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        rotated = point @ rot.T
        np.testing.assert_allclose(rotated, [0.0, 0.0, -1.0], atol=1e-6)

    def test_returns_float32(self) -> None:
        assert rotation_matrix_y(0.3).dtype == np.float32


class TestRotationMatrixZ:
    def test_batched_shape(self) -> None:
        yaw: np.ndarray = np.zeros((4, 5), dtype=np.float32)
        assert rotation_matrix_z(yaw).shape == (4, 5, 3, 3)

    def test_zero_yaw_is_identity(self) -> None:
        rot = rotation_matrix_z(np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(rot, np.broadcast_to(np.eye(3), (2, 3, 3)), atol=1e-7)

    def test_each_matrix_is_a_proper_rotation(self) -> None:
        yaw = np.array([0.1, 1.2, -0.7], dtype=np.float32)
        for mat in rotation_matrix_z(yaw):
            assert _is_rotation(mat)


class TestAxisAngleToRotationMatrix:
    def test_zero_rotation_is_identity(self) -> None:
        out = axis_angle_to_rotation_matrix(np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(out, np.eye(3), atol=1e-6)

    def test_matches_rotation_matrix_z_for_z_axis(self) -> None:
        theta = 0.73
        aa = np.array([0.0, 0.0, theta], dtype=np.float32)
        from_axis_angle = axis_angle_to_rotation_matrix(aa)
        from_z = rotation_matrix_z(np.array(theta, dtype=np.float32))
        np.testing.assert_allclose(from_axis_angle, from_z, atol=1e-6)

    def test_batched_proper_rotations(self) -> None:
        aa = np.array([[0.0, 0.0, 0.5], [0.3, -0.2, 0.1]], dtype=np.float32)
        out = axis_angle_to_rotation_matrix(aa)
        assert out.shape == (2, 3, 3)
        for mat in out:
            assert _is_rotation(mat)


class TestApplyPlcsTransform:
    def test_identity_transform_is_noop(self) -> None:
        verts = np.random.rand(5, 3).astype(np.float32)
        out = apply_plcs_transform(verts, np.zeros(3, dtype=np.float32), 0.0)
        np.testing.assert_allclose(out, verts, atol=1e-6)

    def test_translation_only(self) -> None:
        verts: np.ndarray = np.zeros((3, 3), dtype=np.float32)
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = apply_plcs_transform(verts, pos, 0.0)
        np.testing.assert_allclose(out, np.broadcast_to(pos, (3, 3)), atol=1e-6)

    def test_batch_matches_per_frame(self) -> None:
        rng = np.random.default_rng(0)
        verts = rng.random((4, 6, 3)).astype(np.float32)
        positions = rng.random((4, 3)).astype(np.float32)
        yaws = rng.random(4).astype(np.float32)
        batched = apply_plcs_transform_batch(verts, positions, yaws)
        for t in range(4):
            per_frame = apply_plcs_transform(verts[t], positions[t], float(yaws[t]))
            np.testing.assert_allclose(batched[t], per_frame, atol=1e-5)
