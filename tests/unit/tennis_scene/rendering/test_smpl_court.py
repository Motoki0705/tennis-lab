"""Tests for placing SMPL body-local vertices in court coordinates."""

from __future__ import annotations

import numpy as np
import pytest

from src.tennis_scene.rendering.smpl_court import smpl_vertices_to_court


def test_root_centering_axis_swap_yaw_and_translation() -> None:
    # Two vertices; the regressor's root joint is vertex 0.
    verts = np.array([[[[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]]], dtype=np.float32)
    regressor = np.array([[1.0, 0.0]], dtype=np.float32)
    orient: np.ndarray = np.zeros((1, 1, 3), dtype=np.float32)
    position = np.array([[[5.0, -3.0, 0.9]]], dtype=np.float32)
    yaw = np.array([[np.pi / 2]], dtype=np.float32)

    court = smpl_vertices_to_court(verts, orient, position, yaw, regressor)

    # vertex 1 is +1 along SMPL Y (up) -> court +Z; yaw about Z leaves it up.
    np.testing.assert_allclose(court[0, 0, 0], [5.0, -3.0, 0.9], atol=1e-6)
    np.testing.assert_allclose(court[0, 0, 1], [5.0, -3.0, 1.9], atol=1e-6)


def test_yaw_rotates_horizontal_offsets_about_court_z() -> None:
    # SMPL +X offset, yaw +90deg -> court +Y.
    verts = np.array([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]], dtype=np.float32)
    regressor = np.array([[1.0, 0.0]], dtype=np.float32)
    court = smpl_vertices_to_court(
        verts,
        np.zeros((1, 1, 3), dtype=np.float32),
        np.zeros((1, 1, 3), dtype=np.float32),
        np.array([[np.pi / 2]], dtype=np.float32),
        regressor,
    )
    np.testing.assert_allclose(court[0, 0, 1], [0.0, 1.0, 0.0], atol=1e-6)


def test_regressor_vertex_count_mismatch_is_rejected() -> None:
    verts: np.ndarray = np.zeros((1, 1, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="joint_regressor"):
        smpl_vertices_to_court(
            verts,
            np.zeros((1, 1, 3), dtype=np.float32),
            np.zeros((1, 1, 3), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
        )
