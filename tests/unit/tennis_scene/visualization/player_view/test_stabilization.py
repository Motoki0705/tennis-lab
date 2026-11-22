"""Head-camera stabilization preserves geometry, intent and explicit gaps."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.tennis_scene.visualization.player_view.stabilization import (
    StabilizationConfig,
    smooth_camera_rotations,
)


def _rotation(values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    axes = np.zeros((len(values), 3))
    axes[:, 2] = np.deg2rad(values)
    return cast(np.ndarray, Rotation.from_rotvec(axes).as_matrix())


def test_wrap_crossing_averages_about_180_instead_of_zero() -> None:
    result = smooth_camera_rotations(
        _rotation([179, -179, 179]), np.ones(3, bool), 60, StabilizationConfig(0.06)
    )
    # Its forward-facing X is near -X, never the +X of an Euler component mean.
    assert (result[:, 0, 0] < -0.999).all()
    np.testing.assert_allclose(np.linalg.det(result), 1, atol=1e-14)


def test_constant_tilt_and_roll_are_preserved_without_horizon_lock() -> None:
    rotation = Rotation.from_euler("xyz", [23, -41, 17], degrees=True).as_matrix()
    result = smooth_camera_rotations(
        np.tile(rotation, (7, 1, 1)), np.ones(7, bool), 60, StabilizationConfig(0.12)
    )
    np.testing.assert_allclose(result, np.tile(rotation, (7, 1, 1)), atol=1e-14)


def test_constant_angular_velocity_and_fixed_coordinate_changes() -> None:
    rotation = _rotation(np.arange(121) - 60)
    config = StabilizationConfig(0.06)
    result = smooth_camera_rotations(rotation, np.ones(121, bool), 60, config)
    np.testing.assert_allclose(result[12:-12], rotation[12:-12], atol=1e-14)
    left = Rotation.from_euler("xyz", [37, -21, 12], degrees=True).as_matrix()
    right = Rotation.from_euler("xyz", [-18, 29, 43], degrees=True).as_matrix()
    changed = smooth_camera_rotations(
        left @ rotation @ right, np.ones(121, bool), 60, config
    )
    np.testing.assert_allclose(changed, left @ result @ right, atol=1e-14)


def test_gap_is_never_bridged_and_singleton_uses_its_only_sample() -> None:
    valid = np.array([True, True, False, True, False, True, True])
    rotation = _rotation([10, 40, 0, -53, 0, 70, 90])
    rotation[~valid] = np.nan
    original = rotation.copy()
    result = smooth_camera_rotations(rotation, valid, 60, StabilizationConfig(0.12))
    rotation[5:] = _rotation([-145, -80])
    changed = smooth_camera_rotations(rotation, valid, 60, StabilizationConfig(0.12))
    np.testing.assert_array_equal(result[:5], changed[:5])
    np.testing.assert_allclose(result[3], original[3], atol=1e-14)
    assert np.isnan(result[~valid]).all()
    np.testing.assert_array_equal(rotation[:5], original[:5])


def test_zero_sigma_preserves_raw_values_and_entirely_missing_is_valid() -> None:
    rotation = _rotation([20, -10, 45])
    valid = np.array([True, False, True])
    rotation[~valid] = np.nan
    result = smooth_camera_rotations(rotation, valid, 59.94, StabilizationConfig(0))
    np.testing.assert_array_equal(result, rotation)
    assert not np.shares_memory(result, rotation)
    missing = np.full((3, 3, 3), np.nan)
    assert np.isnan(
        smooth_camera_rotations(
            missing, np.zeros(3, bool), 60, StabilizationConfig(0.06)
        )
    ).all()


@pytest.mark.parametrize("fps", [30, 60])
def test_temporal_sigma_has_known_low_frequency_gain_without_phase_lag(
    fps: int,
) -> None:
    times = np.arange(fps * 4 + 1) / fps
    # Small angles give the independent scalar Gaussian frequency response.
    raw = 1e-4 * np.sin(2 * np.pi * times)
    result = smooth_camera_rotations(
        _rotation(np.rad2deg(raw)),
        np.ones(len(raw), bool),
        fps,
        StabilizationConfig(0.06),
    )
    angle = Rotation.from_matrix(result).as_rotvec()[:, 2]
    interior = slice(fps, -fps)
    basis = np.stack(
        [np.sin(2 * np.pi * times[interior]), np.cos(2 * np.pi * times[interior])],
        axis=1,
    )
    sine, cosine = np.linalg.lstsq(basis, angle[interior], rcond=None)[0]
    assert sine / 1e-4 == pytest.approx(0.932, abs=0.002)
    assert abs(cosine) < 1e-12


def test_nonunique_rotation_mean_is_rejected() -> None:
    axes = np.array([[np.pi, 0, 0], [0, 0, 0], [np.pi, 0, 0]])
    # At the middle: identity has weight 1, the two half-turns weight .5 each.
    config = StabilizationConfig(1 / np.sqrt(2 * np.log(2)))
    with pytest.raises(ValueError, match="nonunique"):
        smooth_camera_rotations(
            Rotation.from_rotvec(axes).as_matrix(), np.ones(3, bool), 1, config
        )


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf, True])
def test_invalid_strength_is_rejected(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        StabilizationConfig(sigma)


@pytest.mark.parametrize("bad", ["rotation", "missing", "mask", "fps"])
def test_invalid_inputs_never_get_repaired(bad: str) -> None:
    rotation = _rotation([10, 30])
    valid: np.ndarray = np.ones(2, bool)
    fps = 60
    if bad == "rotation":
        rotation[0, 0, 0] = 3
    elif bad == "missing":
        valid[1] = False  # Missing rotations must be all NaN, not zeros or identity.
    elif bad == "mask":
        valid = np.ones(2, np.int64)
    else:
        fps = 0
    with pytest.raises(ValueError):
        smooth_camera_rotations(rotation, valid, fps, StabilizationConfig(0.06))
