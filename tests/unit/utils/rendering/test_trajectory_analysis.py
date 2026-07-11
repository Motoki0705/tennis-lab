"""Unit tests for ball-trajectory kinematics (speed and bounce extraction)."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.rendering.trajectory_analysis import compute_speeds, detect_bounces


def _constant_velocity_track(num_frames: int, velocity: tuple[float, float, float]) -> np.ndarray:
    t: np.ndarray = np.arange(num_frames, dtype=np.float64)[:, None]
    track: np.ndarray = np.asarray(t * np.asarray(velocity)[None, :], dtype=np.float32)
    return track


class TestComputeSpeeds:
    def test_constant_velocity_gives_constant_speed(self) -> None:
        fps = 30.0
        positions = _constant_velocity_track(10, (0.1, 0.2, 0.0))
        expected = float(np.linalg.norm([0.1, 0.2, 0.0])) * fps

        speeds = compute_speeds(positions, fps)

        assert speeds.shape == (10,)
        np.testing.assert_allclose(speeds, expected, rtol=1e-5)

    def test_missing_neighbour_falls_back_to_one_sided(self) -> None:
        fps = 10.0
        positions = _constant_velocity_track(5, (1.0, 0.0, 0.0))
        positions[3] = np.nan

        speeds = compute_speeds(positions, fps)

        # t=2: p[3] is missing -> backward difference from p[1] to p[2].
        np.testing.assert_allclose(speeds[2], 1.0 * fps, rtol=1e-6)
        # t=3: the frame itself is missing but both neighbours exist -> central.
        np.testing.assert_allclose(speeds[3], 1.0 * fps, rtol=1e-6)

    def test_isolated_frame_gives_nan(self) -> None:
        positions: np.ndarray = np.full((5, 3), np.nan, dtype=np.float32)
        positions[2] = (1.0, 2.0, 3.0)

        speeds = compute_speeds(positions, 30.0)

        assert np.isnan(speeds).all()

    def test_short_track_gives_all_nan(self) -> None:
        speeds = compute_speeds(np.zeros((1, 3), dtype=np.float32), 30.0)
        assert np.isnan(speeds).all()

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="positions must have shape"):
            compute_speeds(np.zeros((5, 2), dtype=np.float32), 30.0)

    def test_invalid_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="fps must be"):
            compute_speeds(np.zeros((5, 3), dtype=np.float32), 0.0)


def _bouncing_track(bounce_frames: list[int], num_frames: int) -> np.ndarray:
    """V-shaped Z track touching ~0 at each bounce frame, apex ~1 m."""
    z = np.full(num_frames, 1.0)
    for t in range(num_frames):
        nearest = min(abs(t - b) for b in bounce_frames)
        z[t] = min(1.0, 0.02 + 0.1 * nearest)
    positions = np.zeros((num_frames, 3))
    positions[:, 0] = np.arange(num_frames) * 0.2
    positions[:, 2] = z
    return positions.astype(np.float32)


class TestDetectBounces:
    def test_finds_ground_bounces(self) -> None:
        positions = _bouncing_track([10, 30], num_frames=40)

        bounces = detect_bounces(positions)

        assert bounces.tolist() == [10, 30]

    def test_high_local_minimum_is_not_a_bounce(self) -> None:
        # Local minimum at z=1.0 (e.g. a racket impact), never near the ground.
        positions = _bouncing_track([10], num_frames=20)
        positions[:, 2] += 1.0

        bounces = detect_bounces(positions)

        assert bounces.size == 0

    def test_close_minima_merge_to_lowest(self) -> None:
        positions = _bouncing_track([10], num_frames=20)
        # Add a second, shallower dip 2 frames later (within min_separation).
        positions[12, 2] = 0.05
        positions[11, 2] = 0.2
        positions[13, 2] = 0.2

        bounces = detect_bounces(positions, min_separation=5)

        assert bounces.tolist() == [10]

    def test_missing_neighbour_disqualifies_candidate(self) -> None:
        positions = _bouncing_track([10], num_frames=20)
        positions[9] = np.nan

        bounces = detect_bounces(positions)

        assert 10 not in bounces.tolist()

    def test_invalid_parameters_raise(self) -> None:
        positions = _bouncing_track([10], num_frames=20)
        with pytest.raises(ValueError, match="max_bounce_height"):
            detect_bounces(positions, max_bounce_height=0.0)
        with pytest.raises(ValueError, match="min_prominence"):
            detect_bounces(positions, min_prominence=-1.0)
        with pytest.raises(ValueError, match="min_separation"):
            detect_bounces(positions, min_separation=0)
