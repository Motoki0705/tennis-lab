"""Tests for ball-detection trajectory gating."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_detection.inference.trajectory_gate import apply_trajectory_gate


def _linear_positions(
    num_frames: int,
    *,
    velocity_px: float = 20.0,
) -> NDArray[np.float32]:
    frame: NDArray[np.float32] = np.arange(num_frames, dtype=np.float32)
    positions: NDArray[np.float32] = np.stack(
        [
            50.0 + velocity_px * frame,
            np.full(num_frames, 120.0, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return positions


def test_rejects_single_frame_teleport_on_noisy_parabola() -> None:
    frame: NDArray[np.float32] = np.arange(24, dtype=np.float32)
    positions = np.stack(
        [
            80.0 + 12.0 * frame,
            180.0 + 0.2 * (frame - 12.0) ** 2,
        ],
        axis=1,
    ).astype(np.float32)
    noise = np.asarray(
        np.random.default_rng(0).normal(0.0, 0.7, size=positions.shape),
        dtype=np.float32,
    )
    positions += noise
    positions[12, 0] += 180.0
    visibility = np.ones(positions.shape[0], dtype=np.bool_)
    score = np.full(positions.shape[0], 0.9, dtype=np.float32)

    gated_visibility, diagnostics = apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=2,
        max_support_gap=5,
        max_passes=2,
    )

    assert diagnostics.rejected_indices == [12]
    assert diagnostics.rejected[0].residual_px > 60.0
    assert not bool(gated_visibility[12])
    assert gated_visibility.sum() == visibility.sum() - 1


def test_keeps_normal_fast_motion_at_twenty_pixels_per_frame() -> None:
    positions = _linear_positions(16, velocity_px=20.0)
    visibility = np.ones(positions.shape[0], dtype=np.bool_)
    score = np.full(positions.shape[0], 0.8, dtype=np.float32)

    gated_visibility, diagnostics = apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=2,
        max_support_gap=5,
        max_passes=2,
    )

    np.testing.assert_array_equal(gated_visibility, visibility)
    assert diagnostics.rejected_indices == []


def test_rejects_two_frame_false_run_after_second_pass() -> None:
    positions = _linear_positions(12, velocity_px=20.0)
    positions[5, 0] = 350.0
    positions[6, 0] = 270.0
    visibility = np.ones(positions.shape[0], dtype=np.bool_)
    score = np.full(positions.shape[0], 0.9, dtype=np.float32)

    gated_visibility, diagnostics = apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=2,
        max_support_gap=5,
        max_passes=2,
    )

    assert diagnostics.passes_run == 2
    assert diagnostics.rejected_indices == [6, 5]
    assert not bool(gated_visibility[5])
    assert not bool(gated_visibility[6])


def test_keeps_isolated_visible_detection_without_support() -> None:
    positions = _linear_positions(9, velocity_px=20.0)
    visibility = np.zeros(positions.shape[0], dtype=np.bool_)
    visibility[4] = True
    score = np.full(positions.shape[0], 0.7, dtype=np.float32)

    gated_visibility, diagnostics = apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=2,
        max_support_gap=5,
        max_passes=2,
    )

    np.testing.assert_array_equal(gated_visibility, visibility)
    assert diagnostics.rejected_indices == []


def test_support_gap_boundary_is_inclusive() -> None:
    positions = _linear_positions(21, velocity_px=10.0)
    positions[10] = [500.0, 120.0]
    visibility = np.zeros(positions.shape[0], dtype=np.bool_)
    visibility[[4, 5, 10, 15, 16]] = True
    score = np.full(positions.shape[0], 0.9, dtype=np.float32)

    gated_at_boundary, diagnostics_at_boundary = apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=1,
        max_support_gap=5,
        max_passes=2,
    )
    gated_beyond_boundary, diagnostics_beyond_boundary = apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=1,
        max_support_gap=4,
        max_passes=2,
    )

    assert diagnostics_at_boundary.rejected_indices == [10]
    assert not bool(gated_at_boundary[10])
    np.testing.assert_array_equal(gated_beyond_boundary, visibility)
    assert diagnostics_beyond_boundary.rejected_indices == []


def test_apply_trajectory_gate_does_not_mutate_inputs() -> None:
    positions = _linear_positions(12, velocity_px=20.0)
    positions[6, 0] += 180.0
    visibility = np.ones(positions.shape[0], dtype=np.bool_)
    score = np.full(positions.shape[0], 0.9, dtype=np.float32)
    original_positions = positions.copy()
    original_visibility = visibility.copy()
    original_score = score.copy()

    apply_trajectory_gate(
        positions,
        visibility,
        score,
        max_residual_px=60.0,
        k_support=2,
        max_support_gap=5,
        max_passes=2,
    )

    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(visibility, original_visibility)
    np.testing.assert_array_equal(score, original_score)


def test_rejects_ambiguous_input_shapes() -> None:
    positions: NDArray[np.float32] = np.zeros((4, 2), dtype=np.float32)
    visibility: NDArray[np.bool_] = np.ones(3, dtype=np.bool_)
    score: NDArray[np.float32] = np.ones(4, dtype=np.float32)

    with pytest.raises(ValueError, match="visibility length must match"):
        apply_trajectory_gate(
            positions,
            visibility,
            score,
            max_residual_px=60.0,
            k_support=2,
            max_support_gap=5,
            max_passes=2,
        )
