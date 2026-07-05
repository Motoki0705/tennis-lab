"""Unit tests for :mod:`src.utils.inference.windowed`."""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.inference.windowed import blend_windows, window_slices


class TestWindowSlices:
    def test_short_sequence_single_window(self) -> None:
        assert window_slices(100, 256, 64) == [(0, 100)]

    def test_exact_fit_single_window(self) -> None:
        assert window_slices(256, 256, 64) == [(0, 256)]

    def test_covers_every_frame_with_last_window_end_aligned(self) -> None:
        slices = window_slices(811, 256, 64)
        assert slices[0][0] == 0
        assert slices[-1][1] == 811
        covered: np.ndarray = np.zeros(811, dtype=bool)
        for start, end in slices:
            assert end - start == 256
            covered[start:end] = True
        assert covered.all()

    def test_consecutive_windows_overlap(self) -> None:
        slices = window_slices(811, 256, 64)
        for (s0, e0), (s1, _) in zip(slices, slices[1:], strict=False):
            assert s1 < e0  # overlap present
            assert s1 > s0  # forward progress

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="total_len"):
            window_slices(0, 256, 64)
        with pytest.raises(ValueError, match="window_len"):
            window_slices(10, 0, 0)
        with pytest.raises(ValueError, match="overlap"):
            window_slices(10, 4, 4)
        with pytest.raises(ValueError, match="overlap"):
            window_slices(10, 4, -1)


class TestBlendWindows:
    def test_single_chunk_roundtrip(self) -> None:
        values = np.random.rand(10, 3)
        out = blend_windows([(0, values)], 10)
        np.testing.assert_allclose(out, values)

    def test_constant_windows_blend_to_constant(self) -> None:
        chunks = [
            (start, np.full((end - start, 2), 5.0))
            for start, end in window_slices(500, 128, 32)
        ]
        out = blend_windows(chunks, 500)
        np.testing.assert_allclose(out, 5.0)

    def test_overlap_prefers_window_interior(self) -> None:
        # Two windows disagree; the blend must move from window A's value to
        # window B's value across the overlap, weighted toward each interior.
        a = (0, np.zeros((8, 1)))
        b = (4, np.ones((8, 1)))
        out = blend_windows([a, b], 12)
        np.testing.assert_allclose(out[:4], 0.0)  # A only
        np.testing.assert_allclose(out[8:], 1.0)  # B only
        assert (np.diff(out[3:9, 0]) >= 0).all()  # monotone transition

    def test_reconstructs_smooth_signal(self) -> None:
        t = np.linspace(0, 4 * np.pi, 811)
        signal = np.stack([np.sin(t), np.cos(t)], axis=1)
        chunks = [
            (start, signal[start:end]) for start, end in window_slices(811, 256, 64)
        ]
        out = blend_windows(chunks, 811)
        np.testing.assert_allclose(out, signal, atol=1e-12)

    def test_uncovered_frames_raise(self) -> None:
        with pytest.raises(ValueError, match="uncovered"):
            blend_windows([(0, np.zeros((4, 1)))], 10)

    def test_out_of_bounds_chunk_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            blend_windows([(8, np.zeros((4, 1)))], 10)

    def test_mismatched_trailing_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="trailing shape"):
            blend_windows([(0, np.zeros((5, 1))), (5, np.zeros((5, 2)))], 10)

    def test_empty_chunks_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            blend_windows([], 10)
