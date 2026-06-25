"""Unit tests for :mod:`src.utils.video.batching`."""

from __future__ import annotations

import pytest
import torch

from src.utils.video.batching import iter_temporal_batches
from src.utils.video.types import TemporalWindow


def _window(start: int, *, seq_len: int = 2, c: int = 3, h: int = 4, w: int = 4):
    frames = tuple(torch.rand(c, h, w) for _ in range(seq_len))
    return TemporalWindow(
        start_index=start,
        frame_indices=tuple(range(start, start + seq_len)),
        frames=frames,
    )


class TestIterTemporalBatches:
    def test_stacks_into_bt_chw(self) -> None:
        windows = [_window(0), _window(2)]
        batches = list(iter_temporal_batches(windows, batch_size=2))
        assert len(batches) == 1
        assert batches[0].tensor.shape == (2, 2, 3, 4, 4)
        assert batches[0].windows == tuple(windows)

    def test_partial_final_batch(self) -> None:
        windows = [_window(i) for i in range(3)]
        batches = list(iter_temporal_batches(windows, batch_size=2))
        assert [b.tensor.shape[0] for b in batches] == [2, 1]

    def test_empty_input_yields_nothing(self) -> None:
        assert list(iter_temporal_batches([], batch_size=4)) == []

    def test_non_positive_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            list(iter_temporal_batches([_window(0)], batch_size=0))
