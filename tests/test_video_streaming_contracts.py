from __future__ import annotations

import torch

from src.utils.video import (
    FramePacket,
    PrefetchIterator,
    iter_temporal_batches,
    iter_temporal_windows,
)


def _packets(count: int) -> list[FramePacket[int]]:
    return [
        FramePacket(index=index, frame=index, original_size=(640, 360))
        for index in range(count)
    ]


def test_backfill_tail_window_ends_at_last_frame() -> None:
    windows = list(
        iter_temporal_windows(
            _packets(19),
            sequence_length=8,
            stride=8,
            tail_policy="backfill",
        )
    )

    assert [window.start_index for window in windows] == [0, 8, 11]
    assert [window.frame_indices for window in windows] == [
        tuple(range(0, 8)),
        tuple(range(8, 16)),
        tuple(range(11, 19)),
    ]


def test_backfill_tail_does_not_duplicate_exact_final_window() -> None:
    windows = list(
        iter_temporal_windows(
            _packets(16),
            sequence_length=8,
            stride=8,
            tail_policy="backfill",
        )
    )

    assert [window.start_index for window in windows] == [0, 8]
    assert [window.frame_indices for window in windows] == [
        tuple(range(0, 8)),
        tuple(range(8, 16)),
    ]


def test_short_stream_repeats_last_frame_to_form_one_window() -> None:
    windows = list(
        iter_temporal_windows(
            _packets(3),
            sequence_length=8,
            stride=8,
            tail_policy="backfill",
        )
    )

    assert len(windows) == 1
    assert windows[0].start_index == 0
    assert windows[0].frame_indices == (0, 1, 2, 2, 2, 2, 2, 2)


def test_temporal_batches_stack_window_tensors() -> None:
    frame_packets = [
        FramePacket(
            index=index,
            frame=torch.full((3, 2, 2), float(index)),
            original_size=(2, 2),
        )
        for index in range(10)
    ]
    windows = iter_temporal_windows(
        frame_packets,
        sequence_length=4,
        stride=4,
        tail_policy="backfill",
    )

    batches = list(iter_temporal_batches(windows, batch_size=2))

    assert len(batches) == 2
    assert batches[0].tensor.shape == (2, 4, 3, 2, 2)
    assert batches[0].windows[0].frame_indices == (0, 1, 2, 3)
    assert batches[0].windows[1].frame_indices == (4, 5, 6, 7)
    assert batches[1].tensor.shape == (1, 4, 3, 2, 2)
    assert batches[1].windows[0].frame_indices == (6, 7, 8, 9)


def test_prefetch_iterator_preserves_order() -> None:
    prefetched = PrefetchIterator(range(5), max_prefetch=2)

    assert list(prefetched) == [0, 1, 2, 3, 4]
