from __future__ import annotations

import pytest
import torch

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
    validate_lifecycle_capacity,
)


def test_dataset_selects_only_contiguous_source_windows() -> None:
    dataset = CanonicalTrackingDataset(
        config={
            "data": {
                "seq_len_range": [1, 2],
                "lifecycle": {
                    "pack_to_query_slots": True,
                    "min_reuse_gap_frames": 0,
                    "randomize_slots_train": False,
                },
            },
            "model": {"num_queries": 1},
        },
    )
    assert dataset.contiguous_window(4) == slice(1, 3)


def test_collate_pads_only_declared_time_dimensions() -> None:
    batch = [
        {"frame_mask": torch.ones(2, dtype=torch.bool), "value": torch.ones(1, 2, 3)},
        {"frame_mask": torch.ones(4, dtype=torch.bool), "value": torch.ones(1, 4, 3)},
    ]
    collated = pad_and_stack_tracking_batch(
        batch, time_dimensions={"frame_mask": 0, "value": 1}
    )
    assert collated["frame_mask"].shape == (2, 4)
    assert collated["value"].shape == (2, 1, 4, 3)
    assert not collated["frame_mask"][0, 2:].any()


def test_lifecycle_capacity_rejects_generation_gap_shorter_than_packing_gap() -> None:
    with pytest.raises(ValueError, match="min_reuse_gap_frames"):
        validate_lifecycle_capacity(
            timeline_config={"max_concurrent": 4, "min_reuse_gap_frames": 2},
            data_config={"lifecycle": {"min_reuse_gap_frames": 4}},
            num_queries=4,
        )


def test_lifecycle_capacity_rejects_more_concurrent_tracks_than_queries() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        validate_lifecycle_capacity(
            timeline_config={"max_concurrent": 5, "min_reuse_gap_frames": 4},
            data_config={"lifecycle": {"min_reuse_gap_frames": 4}},
            num_queries=4,
        )
