from __future__ import annotations

import torch

from src.tasks.base.data.canonical_tracking import (
    CanonicalTrackingDataset,
    pad_and_stack_tracking_batch,
)


def test_dataset_resolves_explicit_scene_split(tmp_path) -> None:
    (tmp_path / "scenes" / "scene_a").mkdir(parents=True)
    (tmp_path / "train.txt").write_text("scene_a\n")
    dataset = CanonicalTrackingDataset(
        scene_dir=tmp_path, split_file="train.txt"
    )
    assert len(dataset) == 1
    assert dataset.scenes == [tmp_path / "scenes" / "scene_a"]


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
