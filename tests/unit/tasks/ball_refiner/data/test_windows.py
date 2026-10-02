"""Unpadded temporal ownership, source fairness and person-only collation."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.data.evidence import ClipEvidence
from src.tasks.ball_refiner.data.windows import (
    BalancedSourceSampler,
    LoadedClip,
    RefinerWindowDataset,
    collate_windows,
    detector_only_window,
    window_owners,
    window_starts,
)


def test_tail_and_centre_ties_are_label_independent():
    starts = window_starts(12, 5, 3)
    assert starts == (0, 3, 6, 7)
    owners = window_owners(12, starts, 5)
    assert owners.tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3]
    with pytest.raises(ValueError, match="no padding"):
        window_starts(4, 5, 3)
    with pytest.raises(ValueError, match="uncovered"):
        window_owners(12, [0, 7], 5)


def test_dataset_reports_short_and_unknown_windows_and_retains_pts(window_data):
    clip, config = window_data
    short = replace(clip, record=replace(clip.record, clip_id="short"))
    # Keep the short fixture internally aligned instead of pretending repeated frames exist.
    arrays = {key: value[:3].copy() for key, value in clip.evidence.arrays().items()}
    short = LoadedClip(replace(short.record, frame_count=3), ClipEvidence.from_arrays(
        arrays, config=clip.evidence.candidates.config, heatmap_size_hw=(7, 9), window_length=1,
    ), replace(clip.targets, **{key: getattr(clip.targets, key)[:3] for key in clip.targets.__dataclass_fields__}))
    dataset = RefinerWindowDataset([clip, short], length=5, stride=3, config=config)
    assert len(dataset) == 2
    assert dataset.excluded == [
        {"clip_id": clip.record.clip_id, "reason": "no_known_target", "windows": 1},
        {"clip_id": "short", "reason": "short_clip", "frames": 3},
    ]
    item = dataset[1]
    assert item.start == 3
    np.testing.assert_array_equal(item.inputs.timestamps_seconds[0], clip.evidence.timestamps_seconds[3:8])
    assert int(item.target.presence_valid.sum()) == 1
    assert torch.isnan(item.target.uv[~item.target.position_valid]).all()
    assert item.inputs.pose_uv.shape == (1, 5, 0, 4, 2)
    with pytest.raises(ValueError, match="non-train"):
        RefinerWindowDataset([replace(clip, record=replace(clip.record, split="val"))], length=5, stride=3, config=config)
    with pytest.raises(ValueError, match="Detector-only"):
        detector_only_window(clip, 0, 5, replace(config, use_pose=True))


def test_collate_only_pads_people_and_does_not_mutate_samples(window_data):
    clip, config = window_data
    first = detector_only_window(clip, 0, 5, config)
    second = replace(first, inputs=replace(first.inputs,
        pose_uv=torch.ones(1, 5, 2, 4, 2), pose_confidence=torch.ones(1, 5, 2, 4),
        pose_valid=torch.ones(1, 5, 2, 4, dtype=torch.bool)))
    batch = collate_windows([first, second]).to(torch.device("cpu"))
    assert batch.inputs.pose_uv.shape == (2, 5, 2, 4, 2)
    assert not batch.inputs.pose_valid[0].any() and batch.inputs.pose_valid[1].all()
    assert not batch.inputs.pose_uv[0].any()
    assert first.inputs.pose_uv.shape[2] == 0
    batch.inputs.candidates.coords.zero_()
    assert first.inputs.candidates.coords.any()
    with pytest.raises(ValueError, match="real time"):
        collate_windows([first, detector_only_window(clip, 0, 4, config)])


def test_source_balancing_is_seeded_with_replacement():
    sampler = BalancedSourceSampler({"large": list(range(1, 101)), "small": [0], "third": [101]}, draws=101, seed=42)
    values = list(sampler)
    assert values == list(sampler)
    counts = [values.count(0), values.count(101), sum(0 < x < 101 for x in values)]
    assert max(counts) - min(counts) <= 1
    sampler.epoch = 1
    assert values != list(sampler)
    assert len(values) == len(sampler)
