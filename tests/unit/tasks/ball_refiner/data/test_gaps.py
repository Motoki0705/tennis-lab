"""Artificial gaps are label independent, clip-global and remove every feature."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.gaps import (
    fixed_gap_mask,
    mask_detector_evidence,
    random_gap_mask,
    validation_partition,
)
from src.tasks.ball_refiner.data.windows import CANDIDATE_FIELDS, detector_only_window
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def test_gap_removes_every_candidate_field_without_modifying_source(window_data):
    clip, config = window_data
    sample = detector_only_window(clip, 0, 5, config)
    before = {name: getattr(sample.inputs.candidates, name).clone() for name in CANDIDATE_FIELDS}
    mask = torch.tensor([[False, True, True, False, False]])
    result = mask_detector_evidence(sample.inputs, mask)
    for name in CANDIDATE_FIELDS:
        value = getattr(result.candidates, name)
        assert not value[:, 1:3].any()
        assert torch.equal(value[:, :1], before[name][:, :1])
        assert torch.equal(getattr(sample.inputs.candidates, name), before[name])
    assert torch.equal(result.timestamps_seconds, sample.inputs.timestamps_seconds)


def test_fixed_and_train_gaps_are_reproducible_and_bounded():
    mask = fixed_gap_mask(83, clip_id="x", block_length=17, lengths=(1, 4, 8), seed=42)
    np.testing.assert_array_equal(mask, fixed_gap_mask(83, clip_id="x", block_length=17, lengths=(1, 4, 8), seed=42))
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    assert set(np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)) <= {1, 4, 8}
    first = random_gap_mask(50, 17, lengths=(1, 4, 8), probability=.5, generator=torch.Generator().manual_seed(42))
    second = random_gap_mask(50, 17, lengths=(1, 4, 8), probability=.5, generator=torch.Generator().manual_seed(42))
    assert torch.equal(first, second)
    assert set(first.sum(1).tolist()) <= {0, 1, 4, 8}


def test_validation_partition_groups_cameras_and_never_accepts_test(tmp_path):
    path = write_store_clip(tmp_path / "store", "meiji/video_000/clip_000/cam0", [frame(0, ball())], source="meiji", split="val")
    record = BallFrameStore(path).clips[0]
    records = tuple(replace(record, clip_id=f"meiji/video_000/clip_{i:03d}/cam{j}") for i in range(6) for j in range(3))
    partition = validation_partition(records, 42)
    assert partition == validation_partition(tuple(reversed(records)), 42)
    assert len(partition["selection"]) == len(partition["calibration"]) == 9
    assert set(partition["selection"]).isdisjoint(partition["calibration"])
    for i in range(6):
        assert sum(f"meiji/video_000/clip_{i:03d}/cam{j}" in partition["selection"] for j in range(3)) in (0, 3)
    with pytest.raises(ValueError, match="Meiji val"):
        validation_partition((replace(record, split="test"),), 42)
