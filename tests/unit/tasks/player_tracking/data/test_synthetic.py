from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.player_tracking.data.synthetic import SyntheticPlayerTrackingDataset


def _config():
    config = OmegaConf.load(Path("src/tasks/player_tracking/configs/train.yaml")).data
    config.split_sizes.train = 2
    config.min_frames = config.max_frames = 10
    config.min_views = config.max_views = 3
    config.min_persons = config.max_persons = 3
    config.detection_dropout_probability = 0.0
    config.joint_dropout_probability = 0.0
    config.false_positive_probability = 1.0
    return config


def test_player_detections_are_shuffled_without_losing_gt_debug_index() -> None:
    sample = SyntheticPlayerTrackingDataset(_config(), split="train")[0]
    valid = sample["detection_mask"]
    indices = sample["detection_gt_index"][valid]
    assert (indices == -1).any()
    assert (indices >= 0).any()
    assert (indices < sample["target_person_mask"].sum()).all()
    signatures = []
    for view in range(3):
        for frame in range(10):
            row = sample["detection_gt_index"][view, frame][valid[view, frame]]
            true_row = tuple(row[row >= 0].tolist())
            if len(true_row) >= 2:
                signatures.append(true_row)
    assert len(set(signatures)) > 1


def test_player_padding_birth_death_visibility_and_false_positive_masks() -> None:
    config = _config()
    config.min_frames = 6
    config.max_frames = 10
    config.min_views = 1
    config.max_views = 3
    dataset = SyntheticPlayerTrackingDataset(config, split="train")
    sample = dataset[1]
    repeated = dataset[1]
    assert sample["scene_format_version"].item() == 2
    sample_tensors = dict(sample)
    repeated_tensors = dict(repeated)
    for key, value in sample_tensors.items():
        assert torch.equal(value, repeated_tensors[key])
    assert not sample["detection_mask"][:, ~sample["frame_mask"]].any()
    assert sample["human_vis"][sample["detection_mask"]].any(-1).all()
    present = sample["person_present"][:, sample["target_person_mask"]]
    assert present.any() and (~present).any()
