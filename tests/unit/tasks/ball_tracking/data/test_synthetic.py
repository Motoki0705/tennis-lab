from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.tasks.ball_tracking.data.synthetic import SyntheticBallTrackingDataset


def _config():
    path = Path("src/tasks/ball_tracking/configs/train.yaml")
    config = OmegaConf.load(path).data
    config.split_sizes.train = 2
    config.min_frames = config.max_frames = 10
    config.min_views = config.max_views = 3
    config.min_balls = config.max_balls = 3
    config.dropout_probability = 0.0
    config.false_positive_probability = 1.0
    config.duplicate_probability = 0.0
    return config


def test_synthetic_candidates_preserve_debug_correspondence_after_shuffle() -> None:
    dataset = SyntheticBallTrackingDataset(_config(), split="train")
    sample = dataset[0]
    candidate_mask = sample["ball_candidate_mask"]
    debug_index = sample["candidate_gt_index"]
    valid_debug = debug_index[candidate_mask]
    assert (valid_debug >= -1).all()
    assert (valid_debug == -1).any()
    assert (valid_debug >= 0).any()
    assert (valid_debug < sample["target_ball_mask"].sum()).all()

    signatures = []
    for view in range(3):
        for frame in range(10):
            indices = debug_index[view, frame][candidate_mask[view, frame]]
            true_indices = tuple(indices[indices >= 0].tolist())
            if len(true_indices) >= 2:
                signatures.append(true_indices)
    assert len(set(signatures)) > 1


def test_synthetic_padding_birth_death_and_determinism() -> None:
    config = _config()
    config.min_frames = 6
    config.max_frames = 10
    config.min_views = 1
    config.max_views = 3
    dataset = SyntheticBallTrackingDataset(config, split="train")
    first = dataset[1]
    repeated = dataset[1]
    assert first["scene_format_version"].item() == 2
    first_tensors = dict(first)
    repeated_tensors = dict(repeated)
    for key, value in first_tensors.items():
        assert torch.equal(value, repeated_tensors[key])
    assert first["frame_mask"].shape == (10,)
    assert first["view_mask"].shape == (3,)
    invalid_frames = ~first["frame_mask"]
    assert not first["ball_candidate_mask"][:, invalid_frames].any()
    present = first["ball_present"][:, first["target_ball_mask"]]
    assert present.any()
    assert (~present).any()
