"""Tests for deterministic procedural ground-court samples."""

import math
from typing import cast

import pytest
import torch
from torch.utils.data import DataLoader

from src.tasks.court_alignment.data.dataset import (
    GroundCourtDataset,
    GroundCourtDatasetConfig,
    build_ground_court_datasets,
)
from src.tasks.court_alignment.data.splits import GroundCourtSplitConfig
from src.tasks.court_alignment.geometry.court import (
    doubles_footprints_overlap,
)
from src.tasks.court_alignment.geometry.rasterization import render_keypoint_heatmaps


def _config() -> GroundCourtDatasetConfig:
    return GroundCourtDatasetConfig(
        image_size=(48, 64),
        max_courts=3,
        min_courts=2,
        split=GroundCourtSplitConfig(train_size=3, val_size=2, test_size=1, seed=17),
        sigma_px=0.75,
        scale_px_per_metre_range=(1.0, 1.0),
    )


def test_sample_is_deterministic_and_has_collatable_contract() -> None:
    first = GroundCourtDataset(_config(), split="train")[1]
    second = GroundCourtDataset(_config(), split="train")[1]
    for key in (
        "image",
        "target_heatmaps",
        "target_center_votes",
        "target_center_vote_mask",
        "keypoints",
        "visibility",
        "centers",
        "num_courts",
        "instance_ids",
    ):
        torch.testing.assert_close(first[key], second[key])
    assert first["sample_id"] == second["sample_id"] == "train-00000001"
    batch = next(iter(DataLoader(GroundCourtDataset(_config()), batch_size=2)))
    assert batch["image"].shape == (2, 1, 48, 64)
    assert batch["keypoints"].shape == (2, 3, 14, 2)
    assert batch["instance_ids"].shape == (2, 3)


def test_splits_use_stable_disjoint_seeds() -> None:
    datasets = build_ground_court_datasets(_config())
    assert tuple(datasets) == ("train", "val", "test")
    assert datasets["train"][0]["sample_id"] != datasets["val"][0]["sample_id"]
    assert not torch.equal(datasets["train"][0]["image"], datasets["val"][0]["image"])


def test_non_integer_keypoint_gets_exact_positive_at_nearest_lattice_pixel() -> None:
    points = torch.full((1, 14, 2), 10.4, dtype=torch.float32)
    visibility = torch.ones((1, 14), dtype=torch.bool)
    heatmaps = render_keypoint_heatmaps(
        (24, 24), points, visibility, sigma_px=0.75
    )
    assert float(heatmaps[0, 10, 10]) == 1.0


def test_rotation_range_rejects_semantically_ambiguous_full_circle() -> None:
    with pytest.raises(ValueError, match="at most pi"):
        GroundCourtDatasetConfig(rotation_rad_range=(0.0, 2.0 * math.pi))


def test_rotation_range_uses_configured_seam_margin_and_nonzero_span() -> None:
    default = GroundCourtDatasetConfig()
    assert default.rotation_rad_range == (0.05, math.pi - 0.05)
    with pytest.raises(ValueError, match="seam bounds"):
        GroundCourtDatasetConfig(rotation_rad_range=(0.049, math.pi - 0.05))
    with pytest.raises(ValueError, match="span"):
        GroundCourtDatasetConfig(rotation_rad_range=(0.5, 0.5))


def test_center_rejection_sampling_has_bounded_explicit_failure() -> None:
    config = GroundCourtDatasetConfig(
        image_size=(32, 32),
        max_courts=2,
        min_courts=2,
        min_center_distance_px=100.0,
        max_sampling_attempts=2,
        split=GroundCourtSplitConfig(train_size=1, val_size=0, test_size=0),
    )
    with pytest.raises(ValueError, match="Unable to sample"):
        _ = GroundCourtDataset(config)[0]


def test_vote_mask_is_identical_across_sigma_ablation() -> None:
    samples = []
    for sigma in (0.5, 1.0, 2.0):
        config = GroundCourtDatasetConfig(
            image_size=(64, 64),
            max_courts=1,
            min_courts=1,
            sigma_px=sigma,
            split=GroundCourtSplitConfig(train_size=1, val_size=0, test_size=0),
        )
        samples.append(GroundCourtDataset(config)[0])
    for sample in samples[1:]:
        assert torch.equal(
            sample["target_center_vote_mask"], samples[0]["target_center_vote_mask"]
        )


def test_sampling_rejects_overlapping_footprints_with_bounded_retry() -> None:
    config = GroundCourtDatasetConfig(
        image_size=(32, 32),
        max_courts=2,
        min_courts=2,
        min_center_distance_px=0.0,
        scale_px_per_metre_range=(10.0, 10.0),
        max_sampling_attempts=2,
        split=GroundCourtSplitConfig(train_size=1, val_size=0, test_size=0),
    )
    with pytest.raises(ValueError, match="non-overlapping doubles footprints"):
        _ = GroundCourtDataset(config)[0]


def test_generated_multiple_courts_have_non_overlapping_footprints() -> None:
    config = GroundCourtDatasetConfig(
        image_size=(128, 128),
        max_courts=2,
        min_courts=2,
        scale_px_per_metre_range=(1.0, 1.0),
        split=GroundCourtSplitConfig(train_size=8, val_size=0, test_size=0),
    )
    dataset = GroundCourtDataset(config)
    for index in range(len(dataset)):
        sample = dataset[index]
        points = cast(torch.Tensor, sample["keypoints"])[:2]
        footprints = [
            points[item].index_select(0, points.new_tensor((0, 1, 3, 2), dtype=torch.long))
            for item in range(2)
        ]
        assert not doubles_footprints_overlap(footprints[0], footprints[1])
