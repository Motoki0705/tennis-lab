"""Augmentation randomness of the player detection training dataset."""

from __future__ import annotations

import torch

from src.tasks.player_detection.configuration import (
    AugmentationConfig,
    FrameSelectionConfig,
    InputSizeConfig,
)
from src.tasks.player_detection.data.detection_dataset import (
    PlayerDetectionDataset,
    select_detection_frames,
)
from src.tasks.player_detection.data.store import PlayerFrameStore
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from tests.support.tasks.player_detection.chat_root import SyntheticRoot

_AUGMENTATION = AugmentationConfig(
    hflip_prob=0.5,
    short_side_choices=(32, 48, 64),
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
)


def _dataset(synthetic_root: SyntheticRoot, augmentation: AugmentationConfig | None) -> PlayerDetectionDataset:
    store = PlayerFrameStore(build_dataset(synthetic_root.config))
    selection = select_detection_frames(
        store,
        "train",
        FrameSelectionConfig(require_reviewed=True, require_located_players=True, min_visible_box_px=4.0),
        frame_stride=1,
    )
    return PlayerDetectionDataset(
        store,
        selection,
        input_size=InputSizeConfig(short_side=64, max_long_side=96),
        augmentation=augmentation,
    )


def test_repeated_draws_of_a_frame_get_new_augmentation(synthetic_root: SyntheticRoot) -> None:
    # A frame seen again in a later epoch (same index, same persistent worker)
    # must not replay the first epoch's flip/scale/color jitter.
    dataset = _dataset(synthetic_root, _AUGMENTATION)
    torch.manual_seed(0)
    draws = [dataset[0].image for _ in range(4)]
    assert any(
        draws[0].shape != other.shape or not torch.equal(draws[0], other) for other in draws[1:]
    )


def test_augmentation_is_reproducible_under_one_seed(synthetic_root: SyntheticRoot) -> None:
    dataset = _dataset(synthetic_root, _AUGMENTATION)
    torch.manual_seed(7)
    first = [dataset[0].image for _ in range(3)]
    torch.manual_seed(7)
    second = [dataset[0].image for _ in range(3)]
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_evaluation_samples_are_deterministic(synthetic_root: SyntheticRoot) -> None:
    dataset = _dataset(synthetic_root, None)
    assert torch.equal(dataset[0].image, dataset[0].image)
