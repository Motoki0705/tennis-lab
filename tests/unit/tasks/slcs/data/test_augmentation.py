from dataclasses import replace

import torch

from src.tasks.slcs.data.augmentation import (
    ObservationAugmentationConfig,
    augment_observations,
)
from src.tasks.slcs.data.dataset import SLCSWindowDataset


def _config(*, rgb_only=0.0, joint_dropout=0.0, rgb_dropout=0.0):
    return ObservationAugmentationConfig(
        True, joint_dropout, 0.0, 0.0, 0.0, 4, rgb_only, rgb_dropout
    )


def _dataset(index, split, config, *, augment=None):
    return SLCSWindowDataset(
        dataset_root=index.root,
        split_file=split,
        split="train",
        config=config,
        stride=config.train_stride,
        augment=augment,
    )


def test_rgb_only_preserves_targets_rgb_and_source_cache(
    synthetic_dataset, synthetic_split_file, data_config
):
    dataset = _dataset(synthetic_dataset, synthetic_split_file, data_config)
    sample = dataset[0]
    source_pose = sample["player_kp"].clone()
    result = augment_observations(sample, _config(rgb_only=1.0))
    assert not result["player_valid"].any()
    assert not result["ball_vis"].any()
    assert not result["court_vis"].any()
    assert not result["dino_padding_mask"].all()
    result_values = dict(result)
    for key in sample:
        if key.startswith("target_"):
            torch.testing.assert_close(result_values[key], sample[key])
    torch.testing.assert_close(sample["player_kp"], source_pose)
    torch.testing.assert_close(dataset[0]["player_kp"], source_pose)


def test_dropout_visibility_and_coordinates_agree(
    synthetic_dataset, synthetic_split_file, data_config
):
    sample = _dataset(synthetic_dataset, synthetic_split_file, data_config)[0]
    result = augment_observations(sample, _config(joint_dropout=1.0, rgb_dropout=1.0))
    assert not result["player_kp"].any() and not result["player_valid"].any()
    assert result["dino_padding_mask"].all() and not result["dino_tokens"].any()
    torch.testing.assert_close(
        result["target_player_valid"], sample["target_player_valid"]
    )


def test_overfit_evaluation_can_use_train_split_without_augmentation(
    synthetic_dataset, synthetic_split_file, data_config
):
    config = replace(data_config, augmentation=_config(rgb_only=1.0))
    training = _dataset(synthetic_dataset, synthetic_split_file, config)
    evaluation = _dataset(
        synthetic_dataset, synthetic_split_file, config, augment=False
    )
    assert not training[0]["player_valid"].any()
    assert evaluation[0]["player_valid"].any()
