"""Tests for the extensible typed augmentation registry."""

import pytest

from src.tasks.court_alignment.data.augmentation import (
    GroundCourtAugmentationConfig,
    build_augmentation,
)
from src.tasks.court_alignment.data.datamodule import GroundCourtDataModule


def test_identity_is_explicit_baseline() -> None:
    augmentation = build_augmentation(GroundCourtAugmentationConfig(name="identity"))
    assert augmentation.__class__.__name__ == "IdentityAugmentation"


def test_unknown_augmentation_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown ground-court augmentation"):
        build_augmentation("does-not-exist")


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown augmentation config fields"):
        GroundCourtAugmentationConfig.from_mapping({"unknown": "identity"})


def test_datamodule_resolves_unknown_transform_during_construction() -> None:
    with pytest.raises(ValueError, match="Unknown ground-court augmentation"):
        GroundCourtDataModule(
            train_samples=1,
            val_samples=0,
            test_samples=0,
            augmentations=[{"type": "does-not-exist"}],
        )
