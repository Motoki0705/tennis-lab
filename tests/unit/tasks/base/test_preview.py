from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.tasks.base.preview import (
    enable_all_augmentation_blocks,
    resolve_sample_indices,
    resolve_split_file,
)


def _preview_cfg(sample_indices: list[int], max_samples: int) -> OmegaConf:
    return OmegaConf.create(
        {"preview": {"sample_indices": sample_indices, "max_samples": max_samples}}
    )


def test_explicit_sample_indices_take_precedence() -> None:
    cfg = _preview_cfg([2, 5], max_samples=10)
    assert resolve_sample_indices(cfg, 8) == [2, 5]


def test_max_samples_clamped_to_dataset_size() -> None:
    cfg = _preview_cfg([], max_samples=100)
    assert resolve_sample_indices(cfg, 3) == [0, 1, 2]


def test_min_samples_floor_matches_both_legacy_variants() -> None:
    cfg = _preview_cfg([], max_samples=0)
    # preview_heatmaps behaviour: no floor -> empty selection.
    assert resolve_sample_indices(cfg, 5) == []
    # preview_augmentation behaviour: floor at 1 -> one sample.
    assert resolve_sample_indices(cfg, 5, min_samples=1) == [0]


def test_out_of_range_explicit_index_raises() -> None:
    cfg = _preview_cfg([9], max_samples=10)
    with pytest.raises(IndexError):
        resolve_sample_indices(cfg, 5)


def test_enable_all_augmentation_blocks_flips_flags_only() -> None:
    cfg = OmegaConf.create(
        {
            "enabled": False,
            "preserve_clean_targets": True,
            "gaussian_noise": {"enabled": False, "prob": 0.5, "ball_std": 0.01},
            "burst_dropout": {"enabled": True, "prob": 0.3},
            "scale_range": [1.0, 1.0],
        }
    )
    result = enable_all_augmentation_blocks(cfg)
    assert result["enabled"] is True
    assert result["gaussian_noise"]["enabled"] is True
    assert result["burst_dropout"]["enabled"] is True
    # Non-flag parameters are untouched.
    assert result["gaussian_noise"]["prob"] == 0.5
    assert result["gaussian_noise"]["ball_std"] == 0.01
    assert result["preserve_clean_targets"] is True
    assert result["scale_range"] == [1.0, 1.0]
    # The source config is not mutated.
    assert cfg.enabled is False
    assert cfg.gaussian_noise.enabled is False


def test_enable_all_augmentation_blocks_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        enable_all_augmentation_blocks(OmegaConf.create([1, 2]))


def test_resolve_split_file_looks_up_named_key() -> None:
    cfg = OmegaConf.create(
        {"data": {"split": {"train_file": "a.json", "val_file": "b.json"}}}
    )
    assert resolve_split_file(cfg, "val") == "b.json"
    with pytest.raises(ValueError, match="Unknown preview.split"):
        resolve_split_file(cfg, "test")
