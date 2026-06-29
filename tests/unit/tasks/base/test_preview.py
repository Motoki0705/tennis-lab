from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from src.tasks.base.preview import resolve_sample_indices, resolve_split_file


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


def test_resolve_split_file_looks_up_named_key() -> None:
    cfg = OmegaConf.create(
        {"data": {"split": {"train_file": "a.json", "val_file": "b.json"}}}
    )
    assert resolve_split_file(cfg, "val") == "b.json"
    with pytest.raises(ValueError, match="Unknown preview.split"):
        resolve_split_file(cfg, "test")
