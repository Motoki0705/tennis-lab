from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.base.data.court_lines import (
    CourtLineInputBuilder,
    CourtLineInputConfig,
    CourtLineMapAugmentationConfig,
)
from src.tasks.base.preview import (
    build_court_line_preview_rows,
    court_line_frame_metadata,
    enable_all_augmentation_blocks,
    make_court_kp_preview_config,
    make_court_line_preview_builder,
    resolve_court_input_type,
    resolve_sample_indices,
    resolve_split_file,
)
from src.utils.geometry.line_segments import RansacLineConfig


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


def test_preview_court_input_type_is_strict() -> None:
    assert resolve_court_input_type(
        OmegaConf.create({"preview": {"court_input_type": "line"}})
    ) == "line"
    with pytest.raises(ValueError, match="must be 'kp' or 'line'"):
        resolve_court_input_type(
            OmegaConf.create({"preview": {"court_input_type": "both"}})
        )


def test_line_preview_forces_only_the_dataset_copy_to_kp() -> None:
    cfg = OmegaConf.create(
        {
            "data": {"court_input_type": "line", "scene_dir": "data/example"},
            "preview": {"court_input_type": "line", "court_line": {}},
        }
    )
    cloned = make_court_kp_preview_config(cfg)

    assert cloned.data.court_input_type == "kp"
    assert cloned.data.scene_dir == "data/example"
    assert cfg.data.court_input_type == "line"
    assert isinstance(make_court_line_preview_builder(cfg), CourtLineInputBuilder)


def test_build_court_line_preview_rows_is_seeded_and_reports_diagnostics() -> None:
    court = torch.zeros(2, 3, 20, 2)
    court[:, :, 0] = torch.tensor([0.1, 0.2])
    court[:, :, 1] = torch.tensor([0.9, 0.2])
    court[:, :, 2] = torch.tensor([0.1, 0.8])
    court[:, :, 3] = torch.tensor([0.9, 0.8])
    builder = CourtLineInputBuilder(
        CourtLineInputConfig(
            map_width=80,
            map_height=48,
            extractor=RansacLineConfig(
                max_iterations=40,
                distance_threshold_px=1.5,
                min_inliers=5,
                min_segment_length_px=3.0,
                max_lines=4,
                skeletonize=False,
                min_component_size=2,
                max_points=500,
            ),
            augmentation=CourtLineMapAugmentationConfig(),
        )
    )

    first = build_court_line_preview_rows(
        builder,
        court,
        original_seed=2,
        variant_seeds=[3, 4],
    )
    second = build_court_line_preview_rows(
        builder,
        court,
        original_seed=2,
        variant_seeds=[3, 4],
    )

    assert len(first) == 3
    assert all(len(row) == 2 for row in first)
    assert court_line_frame_metadata(first[0][0])["extracted_line_count"] > 0
    for first_row, second_row in zip(first, second, strict=True):
        for first_frame, second_frame in zip(first_row, second_row, strict=True):
            assert torch.equal(
                torch.from_numpy(first_frame.line_map),
                torch.from_numpy(second_frame.line_map),
            )
            assert torch.equal(
                torch.from_numpy(first_frame.extraction.segments),
                torch.from_numpy(second_frame.extraction.segments),
            )
