from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.base.data.court_lines import (
    CourtLineInputBuilder,
    CourtLineInputConfig,
    CourtLineMapAugmentationConfig,
    CourtLineMapBuilder,
    CourtLineMapConfig,
    augment_court_line_map,
    render_court_line_map,
)
from src.utils.geometry.line_segments import RansacLineConfig


def _projected_court() -> torch.Tensor:
    points = torch.zeros(20, 2)
    points[0] = torch.tensor([0.15, 0.15])
    points[1] = torch.tensor([0.85, 0.15])
    points[2] = torch.tensor([0.05, 0.9])
    points[3] = torch.tensor([0.95, 0.9])
    points[4] = torch.tensor([0.25, 0.15])
    points[5] = torch.tensor([0.18, 0.9])
    points[6] = torch.tensor([0.75, 0.15])
    points[7] = torch.tensor([0.82, 0.9])
    points[8] = torch.tensor([0.3, 0.4])
    points[9] = torch.tensor([0.7, 0.4])
    points[10] = torch.tensor([0.25, 0.7])
    points[11] = torch.tensor([0.75, 0.7])
    points[12] = torch.tensor([0.5, 0.4])
    points[13] = torch.tensor([0.5, 0.7])
    return points


def _builder() -> CourtLineInputBuilder:
    return CourtLineInputBuilder(
        CourtLineInputConfig(
            map_width=160,
            map_height=96,
            temporal_variants=2,
            extractor=RansacLineConfig(
                max_iterations=200,
                distance_threshold_px=1.5,
                min_inliers=8,
                min_segment_length_px=5.0,
                max_lines=8,
                skeletonize=False,
                min_component_size=3,
                max_points=2000,
            ),
            augmentation=CourtLineMapAugmentationConfig(enabled=False),
        )
    )


def test_render_and_build_court_lines() -> None:
    court = _projected_court()
    line_map = render_court_line_map(court.numpy(), width=160, height=96, line_width=1)
    assert line_map.shape == (96, 160)
    assert int(line_map.max()) > 0

    court_sequence = court.view(1, 1, 20, 2).expand(2, 4, -1, -1)
    result = _builder().build(
        court_sequence,
        augment=False,
        rng=np.random.default_rng(5),
    )

    assert result.shape == (2, 4, 8, 4)
    assert torch.isfinite(result).all()
    assert torch.any(result != 0)
    torch.testing.assert_close(result[:, 0], result[:, -1])


def test_build_line_maps_returns_normalized_channel_first_tensor() -> None:
    court = _projected_court().view(1, 1, 20, 2).expand(2, 4, -1, -1)
    builder = CourtLineMapBuilder(
        CourtLineMapConfig(
            map_width=80,
            map_height=48,
            augmentation=CourtLineMapAugmentationConfig(enabled=False),
        )
    )

    result = builder.build(court, augment=False, rng=np.random.default_rng(5))

    assert result.shape == (2, 4, 1, 48, 80)
    assert result.dtype == torch.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert torch.any(result > 0)
    torch.testing.assert_close(result[:, 0], result[:, -1])


def test_disabled_map_augmentation_is_seed_independent() -> None:
    court = _projected_court().view(1, 1, 20, 2).expand(1, 3, -1, -1)
    builder = CourtLineMapBuilder(
        CourtLineMapConfig(
            map_width=80,
            map_height=48,
            augmentation=CourtLineMapAugmentationConfig(enabled=False),
        )
    )

    first = builder.build(court, augment=True, rng=np.random.default_rng(1))
    second = builder.build(court, augment=True, rng=np.random.default_rng(999))

    torch.testing.assert_close(first, second)


def test_build_frame_exposes_the_training_map_and_diagnostics() -> None:
    builder = _builder()
    court = _projected_court()
    frame = builder.build_frame(
        court,
        augment=False,
        rng=np.random.default_rng(17),
    )
    sequence = builder.build(
        court.view(1, 1, 20, 2),
        augment=False,
        rng=np.random.default_rng(17),
    )

    assert frame.line_map.shape == (96, 160)
    assert frame.extraction.diagnostics.extracted_line_count > 0
    np.testing.assert_array_equal(
        frame.extraction.segments,
        sequence[0, 0].numpy(),
    )


def test_builder_is_reproducible_for_fixed_seed() -> None:
    court = _projected_court().view(1, 1, 20, 2).expand(1, 3, -1, -1)
    config = CourtLineInputConfig(
        map_width=160,
        map_height=96,
        temporal_variants=2,
        extractor=_builder().config.extractor,
        augmentation=CourtLineMapAugmentationConfig(),
    )
    first = CourtLineInputBuilder(config).build(
        court, augment=True, rng=np.random.default_rng(42)
    )
    second = CourtLineInputBuilder(config).build(
        court, augment=True, rng=np.random.default_rng(42)
    )
    torch.testing.assert_close(first, second)


def test_line_width_variation_changes_rendered_map() -> None:
    court = _projected_court().numpy()
    thin = render_court_line_map(court, width=160, height=96, line_width=1)
    thick = render_court_line_map(court, width=160, height=96, line_width=3)

    assert np.count_nonzero(thick) > np.count_nonzero(thin)


def test_erasure_and_occlusion_remove_line_evidence() -> None:
    line_map = render_court_line_map(
        _projected_court().numpy(), width=160, height=96, line_width=2
    )
    config = CourtLineMapAugmentationConfig(
        partial_erasure_prob=1.0,
        max_partial_erasures=5,
        occlusion_prob=1.0,
        max_occlusions=3,
        false_positive_prob=0.0,
        blur_prob=0.0,
        morphology_prob=0.0,
        far_dropout_prob=0.0,
        near_only_prob=0.0,
    )
    augmented = augment_court_line_map(
        line_map, config=config, rng=np.random.default_rng(7)
    )

    assert np.count_nonzero(augmented) < np.count_nonzero(line_map)


def test_false_positive_lines_are_added_to_empty_map() -> None:
    line_map: NDArray[np.uint8] = np.zeros((96, 160), dtype=np.uint8)
    config = CourtLineMapAugmentationConfig(
        partial_erasure_prob=0.0,
        occlusion_prob=0.0,
        false_positive_prob=1.0,
        max_false_positive_lines=3,
        blur_prob=0.0,
        morphology_prob=0.0,
        far_dropout_prob=0.0,
        near_only_prob=0.0,
    )
    augmented = augment_court_line_map(
        line_map, config=config, rng=np.random.default_rng(9)
    )

    assert np.count_nonzero(augmented) > 0


def test_near_only_removes_far_court_region() -> None:
    line_map = render_court_line_map(
        _projected_court().numpy(), width=160, height=96, line_width=1
    )
    config = CourtLineMapAugmentationConfig(
        partial_erasure_prob=0.0,
        occlusion_prob=0.0,
        false_positive_prob=0.0,
        blur_prob=0.0,
        morphology_prob=0.0,
        far_dropout_prob=0.0,
        near_only_prob=1.0,
    )
    augmented = augment_court_line_map(
        line_map, config=config, rng=np.random.default_rng(11)
    )

    assert not np.array_equal(augmented, line_map)
    assert np.count_nonzero(augmented[: int(0.45 * line_map.shape[0])]) == 0


def test_far_dropout_removes_top_court_region() -> None:
    line_map = render_court_line_map(
        _projected_court().numpy(), width=160, height=96, line_width=1
    )
    config = CourtLineMapAugmentationConfig(
        partial_erasure_prob=0.0,
        occlusion_prob=0.0,
        false_positive_prob=0.0,
        blur_prob=0.0,
        morphology_prob=0.0,
        far_dropout_prob=1.0,
        near_only_prob=0.0,
    )
    augmented = augment_court_line_map(
        line_map, config=config, rng=np.random.default_rng(13)
    )

    assert not np.array_equal(augmented, line_map)
    assert np.count_nonzero(augmented[: int(0.2 * line_map.shape[0])]) == 0


def test_blur_and_morphology_change_map() -> None:
    line_map = render_court_line_map(
        _projected_court().numpy(), width=160, height=96, line_width=1
    )
    config = CourtLineMapAugmentationConfig(
        partial_erasure_prob=0.0,
        occlusion_prob=0.0,
        false_positive_prob=0.0,
        blur_prob=1.0,
        morphology_prob=1.0,
        far_dropout_prob=0.0,
        near_only_prob=0.0,
    )
    augmented = augment_court_line_map(
        line_map, config=config, rng=np.random.default_rng(11)
    )

    assert not np.array_equal(augmented, line_map)
