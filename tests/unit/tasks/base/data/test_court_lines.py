from __future__ import annotations

import numpy as np
import torch

from src.tasks.base.data.court_lines import (
    CourtLineInputBuilder,
    CourtLineInputConfig,
    CourtLineMapAugmentationConfig,
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


def test_builder_is_reproducible_for_fixed_seed() -> None:
    court = _projected_court().view(1, 1, 20, 2).expand(1, 3, -1, -1)
    first = _builder().build(court, augment=True, rng=np.random.default_rng(42))
    second = _builder().build(court, augment=True, rng=np.random.default_rng(42))
    torch.testing.assert_close(first, second)
