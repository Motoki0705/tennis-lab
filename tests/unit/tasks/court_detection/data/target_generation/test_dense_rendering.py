"""Physical widths and class semantics of in-memory Court rasterization."""

from __future__ import annotations

import numpy as np
import torch

from src.tasks.court_detection.data.contracts import (
    CourtInstance2D,
)
from src.tasks.court_detection.data.target_generation.line import generate_line_target
from src.tasks.court_detection.data.target_generation.semantic_line import (
    generate_semantic_line_target,
)
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
    SEMANTIC_LINE_CLASS_BY_NAME,
    line_target_definition,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def test_line_target_width_is_explicitly_previewable() -> None:
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    image_points = torch.stack(
        (
            (points[:, 0] / 12.0 + 0.5) * 255.0,
            (0.5 - points[:, 1] / 26.0) * 255.0,
        ),
        dim=1,
    )
    instance = CourtInstance2D(
        court_instance_id="court",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=image_points,
        point_in_front=torch.ones(14, dtype=torch.bool),
        point_visible=torch.ones(14, dtype=torch.bool),
    )

    narrow = generate_line_target(
        height=256,
        width=256,
        instances=(instance,),
        line_width_metres=0.025,
        baseline_width_metres=0.05,
    )
    wide = generate_line_target(
        height=256,
        width=256,
        instances=(instance,),
        line_width_metres=0.075,
        baseline_width_metres=0.15,
    )

    assert np.count_nonzero(wide) > np.count_nonzero(narrow)


def test_semantic_line_target_matches_binary_coverage_and_camera_view_labels() -> None:
    points = court_keypoints_3d(STANDARD_COURT_CONFIG)[:14, :2]
    image_points = torch.stack(
        (
            (points[:, 0] / 12.0 + 0.5) * 255.0,
            (0.5 - points[:, 1] / 26.0) * 255.0,
        ),
        dim=1,
    )
    instance = CourtInstance2D(
        court_instance_id="court",
        physical_indices=torch.arange(14, dtype=torch.long),
        points_xy=image_points,
        point_in_front=torch.ones(14, dtype=torch.bool),
        point_visible=torch.ones(14, dtype=torch.bool),
    )
    binary = generate_line_target(height=256, width=256, instances=(instance,))
    identity = generate_semantic_line_target(
        height=256,
        width=256,
        instances=(instance,),
        semantic_to_physical=torch.arange(14, dtype=torch.long),
    )
    half_turn = generate_semantic_line_target(
        height=256,
        width=256,
        instances=(instance,),
        semantic_to_physical=torch.tensor(
            (3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 13, 12),
            dtype=torch.long,
        ),
    )

    np.testing.assert_array_equal(identity > 0, binary > 0)
    np.testing.assert_array_equal(half_turn > 0, binary > 0)
    far_baseline_midpoint = (64, 11)
    left_doubles_midpoint = (11, 128)
    assert (
        identity[far_baseline_midpoint[1], far_baseline_midpoint[0]]
        == (SEMANTIC_LINE_CLASS_BY_NAME["far_baseline"])
    )
    assert (
        half_turn[far_baseline_midpoint[1], far_baseline_midpoint[0]]
        == (SEMANTIC_LINE_CLASS_BY_NAME["near_baseline"])
    )
    assert (
        identity[left_doubles_midpoint[1], left_doubles_midpoint[0]]
        == (SEMANTIC_LINE_CLASS_BY_NAME["left_doubles_sideline"])
    )
    assert (
        half_turn[left_doubles_midpoint[1], left_doubles_midpoint[0]]
        == (SEMANTIC_LINE_CLASS_BY_NAME["right_doubles_sideline"])
    )


def test_line_target_schemas_keep_physical_widths_immutable() -> None:
    current = line_target_definition(LINE_TARGET_SCHEMA)
    assert (current.line_width_metres, current.baseline_width_metres) == (0.075, 0.15)
    import pytest

    for schema in ("court_line_binary_v1", "court_line_binary_75mm_150mm_v2"):
        with pytest.raises(ValueError, match="Unsupported Court line target schema"):
            line_target_definition(schema)
