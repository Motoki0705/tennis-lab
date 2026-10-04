"""Unit coverage for source-neutral Court target construction."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_detection.data.contracts import (
    CourtInputCapability,
    CourtInputSpec,
    CourtSampleMetadata,
    CourtTransformedSample,
)
from src.tasks.court_detection.data.processing.targets import (
    SemanticLineTargetBuilder,
)
from src.tasks.court_detection.target_schemas import (
    SEMANTIC_LINE_CLASS_BY_NAME,
    SEMANTIC_LINE_TARGET_SCHEMA,
    SEMANTIC_LINE_TARGET_SCHEMA_HARD,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("soft", [False, True])
def test_semantic_line_builder_swaps_only_left_right_classes_on_flip(
    soft: bool,
) -> None:
    builder = SemanticLineTargetBuilder(
        target_schema=SEMANTIC_LINE_TARGET_SCHEMA
        if soft
        else SEMANTIC_LINE_TARGET_SCHEMA_HARD,
        input_spec=CourtInputSpec(
            source_kind="synthetic_court",
            source_schema="fixture",
            capabilities=frozenset({CourtInputCapability.COURT_INSTANCES}),
        ),
    )
    names = (
        "left_doubles_sideline",
        "right_doubles_sideline",
        "left_singles_sideline",
        "right_singles_sideline",
        "far_baseline",
    )
    mask = torch.tensor(
        [[SEMANTIC_LINE_CLASS_BY_NAME[name] for name in names]],
        dtype=torch.long,
    )
    if soft:
        mask = torch.nn.functional.one_hot(mask, 12).permute(2, 0, 1).float()
    sample = CourtTransformedSample(
        sample_id="fixture",
        image_tensor=torch.zeros(3, 1, 5),
        image_size=torch.tensor([1, 5], dtype=torch.long),
        keypoint_channels=None,
        court_instances=(),
        dense_targets={"semantic_line": mask},
        horizontal_flipped=True,
        metadata=CourtSampleMetadata(
            source_kind="synthetic_court",
            source_schema="fixture",
            source_sample_id="fixture",
            scene_id="B00",
            provenance={},
        ),
    )

    result = builder.build(sample)

    assert isinstance(result, torch.Tensor)
    if soft:
        torch.testing.assert_close(result.sum(0), torch.ones(1, 5))
        result = result.argmax(0)
    assert torch.equal(
        result,
        torch.tensor(
            [
                [
                    SEMANTIC_LINE_CLASS_BY_NAME["right_doubles_sideline"],
                    SEMANTIC_LINE_CLASS_BY_NAME["left_doubles_sideline"],
                    SEMANTIC_LINE_CLASS_BY_NAME["right_singles_sideline"],
                    SEMANTIC_LINE_CLASS_BY_NAME["left_singles_sideline"],
                    SEMANTIC_LINE_CLASS_BY_NAME["far_baseline"],
                ]
            ],
            dtype=torch.long,
        ),
    )
