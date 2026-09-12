"""Unit coverage for source-neutral Court target decoding."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from src.tasks.court_detection.data.contracts import (
    CourtInputCapability,
    CourtInputSpec,
    CourtSampleMetadata,
    CourtTransformedSample,
)
from src.tasks.court_detection.data.processing.targets import (
    SegmentationTargetBuilder,
    SemanticLineTargetBuilder,
)
from src.tasks.court_detection.target_schemas import (
    SEMANTIC_LINE_CLASS_BY_NAME,
    SEMANTIC_LINE_TARGET_SCHEMA,
)

pytestmark = pytest.mark.unit


def test_segmentation_decode_copies_read_only_image_buffer() -> None:
    builder = SegmentationTargetBuilder(
        target_schema="court_cell_segmentation_single_court_v2",
        input_spec=CourtInputSpec(
            source_kind="tennis_court_detector",
            source_schema="fixture",
            capabilities=frozenset({CourtInputCapability.SEGMENTATION_REFERENCE}),
        ),
    )
    array = np.frombuffer(bytes([0, 1, 2, 3]), dtype=np.uint8).reshape(2, 2)
    assert not array.flags.writeable

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decoded = builder._decode(array)
        decoded[0, 0] = 6

    assert int(decoded[0, 0]) == 6


def test_semantic_line_builder_swaps_only_left_right_classes_on_flip() -> None:
    builder = SemanticLineTargetBuilder(
        target_schema=SEMANTIC_LINE_TARGET_SCHEMA,
        input_spec=CourtInputSpec(
            source_kind="synthetic_court",
            source_schema="fixture",
            capabilities=frozenset(
                {CourtInputCapability.SEMANTIC_LINE_REFERENCE}
            ),
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
    assert torch.equal(
        result,
        torch.tensor(
            [[
                SEMANTIC_LINE_CLASS_BY_NAME["right_doubles_sideline"],
                SEMANTIC_LINE_CLASS_BY_NAME["left_doubles_sideline"],
                SEMANTIC_LINE_CLASS_BY_NAME["right_singles_sideline"],
                SEMANTIC_LINE_CLASS_BY_NAME["left_singles_sideline"],
                SEMANTIC_LINE_CLASS_BY_NAME["far_baseline"],
            ]],
            dtype=torch.long,
        ),
    )
