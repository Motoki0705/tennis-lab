"""Unit coverage for source-neutral Court target decoding."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.tasks.court_detection.data.contracts import (
    CourtInputCapability,
    CourtInputSpec,
)
from src.tasks.court_detection.data.processing.targets import (
    SegmentationTargetBuilder,
)

pytestmark = pytest.mark.unit


def test_segmentation_decode_copies_read_only_image_buffer() -> None:
    builder = SegmentationTargetBuilder(
        target_schema="court_cell_segmentation_v1",
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
