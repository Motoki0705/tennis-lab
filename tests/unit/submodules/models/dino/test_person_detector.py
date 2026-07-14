"""Unit tests for DINO person-only decoding and checkpoint validation."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models.dino.person_detector import (
    _preprocess_frame,
    _validate_checkpoint_args,
    _validate_dino_repository,
    decode_person_detections,
)


def test_decode_person_detections_filters_sorts_and_scales() -> None:
    logits = torch.full((1, 3, 91), -10.0)
    logits[0, :, 1] = torch.tensor([0.0, 2.0, 1.0])
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.4], [0.25, 0.5, 0.2, 0.2], [0.8, 0.5, 0.1, 0.2]]]
    )

    result = decode_person_detections(
        {"pred_logits": logits, "pred_boxes": boxes},
        image_width=200,
        image_height=100,
        confidence=0.6,
    )

    np.testing.assert_allclose(
        result.boxes_xyxy,
        np.array([[30, 40, 70, 60], [150, 40, 170, 60]], dtype=np.float32),
    )
    assert result.scores[0] > result.scores[1] > 0.6


def test_decode_person_detections_returns_shaped_empty_arrays() -> None:
    result = decode_person_detections(
        {
            "pred_logits": torch.full((1, 2, 91), -10.0),
            "pred_boxes": torch.zeros((1, 2, 4)),
        },
        image_width=200,
        image_height=100,
        confidence=0.3,
    )
    assert result.boxes_xyxy.shape == (0, 4)
    assert result.scores.shape == (0,)


def test_preprocess_frame_matches_official_resize_constraint() -> None:
    frame: NDArray[np.uint8] = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tensor = _preprocess_frame(frame, short_side=800, max_long_side=1333)
    assert tensor.shape == (3, 750, 1333)
    assert tensor.dtype == torch.float32


def test_checkpoint_architecture_mismatch_is_explicit() -> None:
    args = Namespace(backbone="swin_T_224_1k")
    with pytest.raises(ValueError, match="Unsupported DINO checkpoint architecture"):
        _validate_checkpoint_args(args)


def test_uninitialized_dino_submodule_is_explicit(tmp_path: Path) -> None:
    repository = tmp_path / "DINO"

    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        _validate_dino_repository(repository)
