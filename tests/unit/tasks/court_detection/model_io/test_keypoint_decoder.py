"""Unit tests for the explicit Court keypoint decode contract."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from src.tasks.court_detection.model_io.contracts import (
    CourtKeypointPrediction,
    CourtModelIOError,
)
from src.tasks.court_detection.model_io.keypoint_decoder import (
    CourtKeypointDecoderConfig,
    decode_court_keypoint_logits,
)


def _runtime_config(**kwargs: object) -> Any:
    """Build a config through ``Any`` so runtime type violations stay testable."""
    config_cls = cast(Any, CourtKeypointDecoderConfig)
    return config_cls(**kwargs)


def _decode(
    logits: object,
    *,
    original_size_hw: object = (8, 8),
    subpixel_refine: bool = False,
    config: CourtKeypointDecoderConfig | None = None,
) -> CourtKeypointPrediction:
    """Call the decoder with untyped inputs to exercise runtime validation."""
    decoder = cast(Any, decode_court_keypoint_logits)
    return cast(
        CourtKeypointPrediction,
        decoder(
            logits,
            original_size_hw=original_size_hw,
            subpixel_refine=subpixel_refine,
            config=CourtKeypointDecoderConfig() if config is None else config,
        ),
    )


def test_default_config_matches_the_singleton_court_contract() -> None:
    config = CourtKeypointDecoderConfig()
    assert config.threshold == 0.05
    assert config.nms_kernel == 7
    assert config.max_peaks == 1
    assert CourtKeypointDecoderConfig.__slots__ == (
        "threshold",
        "nms_kernel",
        "max_peaks",
    )
    assert not hasattr(config, "__dict__")


def test_explicit_valid_config_values_are_preserved() -> None:
    config = CourtKeypointDecoderConfig(threshold=0.0, nms_kernel=1, max_peaks=2)
    assert (config.threshold, config.nms_kernel, config.max_peaks) == (0.0, 1, 2)
    assert (
        CourtKeypointDecoderConfig(threshold=1.0, nms_kernel=9, max_peaks=4).max_peaks
        == 4
    )


def test_config_is_frozen() -> None:
    config = CourtKeypointDecoderConfig()
    attribute = "max_peaks"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, attribute, 2)
    attribute = "threshold"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, attribute, 0.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", float("nan")),
        ("threshold", float("inf")),
        ("threshold", -0.1),
        ("threshold", 1.5),
        ("nms_kernel", 0),
        ("nms_kernel", -3),
        ("nms_kernel", 2),
        ("max_peaks", 0),
        ("max_peaks", -1),
    ],
)
def test_out_of_range_config_values_raise_value_error(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _runtime_config(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", True),
        ("threshold", "0.05"),
        ("threshold", None),
        ("nms_kernel", 3.0),
        ("nms_kernel", True),
        ("nms_kernel", "7"),
        ("max_peaks", 1.0),
        ("max_peaks", False),
        ("max_peaks", "1"),
    ],
)
def test_wrongly_typed_config_values_raise_type_error(field: str, value: object) -> None:
    # A coerced value would silently change the decode contract, so the config
    # rejects float 3.0, bool, and string spellings instead of accepting them.
    with pytest.raises(TypeError):
        _runtime_config(**{field: value})


def test_decode_rejects_non_floating_logits() -> None:
    with pytest.raises(CourtModelIOError, match="floating point"):
        _decode(torch.zeros(1, 1, 8, 8, dtype=torch.long))


def test_decode_rejects_non_finite_logits() -> None:
    logits = torch.zeros(1, 1, 8, 8)
    logits[0, 0, 2, 3] = float("nan")
    with pytest.raises(CourtModelIOError, match="finite"):
        _decode(logits)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(1, 0, 8, 8),
        torch.zeros(1, 1, 0, 8),
        torch.zeros(1, 1, 8, 0),
    ],
)
def test_decode_rejects_empty_channel_or_spatial_axis(logits: Tensor) -> None:
    with pytest.raises(CourtModelIOError, match="positive channel and spatial"):
        _decode(logits)


def test_decode_rejects_non_tensor_input() -> None:
    with pytest.raises(CourtModelIOError, match="must be a Tensor"):
        _decode([[0.0]])


@pytest.mark.parametrize("shape", [(2, 1, 8, 8), (1, 8, 8)])
def test_decode_rejects_logits_without_a_singleton_batch(shape: tuple[int, ...]) -> None:
    with pytest.raises(CourtModelIOError, match=r"\(1,C,H,W\)"):
        _decode(torch.zeros(*shape))


@pytest.mark.parametrize(
    "original_size_hw",
    [0, 8, (8,), (8, 8, 8), (8, 8.0), ("8", 8), [8, 8], None],
)
def test_decode_rejects_malformed_original_size(original_size_hw: object) -> None:
    # Bad shape, non-int values, and a plain list must not surface as an
    # incidental unpacking TypeError.
    with pytest.raises(CourtModelIOError):
        _decode(torch.zeros(1, 1, 8, 8), original_size_hw=original_size_hw)


@pytest.mark.parametrize("original_size_hw", [(0, 8), (8, 0), (-8, 8)])
def test_decode_rejects_non_positive_original_size(
    original_size_hw: tuple[int, int],
) -> None:
    with pytest.raises(CourtModelIOError, match="original image size"):
        _decode(torch.zeros(1, 1, 8, 8), original_size_hw=original_size_hw)


def test_decode_scales_normalized_peaks_to_original_pixels() -> None:
    logits = torch.full((1, 1, 5, 6), -20.0)
    logits[0, 0, 2, 3] = 10.0

    prediction = _decode(logits, original_size_hw=(5, 6))

    assert isinstance(prediction, CourtKeypointPrediction)
    assert prediction.keypoints.shape == (1, 1, 2)
    torch.testing.assert_close(prediction.keypoints[:, 0], torch.tensor([[3.0, 2.0]]))


def test_decode_respects_the_configured_peak_threshold() -> None:
    logits = torch.full((1, 1, 16, 16), -20.0)
    logits[0, 0, 4, 4] = torch.logit(torch.tensor(0.3))
    logits[0, 0, 12, 12] = torch.logit(torch.tensor(0.06))

    strict = _decode(
        logits,
        original_size_hw=(16, 16),
        config=CourtKeypointDecoderConfig(threshold=0.2),
    )

    assert strict.valid.tolist() == [[True]]
    torch.testing.assert_close(strict.keypoints[:, 0], torch.tensor([[4.0, 4.0]]))


def test_decode_keeps_extra_peaks_only_with_an_explicit_budget() -> None:
    logits = torch.full((1, 1, 16, 16), -20.0)
    logits[0, 0, 3, 3] = torch.logit(torch.tensor(0.9))
    logits[0, 0, 12, 12] = torch.logit(torch.tensor(0.7))

    default = _decode(logits, original_size_hw=(16, 16))
    multi = _decode(
        logits,
        original_size_hw=(16, 16),
        config=CourtKeypointDecoderConfig(max_peaks=2),
    )

    assert default.keypoints.shape == (1, 1, 2)
    assert default.valid.tolist() == [[True]]
    assert multi.keypoints.shape == (1, 2, 2)
    assert multi.valid.tolist() == [[True, True]]
    torch.testing.assert_close(multi.keypoints[0, 1], torch.tensor([12.0, 12.0]))
    assert float(multi.scores[0, 0]) > float(multi.scores[0, 1])
