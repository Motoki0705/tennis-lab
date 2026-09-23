"""Inference must reproduce the dataset's declared transform before RGB/MDD."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from src.tasks.ball_detection.model_io.contracts import BallModelIOError
from src.tasks.ball_detection.model_io.normalization import BallImageNormalization
from tests.unit.tasks.ball_detection.model_io.test_adapters import _rgb_adapter


@pytest.mark.parametrize("mode", ["rgb", "mdd"])
@pytest.mark.parametrize("layout", ["btchw", "bcthw"])
def test_raw_inference_and_dataset_preprocessed_training_have_same_model_input(
    mode: str, layout: str,
) -> None:
    adapter = _rgb_adapter()
    adapter.spec = replace(adapter.spec, input_mode=mode, input_layout=layout,
                           in_channels=3 if mode == "rgb" else 2,
                           mdd_gain=3.0, mdd_offset=0.1)
    raw = torch.linspace(0, 1, 1 * 3 * 3 * 2 * 2).reshape(1, 3, 3, 2, 2)
    normalization = BallImageNormalization(True, (0.1, 0.2, 0.3), (0.2, 0.4, 0.8))
    # An independent per-channel implementation of the dataset transform.
    prepared = torch.stack([(raw[:, :, c] - m) / s for c, (m, s) in
                            enumerate(zip((0.1, 0.2, 0.3), (0.2, 0.4, 0.8), strict=True))], dim=2)
    if mode == "rgb":
        expected = prepared
    else:
        gray = prepared[:, :, 0] * .299 + prepared[:, :, 1] * .587 + prepared[:, :, 2] * .114
        difference = gray[:, 1:] - gray[:, :-1]
        brighten, darken = torch.zeros_like(gray), torch.zeros_like(gray)
        brighten[:, 1:] = torch.sigmoid(3.0 * (difference.clamp(min=0) - .1))
        darken[:, 1:] = torch.sigmoid(3.0 * ((-difference).clamp(min=0) - .1))
        expected = torch.stack((brighten, darken), dim=2)
    if layout == "bcthw":
        expected = expected.permute(0, 2, 1, 3, 4)
    inference = adapter.prepare_images(raw, image_normalization=normalization)
    training = adapter.prepare_training_batch({
        "images": prepared, "heatmaps": torch.zeros(1, 3, 2, 2),
        "coords": torch.zeros(1, 3, 1, 2), "visibility": torch.ones(1, 3, 1, dtype=torch.bool),
        "original_size": torch.tensor([[2, 2]]),
    }, image_normalization=normalization)
    torch.testing.assert_close(inference.model_input, expected)
    torch.testing.assert_close(training.model_call.model_input, expected)
    assert not torch.allclose(adapter.prepare_images(raw).model_input, expected)
    if mode == "mdd":
        features = adapter.mdd_features(raw, image_normalization=normalization)
        torch.testing.assert_close(features, expected if layout == "bcthw" else expected.permute(0, 2, 1, 3, 4))


def test_raw_boundary_still_rejects_already_normalized_rgb() -> None:
    with pytest.raises(BallModelIOError, match=r"\[0, 1\]"):
        _rgb_adapter().prepare_images(torch.full((1, 2, 3, 2, 2), -1.0),
                                     image_normalization=BallImageNormalization(True))


def test_preprocessed_boundary_checks_declared_channel_range() -> None:
    with pytest.raises(BallModelIOError, match="normalization bounds"):
        _rgb_adapter().prepare_images(torch.full((1, 2, 3, 2, 2), 8.0),
                                     image_normalization=BallImageNormalization(True), preprocessed=True)


@pytest.mark.parametrize("block", [
    {}, {"enabled": "true"}, {"enabled": True},
    {"enabled": True, "mean": [0, 0, float("nan")], "std": [1, 1, 1]},
    {"enabled": True, "mean": [0, 0, 0], "std": [1, 0, 1]},
])
def test_missing_or_invalid_saved_normalization_is_an_error(block: dict) -> None:
    with pytest.raises(BallModelIOError):
        BallImageNormalization.from_config({"data": {"augmentation": {"normalize_imagenet": block}}})


def test_explicitly_disabled_normalization_is_identity() -> None:
    normalization = BallImageNormalization.from_config({"data": {"augmentation": {
        "normalize_imagenet": {"enabled": False}}}})
    images = torch.rand(1, 2, 3, 2, 2)
    assert normalization.apply(images) is images
