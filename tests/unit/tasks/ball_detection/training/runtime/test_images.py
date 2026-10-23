"""Variable-T JPEG grouping at the decoder boundary; no CUDA work is launched."""

from types import SimpleNamespace

import pytest
import torch

from src.tasks.ball_detection.training.runtime.images import (
    decode_coordinate_jpegs,
    image_input,
)


@pytest.mark.parametrize("length", [32, 64, 128])
def test_rgb_input_keeps_shape_dtype_and_values(length: int) -> None:
    rgb = torch.arange(2 * length * 3 * 4 * 4).to(torch.uint8).reshape(2, length, 3, 4, 4)
    actual = image_input(dict(rgb=rgb), torch.device("cpu"))
    assert actual is rgb and torch.equal(actual, rgb)


@pytest.mark.parametrize("length", [32, 64, 128])
def test_nvjpeg_groups_in_batch_then_frame_order(length: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.io

    lengths = tuple(1 for _ in range(2 * length))
    packed = torch.arange(2 * length).to(torch.uint8)

    def decode(payloads: list[torch.Tensor], **kwargs: object) -> list[torch.Tensor]:
        assert kwargs["device"] == torch.device("cuda")
        return [torch.full((3, 4, 4), int(payload[0]), dtype=torch.uint8) for payload in payloads]

    monkeypatch.setattr(torchvision.io, "decode_jpeg", decode)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None))
    batch = dict(jpeg=packed, jpeg_lengths=list(lengths), image_shape=[2, length, 3, 4, 4])
    actual = decode_coordinate_jpegs(batch, torch.device("cuda"))
    assert actual.shape == (2, length, 3, 4, 4)
    assert torch.equal(actual[:, :, 0, 0, 0].flatten(), packed)
    with pytest.raises(ValueError, match="packed JPEGs"):
        decode_coordinate_jpegs(dict(batch, jpeg_lengths=lengths[:-1]), torch.device("cuda"))


def test_invalid_channels_or_single_frame_cannot_silently_pass() -> None:
    for shape in ((1, 64, 2, 4, 4), (1, 1, 3, 4, 4)):
        with pytest.raises(ValueError, match="RGB uint8"):
            image_input(dict(rgb=torch.zeros(shape, dtype=torch.uint8)), torch.device("cpu"))
    with pytest.raises(ValueError, match="no CPU fallback"):
        decode_coordinate_jpegs({}, torch.device("cpu"))
