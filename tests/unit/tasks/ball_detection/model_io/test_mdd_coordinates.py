"""Decoded RGB validation without allocating a detector."""

import pytest
import torch

from src.tasks.ball_detection.model_io.mdd_coordinates import validate_rgb_input


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        pytest.param((1, 31, 3, 8, 8), torch.uint8, id="frames"),
        pytest.param((1, 32, 3, 8, 8), torch.float32, id="dtype"),
        pytest.param((1, 32, 5, 8, 8), torch.uint8, id="channels"),
        pytest.param((1, 5, 32, 8, 8), torch.float32, id="layout"),
        pytest.param((1, 32, 3, 7, 8), torch.uint8, id="height"),
        pytest.param((0, 32, 3, 8, 8), torch.uint8, id="empty"),
        pytest.param((1, 32, 3, 8), torch.uint8, id="rank"),
        pytest.param((1, 32, 3, 8, 7), torch.uint8, id="width"),
    ],
)
def test_rgb_input_rejects_invalid_contract(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    rgb = torch.zeros(shape, dtype=dtype)

    with pytest.raises(ValueError, match="RGB"):
        validate_rgb_input(rgb, frames=32)


def test_rgb_input_accepts_minimum_spatial_size() -> None:
    rgb = torch.zeros(1, 32, 3, 8, 8, dtype=torch.uint8)

    validate_rgb_input(rgb, frames=32)
