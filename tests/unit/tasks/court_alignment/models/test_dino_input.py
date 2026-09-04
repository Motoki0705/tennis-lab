"""Unit tests for one-channel DINO input ablations."""

from __future__ import annotations

import pytest
import torch

from src.tasks.court_alignment.models.dino_input import (
    DINO_DEFAULT_MAX_LONG_SIDE,
    DINO_DEFAULT_SHORT_SIDE,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    DinoHeatmapInputAdapter,
    dino_resize_shape,
)


def _normalise(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_RGB_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_RGB_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


@pytest.mark.parametrize("mode", ["repeat_rgb", "learnable_1x1"])
def test_repeat_and_initial_learnable_modes_are_exact_rgb_copies(mode: str) -> None:
    heatmap = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])
    adapter = DinoHeatmapInputAdapter(
        mode=mode,  # type: ignore[arg-type]
        short_side=2,
        max_long_side=2,
    )

    output = adapter(heatmap)

    expected_rgb = heatmap.repeat(1, 3, 1, 1)
    torch.testing.assert_close(output, _normalise(expected_rgb))
    if mode == "learnable_1x1":
        assert adapter.projection is not None
        torch.testing.assert_close(adapter.projection.weight, torch.ones(3, 1, 1, 1))
        torch.testing.assert_close(adapter.projection.bias, torch.zeros(3))


def test_red_only_zeros_green_and_blue_before_imagenet_normalisation() -> None:
    heatmap = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])
    adapter = DinoHeatmapInputAdapter(
        mode="red_only",
        short_side=2,
        max_long_side=2,
    )

    output = adapter(heatmap)

    expected_rgb = torch.cat((heatmap, torch.zeros(1, 2, 2, 2)), dim=1)
    torch.testing.assert_close(output, _normalise(expected_rgb))
    assert adapter.projection is None


def test_resize_preserves_aspect_ratio_and_obeys_max_long_side() -> None:
    assert (DINO_DEFAULT_SHORT_SIDE, DINO_DEFAULT_MAX_LONG_SIDE) == (800, 1333)
    assert dino_resize_shape(
        256,
        256,
        short_side=DINO_DEFAULT_SHORT_SIDE,
        max_long_side=DINO_DEFAULT_MAX_LONG_SIDE,
    ) == (800, 800)
    assert dino_resize_shape(4, 8, short_side=6, max_long_side=10) == (5, 10)
    adapter = DinoHeatmapInputAdapter(
        mode="repeat_rgb",
        short_side=6,
        max_long_side=10,
    )

    output = adapter(torch.full((1, 1, 4, 8), 0.5))

    assert output.shape == (1, 3, 5, 10)


def test_input_modes_have_only_the_intended_trainable_parameters() -> None:
    repeat = DinoHeatmapInputAdapter(mode="repeat_rgb", short_side=2, max_long_side=2)
    red_only = DinoHeatmapInputAdapter(mode="red_only", short_side=2, max_long_side=2)
    learnable = DinoHeatmapInputAdapter(
        mode="learnable_1x1",
        short_side=2,
        max_long_side=2,
    )

    assert list(repeat.parameters()) == []
    assert list(red_only.parameters()) == []
    assert sum(parameter.numel() for parameter in learnable.parameters()) == 6


@pytest.mark.parametrize("mode", ["rgb", "", "REPEAT_RGB"])
def test_unknown_mode_fails_explicitly(mode: str) -> None:
    with pytest.raises(ValueError, match="Unsupported DINO input mode"):
        DinoHeatmapInputAdapter(
            mode=mode,  # type: ignore[arg-type]
            short_side=2,
            max_long_side=2,
        )
