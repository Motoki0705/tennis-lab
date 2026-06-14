"""Unit tests for the shared data/generate foundation modules."""

from __future__ import annotations

import pytest

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.utils.data.augmentation import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    parse_float_range,
)


def _square(value: int) -> int:
    """Module-level picklable task function for the parallel runner test."""
    return value * value


def test_parse_float_range_valid() -> None:
    assert parse_float_range([0.5, 1.5], "scale") == (0.5, 1.5)
    assert parse_float_range((1.0, 1.0), "scale") == (1.0, 1.0)


@pytest.mark.parametrize(
    "value",
    [
        "not-a-range",
        b"bytes",
        [1.0],
        [1.0, 2.0, 3.0],
        [2.0, 1.0],  # min > max
    ],
)
def test_parse_float_range_invalid(value: object) -> None:
    with pytest.raises(ValueError):
        parse_float_range(value, "scale")


def test_imagenet_constants() -> None:
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def test_run_parallel_scene_generation_matches_sequential() -> None:
    scene_indices = list(range(8))
    expected = list(map(_square, scene_indices))

    result = list(
        run_parallel_scene_generation(
            _square,
            scene_indices,
            num_workers=2,
        )
    )

    assert result == expected
    assert len(result) == len(scene_indices)


def test_run_parallel_scene_generation_invalid_workers() -> None:
    with pytest.raises(ValueError):
        list(run_parallel_scene_generation(_square, [0, 1], num_workers=0))
