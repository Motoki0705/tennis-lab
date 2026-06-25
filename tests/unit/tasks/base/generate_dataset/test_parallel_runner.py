"""Unit tests for the parallel scene-generation fan-out helper.

Under the ``spawn`` start method, the worker callable must be importable by
qualified name in a *fresh* interpreter. A function defined in this test module
is NOT reliably importable there (pytest's ``--import-mode=importlib`` registers
the test module under a synthetic name with no ``__init__.py`` package on disk).
We therefore drive the cross-process tests with the builtin :func:`pow`, which
pickles cleanly and is importable everywhere: ``pow(idx, exponent)``.
"""

from __future__ import annotations

import pytest

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)

pytestmark = pytest.mark.unit


def test_num_workers_below_one_raises() -> None:
    with pytest.raises(ValueError, match="num_workers >= 1"):
        list(run_parallel_scene_generation(pow, [0, 1], 2, num_workers=0))


def test_results_preserve_input_order() -> None:
    indices = [1, 2, 3, 4]
    # pow(idx, 2) == idx ** 2
    out = list(run_parallel_scene_generation(pow, indices, 2, num_workers=2))
    assert out == [1, 4, 9, 16]


def test_repeated_args_broadcast_to_each_call() -> None:
    # exponent 3 broadcast to every call: pow(2,3)=8, pow(5,3)=125
    out = list(run_parallel_scene_generation(pow, [2, 5], 3, num_workers=1))
    assert out == [8, 125]


def test_empty_indices_raises_value_error() -> None:
    # NOTE (likely bug): for an empty index list, max_workers = min(num_workers, 0)
    # == 0, and ProcessPoolExecutor rejects max_workers <= 0. So an empty workload
    # raises instead of yielding nothing. This test documents the current behavior.
    with pytest.raises(ValueError, match="max_workers must be greater than 0"):
        list(run_parallel_scene_generation(pow, [], 2, num_workers=4))
