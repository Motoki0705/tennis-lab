"""Process-pool safety tests for background scene generation."""

from __future__ import annotations

import multiprocessing
from collections.abc import Iterable
from typing import Any

import pytest

from src.tasks.base.generate_dataset import parallel_runner


def _task(value: int, offset: int) -> int:
    return value + offset


def test_parallel_generation_uses_spawn_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeExecutor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def map(
            self,
            task_fn: Any,
            scene_indices: Iterable[int],
            repeated_arg: Iterable[int],
            *,
            chunksize: int,
        ) -> Iterable[int]:
            captured["chunksize"] = chunksize
            return (
                task_fn(scene_index, offset)
                for scene_index, offset in zip(
                    scene_indices, repeated_arg, strict=False
                )
            )

    monkeypatch.setattr(parallel_runner, "ProcessPoolExecutor", FakeExecutor)

    result = list(
        parallel_runner.run_parallel_scene_generation(
            _task,
            [1, 2],
            10,
            num_workers=8,
            chunksize=4,
        )
    )

    assert result == [11, 12]
    assert captured["max_workers"] == 2
    assert captured["chunksize"] == 4
    assert captured["mp_context"].get_start_method() == "spawn"
    assert captured["mp_context"] is multiprocessing.get_context("spawn")


def test_parallel_generation_spawn_smoke() -> None:
    result = list(
        parallel_runner.run_parallel_scene_generation(
            _task,
            [1, 2, 3],
            10,
            num_workers=2,
        )
    )

    assert result == [11, 12, 13]
