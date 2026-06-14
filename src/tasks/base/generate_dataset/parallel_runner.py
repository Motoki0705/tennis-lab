"""Generic ProcessPoolExecutor fan-out for parallel scene generation.

Extracts the executor mechanics shared by the PLCS and BLCS parallel
runners.  The task-specific worker init (the module-level
``_WORKER_SCENE_GENERATOR`` cache and ``_generate_scene_task``) stays in each
task package; this helper only handles validating worker count, capping
``max_workers``, and mapping the task function across scene indices with the
remaining arguments broadcast via :func:`itertools.repeat`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def run_parallel_scene_generation(
    task_fn: Callable[..., Any],
    scene_indices: list[int],
    *repeated_args: Any,
    num_workers: int,
    chunksize: int = 1,
) -> Iterator[Any]:
    """Map ``task_fn`` over ``scene_indices`` across worker processes.

    Args:
        task_fn: Picklable worker function called as
            ``task_fn(scene_index, *repeated_args)``.
        scene_indices: Sequence of scene indices to process.
        *repeated_args: Additional arguments broadcast to every call via
            :func:`itertools.repeat`.
        num_workers: Maximum number of worker processes (must be >= 1).
        chunksize: ``executor.map`` chunk size.

    Yields:
        Results in the order of ``scene_indices``.
    """
    if num_workers < 1:
        raise ValueError(
            f"Parallel scene generation requires num_workers >= 1 (got {num_workers})"
        )

    max_workers = min(num_workers, len(scene_indices))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(
            task_fn,
            scene_indices,
            *(itertools.repeat(arg) for arg in repeated_args),
            chunksize=chunksize,
        )
