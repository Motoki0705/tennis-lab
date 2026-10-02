from __future__ import annotations

import random
from collections.abc import Iterator

import numpy as np
import torch

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.tasks.blcs.generate_dataset.physics_retry import (
    generate_with_bounded_physics_resampling,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    BLCSSceneGenerator,
    GeneratorConfig,
)

_WORKER_SCENE_GENERATOR: BLCSSceneGenerator | None = None


def _require_positive_worker_count(num_workers: int) -> None:
    if num_workers <= 0:
        raise ValueError(
            "Parallel BLCS scene generation requires num_workers >= 1 "
            f"(got {num_workers})"
        )


def _get_worker_scene_generator(
    generator_config: GeneratorConfig, device: str
) -> BLCSSceneGenerator:
    global _WORKER_SCENE_GENERATOR
    if _WORKER_SCENE_GENERATOR is None:
        _WORKER_SCENE_GENERATOR = BLCSSceneGenerator(
            config=generator_config, device=device
        )
    return _WORKER_SCENE_GENERATOR


def _generate_scene_task(
    scene_index: int,
    generator_config: GeneratorConfig,
    device: str,
    base_seed: int,
    maximum_physics_attempts_per_scene: int | None,
) -> BLCSSceneData:
    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel BLCS dataset generation only supports run.device=cpu"
        )
    torch.set_num_threads(1)

    generator = _get_worker_scene_generator(
        generator_config,
        device,
    )
    # Per-scene seeding: forked workers otherwise share the parent's RNG
    # state, producing correlated scenes within each batch of workers. This
    # also makes scenes reproducible regardless of worker scheduling.
    torch.manual_seed(base_seed + scene_index)
    random.seed(base_seed + scene_index)
    np.random.seed(base_seed + scene_index)

    if maximum_physics_attempts_per_scene is None:
        raise ValueError(
            "Single-object BLCS generation requires an explicit bounded "
            "physics proposal budget."
        )
    scene_id = f"scene_{scene_index:06d}"
    return generate_with_bounded_physics_resampling(
        lambda: generator.generate_scene(
            generator.sample_from_cell(),
            generator.sample_side(),
            scene_id,
        ),
        scene_id=scene_id,
        maximum_attempts=maximum_physics_attempts_per_scene,
    )


def generate_parallel_scenes(
    *,
    generator_config: GeneratorConfig,
    device: str,
    num_scenes: int,
    num_workers: int,
    start_index: int,
    seed: int,
    maximum_physics_attempts_per_scene: int | None,
    chunksize: int,
) -> Iterator[BLCSSceneData]:
    # Keep a BLCS-specific guard so the task-specific error message is raised
    # before delegating (the shared runner raises a generic message).
    _require_positive_worker_count(num_workers)
    if num_scenes <= 0:
        raise ValueError(
            f"Parallel BLCS scene generation requires num_scenes >= 1 (got {num_scenes})"
        )
    if (
        isinstance(maximum_physics_attempts_per_scene, bool)
        or not isinstance(maximum_physics_attempts_per_scene, int)
        or maximum_physics_attempts_per_scene <= 0
    ):
        raise ValueError(
            "Single-object BLCS generation requires "
            "maximum_physics_attempts_per_scene >= 1."
        )

    results = run_parallel_scene_generation(
        _generate_scene_task,
        list(range(start_index, start_index + num_scenes)),
        generator_config,
        device,
        seed,
        maximum_physics_attempts_per_scene,
        num_workers=num_workers,
        chunksize=chunksize,
    )

    yield from results
