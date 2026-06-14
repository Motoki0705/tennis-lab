from __future__ import annotations

from collections.abc import Iterator

import torch

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
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
    generator_config: GeneratorConfig,
    device: str,
) -> BLCSSceneGenerator:
    global _WORKER_SCENE_GENERATOR
    if _WORKER_SCENE_GENERATOR is None:
        _WORKER_SCENE_GENERATOR = BLCSSceneGenerator(
            config=generator_config,
            device=device,
        )
    return _WORKER_SCENE_GENERATOR


def _generate_scene_task(
    scene_index: int,
    generator_config: GeneratorConfig,
    device: str,
    base_seed: int,
) -> BLCSSceneData:
    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel BLCS dataset generation only supports run.device=cpu"
        )
    torch.set_num_threads(1)

    # Per-scene seeding: forked workers otherwise share the parent's RNG
    # state, producing correlated scenes within each batch of workers. This
    # also makes scenes reproducible regardless of worker scheduling.
    torch.manual_seed(base_seed + scene_index)

    generator = _get_worker_scene_generator(generator_config, device)
    from_cell = generator.sample_from_cell()
    side = generator.sample_side()
    scene_data = generator.generate_scene(from_cell, side, f"scene_{scene_index:06d}")

    return scene_data


def generate_parallel_scenes(
    generator_config: GeneratorConfig,
    device: str,
    num_scenes: int,
    num_workers: int,
    seed: int = 0,
) -> Iterator[BLCSSceneData]:
    # Keep a BLCS-specific guard so the task-specific error message is raised
    # before delegating (the shared runner raises a generic message).
    _require_positive_worker_count(num_workers)
    if num_scenes <= 0:
        raise ValueError(
            f"Parallel BLCS scene generation requires num_scenes >= 1 (got {num_scenes})"
        )

    yield from run_parallel_scene_generation(
        _generate_scene_task,
        list(range(num_scenes)),
        generator_config,
        device,
        seed,
        num_workers=num_workers,
    )
