from __future__ import annotations

from collections.abc import Iterator

import torch

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneData,
    BLCSSceneGenerator,
    GeneratorConfig,
)

_WORKER_SCENE_GENERATOR: BLCSSceneGenerator | MultiBallSceneGenerator | None = None


def _require_positive_worker_count(num_workers: int) -> None:
    if num_workers <= 0:
        raise ValueError(
            "Parallel BLCS scene generation requires num_workers >= 1 "
            f"(got {num_workers})"
        )


def _get_worker_scene_generator(
    generator_config: GeneratorConfig,
    device: str,
    multi_object: bool,
    min_balls: int,
    max_balls: int,
) -> BLCSSceneGenerator | MultiBallSceneGenerator:
    global _WORKER_SCENE_GENERATOR
    if _WORKER_SCENE_GENERATOR is None:
        base = BLCSSceneGenerator(
            config=generator_config,
            device=device,
        )
        _WORKER_SCENE_GENERATOR = (
            MultiBallSceneGenerator(
                base, min_balls=min_balls, max_balls=max_balls
            )
            if multi_object
            else base
        )
    return _WORKER_SCENE_GENERATOR


def _generate_scene_task(
    scene_index: int,
    generator_config: GeneratorConfig,
    device: str,
    base_seed: int,
    multi_object: bool,
    min_balls: int,
    max_balls: int,
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

    generator = _get_worker_scene_generator(
        generator_config, device, multi_object, min_balls, max_balls
    )
    if isinstance(generator, MultiBallSceneGenerator):
        return generator.generate_scene(f"scene_{scene_index:06d}")
    from_cell = generator.sample_from_cell()
    side = generator.sample_side()
    scene_data = generator.generate_scene(from_cell, side, f"scene_{scene_index:06d}")
    if scene_data is None:
        raise RuntimeError("BLCS physical scene generation returned no scene.")
    return scene_data


def generate_parallel_scenes(
    generator_config: GeneratorConfig,
    device: str,
    num_scenes: int,
    num_workers: int,
    seed: int = 0,
    multi_object: bool = False,
    min_balls: int = 1,
    max_balls: int = 1,
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
        multi_object,
        min_balls,
        max_balls,
        num_workers=num_workers,
    )
