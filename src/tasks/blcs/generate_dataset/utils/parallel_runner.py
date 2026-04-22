from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from collections.abc import Iterator

import torch

from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneGenerator,
    BLCSSceneData,
    GeneratorConfig,
)

_WORKER_SCENE_GENERATOR: BLCSSceneGenerator | None = None


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
) -> BLCSSceneData:
    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel BLCS dataset generation only supports run.device=cpu"
        )
    torch.set_num_threads(1)

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
) -> Iterator[BLCSSceneData]:
    max_workers = min(num_workers, num_scenes)
    if max_workers <= 0:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(
            _generate_scene_task,
            range(num_scenes),
            repeat(generator_config),
            repeat(device),
            chunksize=1,
        )
