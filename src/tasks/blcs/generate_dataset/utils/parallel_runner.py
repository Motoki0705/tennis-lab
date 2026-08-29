from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.physics_retry import (
    generate_with_bounded_physics_resampling,
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
    timeline_config: dict[str, Any] | None,
    maximum_physics_attempts_per_object: int | None,
) -> BLCSSceneGenerator | MultiBallSceneGenerator:
    global _WORKER_SCENE_GENERATOR
    if _WORKER_SCENE_GENERATOR is None:
        base = BLCSSceneGenerator(
            config=generator_config,
            device=device,
        )
        if multi_object:
            if timeline_config is None:
                raise ValueError(
                    "Multi-object BLCS generation requires timeline config."
                )
            if maximum_physics_attempts_per_object is None:
                raise ValueError(
                    "Multi-object BLCS generation requires an explicit bounded "
                    "physics proposal budget."
                )
            _WORKER_SCENE_GENERATOR = MultiBallSceneGenerator(
                base,
                timeline=timeline_config,
                maximum_physics_attempts_per_object=(
                    maximum_physics_attempts_per_object
                ),
                rng=random.Random(random.getrandbits(64)),
            )
        else:
            _WORKER_SCENE_GENERATOR = base
    return _WORKER_SCENE_GENERATOR


def _generate_scene_task(
    scene_index: int,
    generator_config: GeneratorConfig,
    device: str,
    base_seed: int,
    multi_object: bool,
    timeline_config: dict[str, Any] | None,
    maximum_physics_attempts_per_scene: int | None,
    maximum_physics_attempts_per_object: int | None,
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
    random.seed(base_seed + scene_index)
    np.random.seed(base_seed + scene_index)

    generator = _get_worker_scene_generator(
        generator_config,
        device,
        multi_object,
        timeline_config,
        maximum_physics_attempts_per_object,
    )
    if isinstance(generator, MultiBallSceneGenerator):
        generator.composer.rng.seed(base_seed + scene_index)
        return generator.generate_scene(f"scene_{scene_index:06d}")
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
    multi_object: bool,
    timeline_config: dict[str, Any] | None,
    maximum_physics_attempts_per_scene: int | None,
    maximum_physics_attempts_per_object: int | None,
    chunksize: int,
) -> Iterator[BLCSSceneData]:
    # Keep a BLCS-specific guard so the task-specific error message is raised
    # before delegating (the shared runner raises a generic message).
    _require_positive_worker_count(num_workers)
    if num_scenes <= 0:
        raise ValueError(
            f"Parallel BLCS scene generation requires num_scenes >= 1 (got {num_scenes})"
        )
    if multi_object:
        if maximum_physics_attempts_per_scene is not None:
            raise ValueError(
                "Multi-object BLCS generation does not accept a single-object "
                "physics proposal budget."
            )
        if (
            isinstance(maximum_physics_attempts_per_object, bool)
            or not isinstance(maximum_physics_attempts_per_object, int)
            or maximum_physics_attempts_per_object <= 0
        ):
            raise ValueError(
                "Multi-object BLCS generation requires "
                "maximum_physics_attempts_per_object >= 1."
            )
    elif maximum_physics_attempts_per_object is not None:
        raise ValueError(
            "Single-object BLCS generation does not accept a multi-object "
            "physics proposal budget."
        )
    elif (
        isinstance(maximum_physics_attempts_per_scene, bool)
        or not isinstance(maximum_physics_attempts_per_scene, int)
        or maximum_physics_attempts_per_scene <= 0
    ):
        raise ValueError(
            "Single-object BLCS generation requires "
            "maximum_physics_attempts_per_scene >= 1."
        )

    yield from run_parallel_scene_generation(
        _generate_scene_task,
        list(range(start_index, start_index + num_scenes)),
        generator_config,
        device,
        seed,
        multi_object,
        timeline_config,
        maximum_physics_attempts_per_scene,
        maximum_physics_attempts_per_object,
        num_workers=num_workers,
        chunksize=chunksize,
    )
