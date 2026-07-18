from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.tasks.plcs.generate_dataset.multi_object_scene_generator import (
    MultiPersonSceneGenerator,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSampler
from src.tasks.plcs.generate_dataset.scene_generator import SceneData, SceneGenerator

_WORKER_SCENE_GENERATOR: SceneGenerator | MultiPersonSceneGenerator | None = None


def build_scene_generator(
    config: DictConfig,
    device: str,
) -> SceneGenerator:
    """Create a PLCS scene generator from a resolved Hydra config."""
    motion_sampler = MotionSampler(
        config=config,
        smplh_model_path=config.paths.smplh_model_path,
        device=device,
    )
    return SceneGenerator(
        config=config,
        motion_sampler=motion_sampler,
        device=device,
    )


def _get_worker_scene_generator(
    config_dict: dict[str, Any],
    device: str,
) -> SceneGenerator | MultiPersonSceneGenerator:
    global _WORKER_SCENE_GENERATOR
    if _WORKER_SCENE_GENERATOR is None:
        cfg = OmegaConf.create(config_dict)
        if not isinstance(cfg, DictConfig):
            raise TypeError("PLCS worker config must resolve to a DictConfig.")
        base = build_scene_generator(cfg, device)
        generation = cfg.get("generation", {})
        _WORKER_SCENE_GENERATOR = (
            MultiPersonSceneGenerator(
                base,
                timeline=OmegaConf.to_container(generation.timeline, resolve=True),
                rng=random.Random(random.getrandbits(64)),
            )
            if str(generation.get("mode", "single_object")) == "multi_object"
            else base
        )
    return _WORKER_SCENE_GENERATOR


def _generate_scene_task(
    scene_index: int,
    config_dict: dict[str, Any],
    device: str,
) -> SceneData:
    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel PLCS dataset generation only supports run.device=cpu"
        )

    torch.set_num_threads(1)
    random.seed(scene_index)
    np.random.seed(scene_index)
    torch.manual_seed(scene_index)
    scene_generator = _get_worker_scene_generator(config_dict, device)
    if isinstance(scene_generator, MultiPersonSceneGenerator):
        scene_generator.composer.rng.seed(scene_index)
    return scene_generator.generate_scene(scene_id=f"scene_{scene_index:06d}")


def generate_parallel_scenes(
    config: DictConfig,
    device: str,
    start_index: int,
    num_scenes: int,
    num_workers: int,
) -> Iterator[SceneData]:
    """Generate PLCS scenes in parallel worker processes."""
    if num_workers < 1:
        raise ValueError(
            "Parallel PLCS scene generation requires num_workers >= 1 "
            f"(got {num_workers})"
        )
    if num_scenes <= 0:
        raise ValueError(
            f"Parallel PLCS scene generation requires num_scenes >= 1 (got {num_scenes})"
        )

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("PLCS parallel config must resolve to a dictionary.")

    yield from run_parallel_scene_generation(
        _generate_scene_task,
        list(range(start_index, start_index + num_scenes)),
        config_dict,
        device,
        num_workers=num_workers,
    )
