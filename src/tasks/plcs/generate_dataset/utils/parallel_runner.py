from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from src.tasks.plcs.generate_dataset.sampling.motion_sampler import MotionSampler
from src.tasks.plcs.generate_dataset.scene_generator import SceneData, SceneGenerator

_WORKER_SCENE_GENERATOR: SceneGenerator | None = None


def _require_positive_worker_count(num_workers: int) -> None:
    if num_workers <= 0:
        raise ValueError(
            "Parallel PLCS scene generation requires num_workers >= 1 "
            f"(got {num_workers})"
        )


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
) -> SceneGenerator:
    global _WORKER_SCENE_GENERATOR
    if _WORKER_SCENE_GENERATOR is None:
        cfg = OmegaConf.create(config_dict)
        if not isinstance(cfg, DictConfig):
            raise TypeError("PLCS worker config must resolve to a DictConfig.")
        _WORKER_SCENE_GENERATOR = build_scene_generator(cfg, device)
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
    scene_generator = _get_worker_scene_generator(config_dict, device)
    return scene_generator.generate_scene(scene_id=f"scene_{scene_index:06d}")


def generate_parallel_scenes(
    config: DictConfig,
    device: str,
    start_index: int,
    num_scenes: int,
    num_workers: int,
) -> Iterator[SceneData]:
    """Generate PLCS scenes in parallel worker processes."""
    _require_positive_worker_count(num_workers)
    if num_scenes <= 0:
        raise ValueError(
            f"Parallel PLCS scene generation requires num_scenes >= 1 (got {num_scenes})"
        )

    max_workers = min(num_workers, num_scenes)

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("PLCS parallel config must resolve to a dictionary.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(
            _generate_scene_task,
            range(start_index, start_index + num_scenes),
            repeat(config_dict),
            repeat(device),
            chunksize=1,
        )
