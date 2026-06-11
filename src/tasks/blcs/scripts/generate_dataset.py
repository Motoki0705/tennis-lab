"""Generate a BLCS dataset with Hydra-managed configuration.

Usage:
    python -m src.tasks.blcs.scripts.generate_dataset
    python -m src.tasks.blcs.scripts.generate_dataset generator.num_scenes=100
    python -m src.tasks.blcs.scripts.generate_dataset run.output_dir=data/blcs generator.num_scenes=500
    python -m src.tasks.blcs.scripts.generate_dataset run.num_workers=4

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/generate_dataset.yaml`.
    - The script generates scenes, writes splits, and persists dataset metadata.
    - Parallel scene generation uses ProcessPoolExecutor and currently supports CPU workers.
"""

from __future__ import annotations

import logging
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., int])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _hydra_main(func: F) -> F:
    return cast(
        F,
        hydra.main(config_path="../configs", config_name="generate_dataset", version_base="1.3")(
            func
        ),
    )


@_hydra_main
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Generate scenes and write them to disk."""
    logger.info("=" * 60)
    logger.info("BLCS Dataset Generator")
    logger.info("=" * 60)

    output_dir = Path(to_absolute_path(str(cfg.run.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, output_dir / "config.yaml")

    seed = int(cfg.run.seed)
    _seed_everything(seed)

    train_ratio = float(cfg.run.train_ratio)
    val_ratio = float(cfg.run.val_ratio)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio} (sum > 1.0)"
        )

    num_scenes = int(cfg.generator.num_scenes)
    num_workers = int(cfg.run.get("num_workers", 1))
    device = str(cfg.run.device)
    generator_config = build_generator_config(cfg)

    logger.info("Output directory: %s", output_dir)
    logger.info("Number of scenes: %s", num_scenes)
    logger.info("Max rallies per scene: %s", cfg.rally.max_rallies)
    logger.info("Device: %s", device)

    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel BLCS dataset generation requires run.device=cpu when "
            f"run.num_workers={num_workers}"
        )

    writer = BLCSDatasetWriter(output_dir)

    logger.info("Starting scene generation...")
    logger.info("Scene generation mode: parallel")
    logger.info("Scene generation workers: %s", num_workers)

    total_scenes = 0

    for scene_data in tqdm(
        generate_parallel_scenes(
            generator_config=generator_config,
            device=device,
            num_scenes=num_scenes,
            num_workers=num_workers,
            seed=seed,
        ),
        desc="Generating scenes",
        total=num_scenes,
    ):
        writer.save_scene(scene_data)
        total_scenes += 1

        if total_scenes % 100 == 0:
            logger.info(
                "Progress: %s scenes",
                total_scenes,
            )

    logger.info("Generation complete: %s scenes", total_scenes)

    if total_scenes < num_scenes:
        logger.warning("Only generated %s/%s scenes", total_scenes, num_scenes)

    logger.info("Creating train/val/test splits...")
    writer.save_split_info(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    writer.save_meta_json(config=OmegaConf.to_container(cfg, resolve=True))

    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info("Output: %s", output_dir)
    logger.info("Total scenes: %s", total_scenes)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
