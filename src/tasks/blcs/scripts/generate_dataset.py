"""Generate a BLCS dataset with Hydra-managed configuration.

Usage:
    python -m src.tasks.blcs.scripts.generate_dataset
    python -m src.tasks.blcs.scripts.generate_dataset generator.num_scenes=100
    python -m src.tasks.blcs.scripts.generate_dataset run.output_dir=blcs_norm-v1_custom generator.num_scenes=500
    python -m src.tasks.blcs.scripts.generate_dataset run.num_workers=4

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/generate_dataset.yaml`.
    - The script generates scenes, writes splits, and persists dataset metadata.
    - Parallel scene generation uses ProcessPoolExecutor and currently supports CPU workers.
    - `generation` changes only object cardinality; both modes use the same simulator and writer.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.tasks.blcs.configuration import parse_generation_run
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.utils.parallel_runner import (
    generate_parallel_scenes,
)
from src.utils.hydra import hydra_main
from src.utils.seeding import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@hydra_main(
    config_path="../configs",
    config_name="generate_dataset",
    version_base="1.3",
    validation_boundary="blcs.generate_dataset",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Generate scenes and write them to disk."""
    run, _resolver = parse_generation_run(cfg)
    generator_config = build_generator_config(cfg)
    output_dir = run.output_dir
    writer = BLCSDatasetWriter(
        output_dir,
        court_coordinate_normalization=(
            generator_config.court_coordinate_normalization
        ),
    )
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info("=" * 60)
    logger.info("BLCS Dataset Generator")
    logger.info("=" * 60)

    seed = run.seed
    seed_everything(seed)

    generation_mode = str(cfg.generation.mode)
    if generation_mode not in {"single_object", "multi_object"}:
        raise ValueError(
            f"Unsupported generation.mode='{generation_mode}'. "
            "Supported: ['single_object', 'multi_object']"
        )

    train_ratio = run.train_ratio
    val_ratio = run.val_ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio} (sum > 1.0)"
        )

    num_scenes = int(cfg.generator.num_scenes)
    num_workers = run.num_workers
    device = run.device

    logger.info("Output directory: %s", output_dir)
    logger.info("Number of scenes: %s", num_scenes)
    logger.info("Max rallies per scene: %s", cfg.rally.max_rallies)
    logger.info("Device: %s", device)

    logger.info("Starting scene generation...")
    logger.info("Scene generation mode: parallel")
    logger.info("Scene generation workers: %s", num_workers)

    total_scenes = 0

    scene_iterator = generate_parallel_scenes(
        generator_config=generator_config,
        device=device,
        num_scenes=num_scenes,
        num_workers=num_workers,
        start_index=0,
        seed=seed,
        multi_object=generation_mode == "multi_object",
        timeline_config=(
            cast(
                dict[str, object],
                OmegaConf.to_container(cfg.generation.timeline, resolve=True),
            )
            if generation_mode == "multi_object"
            else None
        ),
        maximum_physics_attempts_per_object=(
            int(cfg.generation.maximum_physics_attempts_per_object)
            if generation_mode == "multi_object"
            else None
        ),
        chunksize=run.chunksize,
    )

    for scene_data in tqdm(
        scene_iterator,
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

    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("BLCS generator configuration must resolve to a mapping.")
    writer.save_meta_json(config=resolved_config)

    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info("Output: %s", output_dir)
    logger.info("Total scenes: %s", total_scenes)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
