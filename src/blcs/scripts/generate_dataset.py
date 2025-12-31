"""Generate a BLCS dataset with Hydra-managed configuration.

Example commands:
    # Shot mode (default):
    `uv run python -m src.blcs.scripts.generate_dataset`
    `uv run python -m src.blcs.scripts.generate_dataset run.output_dir=data/blcs sampling.per_from_cell_samples=10`

    # Rally mode:
    `uv run python -m src.blcs.scripts.generate_dataset generator.mode=rally generator.num_rally_scenes=100`

Config entry point: `src/blcs/configs/generate_dataset.yaml`
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

from src.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.blcs.generate_dataset.sampling.distribution_sampler import SamplingConfig
from src.blcs.generate_dataset.scene_generator import (
    BLCSSceneGenerator,
    GenerationMode,
    GeneratorConfig,
)
from src.blcs.simulation.ball_physics import PhysicsConfig
from src.blcs.simulation.cell_manager import ShotCategory
from src.blcs.simulation.rally_simulator import RallyConfig
from src.blcs.simulation.shot_simulator import ShotConfig
from src.utils.projection.camera_projector import CameraConfig

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


def _build_generator_config(cfg: DictConfig) -> GeneratorConfig:
    physics_config = PhysicsConfig(
        gravity=float(cfg.physics.gravity),
        k_drag=float(cfg.physics.k_drag),
        k_magnus=float(cfg.physics.k_magnus),
        e_z=float(cfg.physics.e_z),
        mu=float(cfg.physics.mu),
        alpha_net=float(cfg.physics.alpha_net),
        dt=float(cfg.physics.dt),
        use_drag=bool(cfg.physics.use_drag),
        use_magnus=bool(cfg.physics.use_magnus),
    )

    shot_config = ShotConfig(
        z_range=tuple(cfg.shot.z_range),
        speed_range=tuple(cfg.shot.speed_range),
        azimuth_range_deg=tuple(cfg.shot.azimuth_range_deg),
        elevation_range_deg=tuple(cfg.shot.elevation_range_deg),
        spin_x_range=tuple(cfg.shot.spin_x_range),
        spin_y_range=tuple(cfg.shot.spin_y_range),
        spin_z_range=tuple(cfg.shot.spin_z_range),
        max_sim_frames=int(cfg.shot.max_sim_frames),
        output_fps=int(cfg.shot.output_fps),
        sim_fps=int(cfg.shot.sim_fps),
    )

    rally_config = RallyConfig(
        max_rallies=int(cfg.rally.max_rallies),
        max_total_frames=int(cfg.rally.max_total_frames),
        court_margin=float(cfg.rally.court_margin),
        hit_timing_range=tuple(cfg.rally.hit_timing_range),
        return_z_range=tuple(cfg.rally.return_z_range),
    )

    camera_config = CameraConfig(
        z_min=float(cfg.camera.z_min),
        z_max=float(cfg.camera.z_max),
        hfov_deg=float(cfg.camera.hfov_deg),
        image_size=tuple(cfg.camera.image_size),
    )

    category_ratios = cfg.sampling.category_ratios
    sampling_config = SamplingConfig(
        category_ratios={
            ShotCategory.DIRECT_NET: float(category_ratios.direct_net),
            ShotCategory.DIRECT_FENCE: float(category_ratios.direct_fence),
            ShotCategory.IN_COURT: float(category_ratios.in_court),
            ShotCategory.OUT_COURT: float(category_ratios.out_court),
        },
        in_court_cell_weights=cfg.sampling.in_court_cell_weights,
        out_court_cell_weights=cfg.sampling.out_court_cell_weights,
        per_from_cell_samples=int(cfg.sampling.per_from_cell_samples),
    )

    # Parse generation mode
    mode_str = str(cfg.generator.mode).lower()
    mode = GenerationMode.RALLY if mode_str == "rally" else GenerationMode.SHOT

    return GeneratorConfig(
        physics=physics_config,
        shot=shot_config,
        rally=rally_config,
        camera=camera_config,
        sampling=sampling_config,
        mode=mode,
        num_cameras_sampled=int(cfg.generator.num_cameras_sampled),
        ball_visibility_threshold=float(cfg.generator.ball_visibility_threshold),
        max_attempts_per_cell=int(cfg.generator.max_attempts_per_cell),
    )


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

    generator_config = _build_generator_config(cfg)
    mode = generator_config.mode

    logger.info("Output directory: %s", output_dir)
    logger.info("Generation mode: %s", mode.value)
    if mode == GenerationMode.SHOT:
        logger.info("Samples per from-cell: %s", cfg.sampling.per_from_cell_samples)
    else:
        logger.info("Number of rally scenes: %s", cfg.generator.num_rally_scenes)
        logger.info("Max rallies per scene: %s", cfg.rally.max_rallies)
    logger.info("Cameras sampled per scene: %s", generator_config.num_cameras_sampled)
    logger.info("Ball visibility threshold: %s", generator_config.ball_visibility_threshold)
    logger.info("Device: %s", cfg.run.device)

    generator = BLCSSceneGenerator(config=generator_config, device=str(cfg.run.device))
    writer = BLCSDatasetWriter(output_dir)

    logger.info("Starting scene generation...")
    total_scenes = 0
    total_cameras = 0

    if mode == GenerationMode.SHOT:
        # Shot mode: use distribution-controlled generation
        for scene_data in tqdm(generator.generate_all_scenes(), desc="Generating shots"):
            writer.save_scene(scene_data)
            total_scenes += 1
            total_cameras += len(scene_data.cameras)

            if total_scenes % 100 == 0:
                avg_cams = total_cameras / total_scenes
                logger.info(
                    "Progress: %s scenes, %s cameras (avg %.1f/scene)",
                    total_scenes,
                    total_cameras,
                    avg_cams,
                )
    else:
        # Rally mode: generate fixed number of rally scenes
        num_rally_scenes = int(cfg.generator.num_rally_scenes)
        for scene_data in tqdm(
            generator.generate_rally_scenes(num_rally_scenes),
            desc="Generating rallies",
            total=num_rally_scenes,
        ):
            writer.save_rally_scene(scene_data)
            total_scenes += 1
            total_cameras += len(scene_data.cameras)

            if total_scenes % 100 == 0:
                avg_cams = total_cameras / total_scenes
                logger.info(
                    "Progress: %s rally scenes, %s cameras (avg %.1f/scene)",
                    total_scenes,
                    total_cameras,
                    avg_cams,
                )

    logger.info("Generation complete: %s scenes, %s cameras", total_scenes, total_cameras)

    logger.info("Creating train/val/test splits...")
    writer.save_split_info(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    stats = generator.get_statistics()
    writer.save_meta_json(config=OmegaConf.to_container(cfg, resolve=True))
    writer.save_dataset_info(stats)

    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info("Output: %s", output_dir)
    logger.info("Mode: %s", mode.value)
    logger.info("Total scenes: %s", total_scenes)
    logger.info("Total cameras: %s", total_cameras)
    if total_scenes > 0:
        logger.info("Avg cameras/scene: %.2f", total_cameras / total_scenes)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
