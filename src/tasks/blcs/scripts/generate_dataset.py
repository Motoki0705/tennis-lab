"""Generate a BLCS dataset with Hydra-managed configuration.

Example commands:
    `uv run python -m src.tasks.blcs.scripts.generate_dataset`
    `uv run python -m src.tasks.blcs.scripts.generate_dataset generator.num_scenes=100`
    `uv run python -m src.tasks.blcs.scripts.generate_dataset run.output_dir=data/blcs generator.num_scenes=500`

Config entry point: `src/tasks/blcs/configs/generate_dataset.yaml`
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

from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import (
    BLCSSceneGenerator,
    GeneratorConfig,
)
from src.tasks.blcs.generate_dataset.ball_physics import PhysicsConfig
from src.tasks.blcs.generate_dataset.rally_simulator import RallyConfig
from src.tasks.blcs.generate_dataset.shot_simulator import ShotConfig
from src.tasks.blcs.generate_dataset.targeted_velocity_sampler import TargetedVelocityConfig
from src.utils.projection.camera_projector import CameraConfig
from src.utils.schema.court import CourtConfig

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
        wind=tuple(cfg.physics.wind) if cfg.physics.get("wind") else (0.0, 0.0, 0.0),
        gravity_range=tuple(cfg.physics.gravity_range)
        if cfg.physics.get("gravity_range") else None,
        k_drag_range=tuple(cfg.physics.k_drag_range)
        if cfg.physics.get("k_drag_range") else None,
        k_magnus_range=tuple(cfg.physics.k_magnus_range)
        if cfg.physics.get("k_magnus_range") else None,
        e_z_range=tuple(cfg.physics.e_z_range)
        if cfg.physics.get("e_z_range") else None,
        mu_range=tuple(cfg.physics.mu_range)
        if cfg.physics.get("mu_range") else None,
        wind_speed_range=tuple(cfg.physics.wind_speed_range)
        if cfg.physics.get("wind_speed_range") else None,
        wind_direction_range_deg=tuple(cfg.physics.wind_direction_range_deg)
        if cfg.physics.get("wind_direction_range_deg") else None,
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
        serve_speed_range=tuple(cfg.shot.serve_speed_range)
        if cfg.shot.get("serve_speed_range") else (30.0, 55.0),
        serve_elevation_range_deg=tuple(cfg.shot.serve_elevation_range_deg)
        if cfg.shot.get("serve_elevation_range_deg") else (2.0, 10.0),
        serve_z_range=tuple(cfg.shot.serve_z_range)
        if cfg.shot.get("serve_z_range") else (2.0, 2.8),
        serve_azimuth_range_deg=tuple(cfg.shot.serve_azimuth_range_deg)
        if cfg.shot.get("serve_azimuth_range_deg") else (-15.0, 15.0),
    )

    rally_config = RallyConfig(
        max_rallies=int(cfg.rally.max_rallies),
        max_total_frames=int(cfg.rally.max_total_frames),
        court_margin=float(cfg.rally.court_margin),
        hit_timing_range=tuple(cfg.rally.hit_timing_range),
        return_z_range=tuple(cfg.rally.return_z_range),
        min_rally_length=int(cfg.rally.min_rally_length),
        net_fault_accept_prob=float(cfg.rally.net_fault_accept_prob),
        serve_probability=float(cfg.rally.get("serve_probability", 0.3)),
        serve_speed_range=tuple(cfg.rally.serve_speed_range)
        if cfg.rally.get("serve_speed_range") else (30.0, 55.0),
        serve_elevation_range_deg=tuple(cfg.rally.serve_elevation_range_deg)
        if cfg.rally.get("serve_elevation_range_deg") else (2.0, 10.0),
        volley_probability=float(cfg.rally.get("volley_probability", 0.05)),
        normal_return_probability=float(cfg.rally.get("normal_return_probability", 0.85)),
        late_return_probability=float(cfg.rally.get("late_return_probability", 0.10)),
    )

    camera_config = CameraConfig(
        z_min=float(cfg.camera.z_min),
        z_max=float(cfg.camera.z_max),
        hfov_deg=float(cfg.camera.hfov_deg),
        image_size=tuple(cfg.camera.image_size),
        hfov_noise_deg=float(cfg.camera.get("hfov_noise_deg", 0.0)),
        look_at_noise_std=float(cfg.camera.get("look_at_noise_std", 0.0)),
    )

    # Court config
    court_cfg = cfg.generator.get("court", {})
    court_config = CourtConfig(
        net_post_offset_x=float(court_cfg.get("net_post_offset_x", 0.914)),
        net_post_offset_x_range=tuple(court_cfg.net_post_offset_x_range)
        if court_cfg.get("net_post_offset_x_range") else None,
    )

    targeted_velocity_config = TargetedVelocityConfig(
        azimuth_noise_deg=float(cfg.targeted_velocity.azimuth_noise_deg),
        elevation_noise_deg=float(cfg.targeted_velocity.elevation_noise_deg),
        speed_variation=float(cfg.targeted_velocity.speed_variation),
        min_elevation_deg=float(cfg.targeted_velocity.min_elevation_deg),
        max_elevation_deg=float(cfg.targeted_velocity.max_elevation_deg),
        drive_elevation_range_deg=tuple(cfg.targeted_velocity.drive_elevation_range_deg),
        lob_elevation_range_deg=tuple(cfg.targeted_velocity.lob_elevation_range_deg),
        lob_probability=float(cfg.targeted_velocity.lob_probability),
        min_speed=float(cfg.targeted_velocity.min_speed),
        max_speed=float(cfg.targeted_velocity.max_speed),
        gravity=float(cfg.targeted_velocity.gravity),
        speed_solve_max_iters=int(cfg.targeted_velocity.speed_solve_max_iters),
        speed_solve_tol=float(cfg.targeted_velocity.speed_solve_tol),
        refine_enabled=bool(cfg.targeted_velocity.refine_enabled),
        refine_iters=int(cfg.targeted_velocity.refine_iters),
        refine_speed_scale_min=float(cfg.targeted_velocity.refine_speed_scale_min),
        refine_speed_scale_max=float(cfg.targeted_velocity.refine_speed_scale_max),
        refine_max_azimuth_adjust_deg=float(
            cfg.targeted_velocity.refine_max_azimuth_adjust_deg
        ),
        refine_max_frames=int(cfg.targeted_velocity.refine_max_frames),
        net_clearance_enabled=bool(cfg.targeted_velocity.net_clearance_enabled),
        net_clearance_min=float(cfg.targeted_velocity.net_clearance_min),
        net_clearance_max_attempts=int(cfg.targeted_velocity.net_clearance_max_attempts),
        net_clearance_max_frames=int(cfg.targeted_velocity.net_clearance_max_frames),
    )

    return GeneratorConfig(
        physics=physics_config,
        shot=shot_config,
        rally=rally_config,
        camera=camera_config,
        targeted_velocity=targeted_velocity_config,
        court=court_config,
        num_cameras_sampled=int(cfg.generator.num_cameras_sampled),
        ball_visibility_threshold=float(cfg.generator.ball_visibility_threshold),
        max_attempts_multiplier=int(cfg.generator.max_attempts_multiplier),
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

    logger.info("Output directory: %s", output_dir)
    logger.info("Number of scenes: %s", cfg.generator.num_scenes)
    logger.info("Max rallies per scene: %s", cfg.rally.max_rallies)
    logger.info("Cameras sampled per scene: %s", generator_config.num_cameras_sampled)
    logger.info("Ball visibility threshold: %s", generator_config.ball_visibility_threshold)
    logger.info("Device: %s", cfg.run.device)

    generator = BLCSSceneGenerator(config=generator_config, device=str(cfg.run.device))
    writer = BLCSDatasetWriter(output_dir)

    logger.info("Starting scene generation...")
    total_scenes = 0
    total_cameras = 0

    num_scenes = int(cfg.generator.num_scenes)
    for scene_data in tqdm(
        generator.generate(num_scenes),
        desc="Generating scenes",
        total=num_scenes,
    ):
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
    logger.info("Total scenes: %s", total_scenes)
    logger.info("Total cameras: %s", total_cameras)
    if total_scenes > 0:
        logger.info("Avg cameras/scene: %.2f", total_cameras / total_scenes)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(cast(Callable[[], int], main)())
