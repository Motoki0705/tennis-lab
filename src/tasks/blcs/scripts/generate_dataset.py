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

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import repeat
import logging
import random
import sys
from collections.abc import Callable, Iterator
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
    BLCSSceneData,
    GeneratorConfig,
)
from src.tasks.blcs.generate_dataset.simulation.cell_manager import NUM_CELLS_PER_SIDE
from src.tasks.blcs.generate_dataset.simulation.ball_physics import PhysicsConfig
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyConfig
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    TargetedVelocityConfig,
)
from src.utils.projection.camera_projector import CameraConfig
from src.utils.schema.court import CourtConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., int])


@dataclass(frozen=True)
class _SceneGenerationTask:
    """Inputs required to generate one scene in a worker process."""

    scene_index: int
    scene_id: str
    seed: int


@dataclass(frozen=True)
class _SceneGenerationResult:
    """Generated scene payload and aggregated counters for one task."""

    scene_index: int
    scene_data: BLCSSceneData | None
    total_cameras: int
    total_cameras_tried: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _iter_scene_tasks(num_scenes: int, seed: int) -> Iterator[_SceneGenerationTask]:
    for scene_index in range(num_scenes):
        yield _SceneGenerationTask(
            scene_index=scene_index,
            scene_id=f"scene_{scene_index:06d}",
            seed=seed + scene_index,
        )


def _generate_scene_task(
    task: _SceneGenerationTask,
    generator_config: GeneratorConfig,
    device: str,
) -> _SceneGenerationResult:
    if torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel BLCS dataset generation only supports run.device=cpu"
        )

    _seed_everything(task.seed)
    torch.set_num_threads(1)

    generator = BLCSSceneGenerator(config=generator_config, device=device)
    from_cell = int(torch.randint(0, NUM_CELLS_PER_SIDE, (1,)).item())
    side = "near" if torch.rand(1).item() < 0.5 else "far"
    scene_data = generator.generate_scene(from_cell, side, task.scene_id)

    return _SceneGenerationResult(
        scene_index=task.scene_index,
        scene_data=scene_data,
        total_cameras=len(scene_data.cameras) if scene_data is not None else 0,
        total_cameras_tried=(
            scene_data.num_cameras_sampled if scene_data is not None else 0
        ),
    )


def _generate_parallel_scenes(
    generator_config: GeneratorConfig,
    device: str,
    num_scenes: int,
    seed: int,
    num_workers: int,
) -> Iterator[_SceneGenerationResult]:
    max_workers = min(num_workers, num_scenes)
    if max_workers <= 0:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(
            _generate_scene_task,
            _iter_scene_tasks(num_scenes, seed),
            repeat(generator_config),
            repeat(device),
            chunksize=1,
        )


def _build_parallel_stats(
    total_scenes: int,
    total_cameras: int,
    total_cameras_tried: int,
) -> dict[str, float | int]:
    acceptance_rate = (
        total_cameras / total_cameras_tried if total_cameras_tried > 0 else 0.0
    )
    avg_cameras = total_cameras / total_scenes if total_scenes > 0 else 0.0
    return {
        "total_scenes": total_scenes,
        "total_scenes_generated": total_scenes,
        "total_cameras": total_cameras,
        "total_cameras_tried": total_cameras_tried,
        "camera_acceptance_rate": acceptance_rate,
        "avg_cameras_per_scene": avg_cameras,
    }


def _build_generator_config(cfg: DictConfig) -> GeneratorConfig:
    physics_config = PhysicsConfig(
        gravity=float(cfg.physics.gravity),
        k_drag=float(cfg.physics.k_drag),
        k_magnus=float(cfg.physics.k_magnus),
        e_z=float(cfg.physics.e_z),
        mu=float(cfg.physics.mu),
        alpha_net=float(cfg.physics.alpha_net),
        alpha_net_cord=float(cfg.physics.alpha_net_cord),
        alpha_fence=float(cfg.physics.alpha_fence),
        net_half_thickness=float(cfg.physics.net_half_thickness),
        net_cord_radius=float(cfg.physics.net_cord_radius),
        dt=float(cfg.physics.dt),
        use_drag=bool(cfg.physics.use_drag),
        use_magnus=bool(cfg.physics.use_magnus),
        wind=tuple(cfg.physics.wind),
        gravity_range=tuple(cfg.physics.gravity_range),
        k_drag_range=tuple(cfg.physics.k_drag_range),
        k_magnus_range=tuple(cfg.physics.k_magnus_range),
        e_z_range=tuple(cfg.physics.e_z_range),
        mu_range=tuple(cfg.physics.mu_range),
        wind_speed_range=tuple(cfg.physics.wind_speed_range),
        wind_direction_range_deg=tuple(cfg.physics.wind_direction_range_deg),
    )

    rally_config = RallyConfig(
        z_range=tuple(cfg.rally.z_range),
        spin_x_range=tuple(cfg.rally.spin_x_range),
        spin_y_range=tuple(cfg.rally.spin_y_range),
        spin_z_range=tuple(cfg.rally.spin_z_range),
        max_sim_frames=int(cfg.rally.max_sim_frames),
        output_fps=int(cfg.rally.output_fps),
        sim_fps=int(cfg.rally.sim_fps),
        max_rallies=int(cfg.rally.max_rallies),
        max_total_frames=int(cfg.rally.max_total_frames),
        hit_timing_range=tuple(cfg.rally.hit_timing_range),
        return_z_range=tuple(cfg.rally.return_z_range),
        serve_probability=float(cfg.rally.serve_probability),
        serve_z_range=tuple(cfg.rally.serve_z_range),
        toss_vz_range=tuple(cfg.rally.toss_vz_range),
        toss_xy_noise_range=tuple(cfg.rally.toss_xy_noise_range),
        toss_max_frames=int(cfg.rally.toss_max_frames),
        toss_z0_tolerance=float(cfg.rally.toss_z0_tolerance),
        volley_probability=float(cfg.rally.volley_probability),
        normal_return_probability=float(cfg.rally.normal_return_probability),
        late_return_probability=float(cfg.rally.late_return_probability),
    )

    camera_config = CameraConfig(
        z_min=float(cfg.camera.z_min),
        z_max=float(cfg.camera.z_max),
        hfov_deg=float(cfg.camera.hfov_deg),
        image_size=tuple(cfg.camera.image_size),
        fixed_look_at=tuple(cfg.camera.fixed_look_at),
        fixed_baseline_clear_extra=float(cfg.camera.fixed_baseline_clear_extra),
        fixed_position_noise_radius=float(cfg.camera.fixed_position_noise_radius),
        fixed_look_at_xy_radius=float(cfg.camera.fixed_look_at_xy_radius),
    )

    # Court config
    court_cfg = cfg.generator.court
    court_config = CourtConfig(
        net_post_offset_x=float(court_cfg.net_post_offset_x),
        net_post_offset_x_range=tuple(court_cfg.net_post_offset_x_range),
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
        refine_max_azimuth_adjust_deg=float(cfg.targeted_velocity.refine_max_azimuth_adjust_deg),
        refine_max_frames=int(cfg.targeted_velocity.refine_max_frames),
        net_clearance_enabled=bool(cfg.targeted_velocity.net_clearance_enabled),
        net_clearance_min=float(cfg.targeted_velocity.net_clearance_min),
        net_clearance_max_attempts=int(cfg.targeted_velocity.net_clearance_max_attempts),
        net_clearance_max_frames=int(cfg.targeted_velocity.net_clearance_max_frames),
    )

    return GeneratorConfig(
        physics=physics_config,
        rally=rally_config,
        camera=camera_config,
        targeted_velocity=targeted_velocity_config,
        court=court_config,
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

    num_scenes = int(cfg.generator.num_scenes)
    num_workers = int(cfg.run.get("num_workers", 1))
    effective_workers = min(num_workers, num_scenes)
    device = str(cfg.run.device)

    logger.info("Output directory: %s", output_dir)
    logger.info("Number of scenes: %s", num_scenes)
    logger.info("Max rallies per scene: %s", cfg.rally.max_rallies)
    logger.info("Device: %s", device)

    if effective_workers > 1 and torch.device(device).type != "cpu":
        raise ValueError(
            "Parallel BLCS dataset generation requires run.device=cpu when "
            f"run.num_workers={num_workers}"
        )

    generator = None
    if effective_workers <= 1:
        generator = BLCSSceneGenerator(config=generator_config, device=device)

    writer = BLCSDatasetWriter(output_dir)

    logger.info("Starting scene generation...")
    logger.info(
        "Scene generation mode: %s",
        "parallel" if effective_workers > 1 else "serial",
    )
    logger.info("Scene generation workers: %s", effective_workers)

    total_scenes = 0
    total_cameras = 0
    total_cameras_tried = 0

    if generator is not None:
        for scene_data in tqdm(
            generator.generate(num_scenes),
            desc="Generating scenes",
            total=num_scenes,
        ):
            writer.save_scene(scene_data)
            total_scenes += 1
            total_cameras += len(scene_data.cameras)
            total_cameras_tried += scene_data.num_cameras_sampled

            if total_scenes % 100 == 0:
                avg_cams = total_cameras / total_scenes
                logger.info(
                    "Progress: %s scenes, %s cameras (avg %.1f/scene)",
                    total_scenes,
                    total_cameras,
                    avg_cams,
                )
        stats = generator.get_statistics()
    else:
        for result in tqdm(
            _generate_parallel_scenes(
                generator_config=generator_config,
                device=device,
                num_scenes=num_scenes,
                seed=seed,
                num_workers=effective_workers,
            ),
            desc="Generating scenes",
            total=num_scenes,
        ):
            if result.scene_data is None:
                continue

            writer.save_scene(result.scene_data)
            total_scenes += 1
            total_cameras += result.total_cameras
            total_cameras_tried += result.total_cameras_tried

            if total_scenes % 100 == 0:
                avg_cams = total_cameras / total_scenes
                logger.info(
                    "Progress: %s scenes, %s cameras (avg %.1f/scene)",
                    total_scenes,
                    total_cameras,
                    avg_cams,
                )

        stats = _build_parallel_stats(
            total_scenes=total_scenes,
            total_cameras=total_cameras,
            total_cameras_tried=total_cameras_tried,
        )

    logger.info("Generation complete: %s scenes, %s cameras", total_scenes, total_cameras)

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
