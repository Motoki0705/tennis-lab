#!/usr/bin/env python
"""Generate BLCS dataset with controlled distribution.

Usage:
    python -m blcs.scripts.generate_dataset --output-dir data/blcs --samples-per-cell 100
    python -m blcs.scripts.generate_dataset --config configs/blcs_dataset.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml
from tqdm.auto import tqdm

from src.blcs.data.camera_projector import CameraConfig
from src.blcs.data.dataset_writer import BLCSDatasetWriter
from src.blcs.data.distribution_sampler import SamplingConfig
from src.blcs.data.scene_generator import BLCSSceneGenerator, GeneratorConfig
from src.blcs.simulation.ball_physics import PhysicsConfig
from src.blcs.simulation.cell_manager import ShotCategory
from src.blcs.simulation.shot_simulator import ShotConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate BLCS dataset with controlled distribution."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/blcs",
        help="Output directory for dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--samples-per-cell",
        type=int,
        default=100,
        help="Target samples per from_cell.",
    )
    parser.add_argument(
        "--num-cameras-sampled",
        type=int,
        default=15,
        help="Number of cameras to try per scene (filtered by visibility).",
    )
    parser.add_argument(
        "--ball-visibility-threshold",
        type=float,
        default=0.8,
        help="Min ball visibility ratio to keep a camera (0.0-1.0).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation.",
    )

    return parser.parse_args()


def load_config(config_path: str | None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        dict: Configuration dictionary.

    """
    if config_path is None:
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_generator_config(
    args: argparse.Namespace, yaml_config: dict
) -> GeneratorConfig:
    """Create generator config from args and YAML.

    Args:
        args: Command line arguments.
        yaml_config: YAML configuration.

    Returns:
        GeneratorConfig instance.

    """
    # Physics config
    phys_cfg = yaml_config.get("physics", {})
    physics_config = PhysicsConfig(
        gravity=phys_cfg.get("gravity", 9.81),
        k_drag=phys_cfg.get("k_drag", 0.01),
        k_magnus=phys_cfg.get("k_magnus", 0.001),
        e_z=phys_cfg.get("e_z", 0.75),
        mu=phys_cfg.get("mu", 0.1),
        alpha_net=phys_cfg.get("alpha_net", 0.3),
        dt=phys_cfg.get("dt", 1 / 240),
        use_drag=phys_cfg.get("use_drag", True),
        use_magnus=phys_cfg.get("use_magnus", True),
    )

    # Shot config
    shot_cfg = yaml_config.get("shot", {})
    shot_config = ShotConfig(
        z_range=tuple(shot_cfg.get("z_range", [0.8, 1.4])),
        speed_range=tuple(shot_cfg.get("speed_range", [15.0, 35.0])),
        azimuth_range_deg=tuple(shot_cfg.get("azimuth_range_deg", [-30.0, 30.0])),
        elevation_range_deg=tuple(shot_cfg.get("elevation_range_deg", [5.0, 25.0])),
        spin_x_range=tuple(shot_cfg.get("spin_x_range", [-20.0, 20.0])),
        spin_y_range=tuple(shot_cfg.get("spin_y_range", [-80.0, -40.0])),
        spin_z_range=tuple(shot_cfg.get("spin_z_range", [-20.0, 20.0])),
        max_sim_frames=shot_cfg.get("max_sim_frames", 2000),
        output_fps=shot_cfg.get("output_fps", 30),
        sim_fps=shot_cfg.get("sim_fps", 240),
    )

    # Camera config
    cam_cfg = yaml_config.get("camera", {})
    camera_config = CameraConfig(
        z_min=cam_cfg.get("z_min", 3.0),
        z_max=cam_cfg.get("z_max", 5.0),
        hfov_deg=cam_cfg.get("hfov_deg", 60.0),
        image_size=tuple(cam_cfg.get("image_size", [1280, 720])),
    )

    # Sampling config
    samp_cfg = yaml_config.get("sampling", {})
    cat_ratios = samp_cfg.get("category_ratios", {})
    sampling_config = SamplingConfig(
        category_ratios={
            ShotCategory.DIRECT_NET: cat_ratios.get("direct_net", 0.05),
            ShotCategory.DIRECT_FENCE: cat_ratios.get("direct_fence", 0.05),
            ShotCategory.IN_COURT: cat_ratios.get("in_court", 0.60),
            ShotCategory.OUT_COURT: cat_ratios.get("out_court", 0.30),
        },
        in_court_cell_weights=samp_cfg.get("in_court_cell_weights", "uniform"),
        out_court_cell_weights=samp_cfg.get("out_court_cell_weights", "uniform"),
        per_from_cell_samples=args.samples_per_cell,
    )

    # Get camera sampling params (CLI overrides YAML)
    num_cameras_sampled = yaml_config.get(
        "num_cameras_sampled", args.num_cameras_sampled
    )
    ball_visibility_threshold = yaml_config.get(
        "ball_visibility_threshold", args.ball_visibility_threshold
    )

    return GeneratorConfig(
        physics=physics_config,
        shot=shot_config,
        camera=camera_config,
        sampling=sampling_config,
        num_cameras_sampled=num_cameras_sampled,
        ball_visibility_threshold=ball_visibility_threshold,
        max_attempts_per_cell=yaml_config.get("max_attempts_per_cell", 10000),
    )


def main() -> int:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BLCS Dataset Generator (PLCS-unified format)")
    logger.info("=" * 60)

    # Load config
    yaml_config = load_config(args.config)

    # Create generator config
    generator_config = create_generator_config(args, yaml_config)

    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Samples per cell: {args.samples_per_cell}")
    logger.info(f"Cameras sampled per scene: {generator_config.num_cameras_sampled}")
    logger.info(
        f"Ball visibility threshold: {generator_config.ball_visibility_threshold}"
    )
    logger.info(f"Device: {args.device}")

    # Initialize generator and writer
    generator = BLCSSceneGenerator(config=generator_config, device=args.device)
    writer = BLCSDatasetWriter(args.output_dir)

    # Generate and save scenes (1 scene = 1 file with N cameras)
    logger.info("Starting scene generation...")
    total_scenes = 0
    total_cameras = 0

    for scene_data in tqdm(
        generator.generate_all_scenes(),
        desc="Generating scenes",
    ):
        filepath = writer.save_scene(scene_data)
        total_scenes += 1
        total_cameras += len(scene_data.cameras)

        if total_scenes % 100 == 0:
            avg_cams = total_cameras / total_scenes
            logger.info(
                f"Progress: {total_scenes} scenes, "
                f"{total_cameras} cameras (avg {avg_cams:.1f}/scene)"
            )

    logger.info(f"Generation complete: {total_scenes} scenes, {total_cameras} cameras")

    # Save splits
    logger.info("Creating train/val/test splits...")
    writer.save_split_info(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed,
    )

    # Save meta.json with all scene information
    stats = generator.get_statistics()
    writer.save_meta_json(config=yaml_config)
    writer.save_dataset_info(stats)

    logger.info("=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Total scenes: {total_scenes}")
    logger.info(f"Total cameras: {total_cameras}")
    logger.info(f"Avg cameras/scene: {total_cameras / total_scenes:.2f}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
