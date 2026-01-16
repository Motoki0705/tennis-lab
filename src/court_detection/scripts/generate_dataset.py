"""Generate synthetic dataset for court keypoint detection.

This script generates training data by projecting 3D court keypoints
to 2D using random camera views.

Example:
    uv run python -m src.court_detection.scripts.generate_dataset

Config entry point: `src/court_detection/configs/generate_dataset.yaml`
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from src.court_detection.generate_dataset.scene_generator import (
    CameraConfig,
    CourtSceneGenerator,
    GenerationConfig,
)

LOGGER = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="generate_dataset",
)
def main(cfg: DictConfig) -> None:
    """Generate synthetic court keypoint dataset."""
    LOGGER.info("Starting dataset generation")

    # Build camera config
    camera_config = CameraConfig(
        hfov_range=tuple(cfg.camera.get("hfov_range", [40.0, 80.0])),
        height_range=tuple(cfg.camera.get("height_range", [1.5, 4.0])),
        distance_range=tuple(cfg.camera.get("distance_range", [15.0, 30.0])),
        pitch_noise_deg=cfg.camera.get("pitch_noise_deg", 5.0),
        yaw_noise_deg=cfg.camera.get("yaw_noise_deg", 10.0),
    )

    # Build generation config
    generation_config = GenerationConfig(
        num_scenes=cfg.generation.get("num_scenes", 10000),
        output_dir=cfg.generation.get("output_dir", "data/court_detection/scenes"),
        image_size=tuple(cfg.generation.get("image_size", [1280, 720])),
        render_synthetic=cfg.generation.get("render_synthetic", True),
    )

    # Create generator and run
    generator = CourtSceneGenerator(
        camera_config=camera_config,
        generation_config=generation_config,
    )

    generator.generate_dataset(seed=cfg.run.get("seed", 42))

    LOGGER.info("Dataset generation complete")


if __name__ == "__main__":
    main()
