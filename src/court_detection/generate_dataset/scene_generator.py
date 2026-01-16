"""Scene generator for synthetic court keypoint data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.geometry.court import (
    court_keypoints_3d,
    make_look_at_camera,
    project_points,
    sample_camera_position_on_fence,
)

LOGGER = logging.getLogger(__name__)

NUM_KEYPOINTS = 20


@dataclass
class CameraConfig:
    """Camera sampling configuration."""

    hfov_range: tuple[float, float] = (40.0, 80.0)
    height_range: tuple[float, float] = (1.5, 4.0)
    distance_range: tuple[float, float] = (15.0, 30.0)
    pitch_noise_deg: float = 5.0
    yaw_noise_deg: float = 10.0


@dataclass
class GenerationConfig:
    """Data generation configuration."""

    num_scenes: int = 10000
    output_dir: str = "data/court_detection/scenes"
    image_size: tuple[int, int] = (1280, 720)
    render_synthetic: bool = True


class CourtSceneGenerator:
    """Generator for synthetic court keypoint training data.

    Creates random camera views of the tennis court and projects
    the 3D keypoints to 2D image coordinates.

    Args:
        camera_config: Camera sampling configuration.
        generation_config: Generation configuration.
    """

    def __init__(
        self,
        camera_config: CameraConfig | dict[str, Any] | None = None,
        generation_config: GenerationConfig | dict[str, Any] | None = None,
    ) -> None:
        if isinstance(camera_config, dict):
            camera_config = CameraConfig(**camera_config)
        self.camera_config = camera_config or CameraConfig()

        if isinstance(generation_config, dict):
            generation_config = GenerationConfig(**generation_config)
        self.generation_config = generation_config or GenerationConfig()

        # Get 3D court keypoints
        self.court_kp_3d = court_keypoints_3d()

    def generate_scene(
        self,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate a single scene with random camera view.

        Args:
            rng: Random number generator.

        Returns:
            Dictionary with:
                - 'keypoints': 2D keypoint coordinates (K, 2) normalized [0, 1]
                - 'visibility': Visibility flags (K,)
                - 'camera_params': Camera parameters dict
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample camera position
        camera_center = self._sample_camera_position(rng)

        # Sample field of view
        hfov = rng.uniform(*self.camera_config.hfov_range)

        # Add noise to look-at target
        look_at = np.array([0.0, 0.0, 0.5])
        look_at[0] += rng.uniform(-2.0, 2.0)
        look_at[1] += rng.uniform(-2.0, 2.0)

        # Create camera
        camera = make_look_at_camera(
            center=camera_center,
            look_at=look_at.tolist(),
            image_size=self.generation_config.image_size,
            hfov_deg=hfov,
        )

        # Project 3D keypoints to 2D
        uv, in_front = project_points(camera, self.court_kp_3d)

        # Determine visibility
        w, h = self.generation_config.image_size
        in_bounds = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        visibility = (in_front & in_bounds).float()

        # Normalize coordinates to [0, 1]
        keypoints_norm = uv.clone()
        keypoints_norm[:, 0] /= w
        keypoints_norm[:, 1] /= h

        # Clamp to valid range
        keypoints_norm = torch.clamp(keypoints_norm, 0.0, 1.0)

        return {
            "keypoints": keypoints_norm.numpy(),
            "visibility": visibility.numpy(),
            "camera_params": {
                "center": camera_center.tolist(),
                "look_at": look_at.tolist(),
                "hfov": hfov,
                "image_size": self.generation_config.image_size,
            },
        }

    def _sample_camera_position(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a camera position around the court."""
        # Randomly choose a side
        sides = ["near", "far", "left", "right"]
        weights = [0.4, 0.3, 0.15, 0.15]  # Bias toward near/far sides
        side = rng.choice(sides, p=weights)

        # Sample position along the side
        t = rng.uniform(0.2, 0.8)

        # Get base position on fence
        x, y, z = sample_camera_position_on_fence(t, side)

        # Add height variation
        z = rng.uniform(*self.camera_config.height_range)

        # Add distance variation (move camera further from court)
        distance_factor = rng.uniform(0.8, 1.2)
        if side in ["near", "far"]:
            y *= distance_factor
        else:
            x *= distance_factor

        return np.array([x, y, z])

    def generate_dataset(
        self,
        seed: int = 42,
    ) -> None:
        """Generate full dataset and save to disk.

        Args:
            seed: Random seed.
        """
        output_dir = Path(self.generation_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(seed)

        LOGGER.info(
            "Generating %d scenes to %s",
            self.generation_config.num_scenes,
            output_dir,
        )

        for i in range(self.generation_config.num_scenes):
            scene = self.generate_scene(rng)

            # Save scene
            scene_path = output_dir / f"scene_{i:06d}.npz"
            np.savez(
                scene_path,
                keypoints=scene["keypoints"],
                visibility=scene["visibility"],
                camera_center=scene["camera_params"]["center"],
                camera_look_at=scene["camera_params"]["look_at"],
                camera_hfov=scene["camera_params"]["hfov"],
                image_size=scene["camera_params"]["image_size"],
            )

            if (i + 1) % 1000 == 0:
                LOGGER.info("Generated %d/%d scenes", i + 1, self.generation_config.num_scenes)

        LOGGER.info("Dataset generation complete")
