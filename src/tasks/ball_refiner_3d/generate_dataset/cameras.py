"""Rear-fence multiview camera sampling."""

from __future__ import annotations

import numpy as np

from src.tasks.ball_refiner_3d.configuration.generation import CameraSampling
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.projection import make_look_at_camera
from src.utils.schema.court import (
    BASELINE_CLEAR,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    SIDELINE_CLEAR,
)


def sample_cameras(
    config: CameraSampling, rng: np.random.Generator
) -> tuple[PinholeCamera, ...]:
    result = []
    for index in range(config.views):
        # Same fence planes for every clip, continuous X/Z within those planes.
        center = (
            rng.uniform(
                -HALF_DOUBLES_WIDTH - SIDELINE_CLEAR,
                HALF_DOUBLES_WIDTH + SIDELINE_CLEAR,
            ),
            (-1 if index % 2 == 0 else 1) * (HALF_LENGTH + BASELINE_CLEAR),
            rng.uniform(config.z_min, config.z_max),
        )
        look = (
            rng.uniform(-config.look_x, config.look_x),
            rng.uniform(-config.look_y, config.look_y),
            rng.uniform(config.look_z_min, config.look_z_max),
        )
        camera = make_look_at_camera(
            center,
            look_at=look,
            image_size=(config.width, config.height),
            hfov_deg=rng.uniform(config.hfov_min, config.hfov_max),
        )
        intrinsic = np.array(
            [[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]],
            dtype=np.float64,
        )
        rotation = camera.R.numpy().astype(np.float64)
        translation = -rotation @ camera.C.numpy().astype(np.float64)
        result.append(PinholeCamera(f"cam{index}", intrinsic, rotation, translation))
    return tuple(result)


def sample_visible_cameras(
    config: CameraSampling, xyz: np.ndarray, rng: np.random.Generator
) -> tuple[PinholeCamera, ...] | None:
    """Condition camera choice on clean full-rally visibility, before corruption.

    Otherwise near-plane projections can create tens of thousands of pixels of
    ground truth, making an occlusion benchmark primarily an offscreen task.
    No coordinates or noise are clipped to pass this selection.
    """
    accepted: dict[int, PinholeCamera] = {}
    for _ in range(config.maximum_attempts):
        for index, camera in enumerate(sample_cameras(config, rng)):
            if index in accepted:
                continue
            uv, front = camera.project(xyz)
            if (
                front.all()
                and (uv >= 0).all()
                and (uv[:, 0] < config.width).all()
                and (uv[:, 1] < config.height).all()
            ):
                accepted[index] = camera
        if len(accepted) == config.views:
            return tuple(accepted[index] for index in range(config.views))
    return None
