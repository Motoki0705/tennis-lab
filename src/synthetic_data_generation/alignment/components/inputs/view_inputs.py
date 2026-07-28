"""Shared fit/holdout camera and provider-image input boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.scene_contract import SceneCamera


def partition_fit_and_holdout_cameras(
    cameras: Sequence[SceneCamera],
    *,
    holdout_group_ids: Sequence[int],
) -> tuple[tuple[SceneCamera, ...], tuple[SceneCamera, ...]]:
    """Partition cameras by immutable provider group without opening images."""
    holdout_groups = set(holdout_group_ids)
    fit = tuple(camera for camera in cameras if camera.group_id not in holdout_groups)
    holdout = tuple(camera for camera in cameras if camera.group_id in holdout_groups)
    return fit, holdout


def load_provider_rgb_image(path: Path) -> NDArray[np.uint8]:
    """Decode one provider image explicitly as uint8 RGB."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return rgb
