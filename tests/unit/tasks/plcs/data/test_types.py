"""Tests for strict PLCS scene metadata contracts."""

from __future__ import annotations

import pytest

from src.tasks.plcs.data.types import PLCSSceneMeta
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def test_scene_metadata_requires_track_instances_field() -> None:
    metadata = {
        "scene_id": "scene",
        "motion_source": "source",
        "motion_category": "category",
        "gender": "neutral",
        "fps": 30,
        "num_frames": 2,
        "initial_position": [0.0, 0.0],
        "initial_yaw": 0.0,
        "num_cameras_sampled": 1,
        "num_cameras": 1,
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
    }

    with pytest.raises(KeyError, match="track_instances"):
        PLCSSceneMeta.from_dict(metadata)
