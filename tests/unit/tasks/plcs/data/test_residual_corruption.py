"""Numerical contracts for PLCS-only residual augmentation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.data.augmentation.persistent_pose import sample_persistent_events
from src.tasks.plcs.data.augmentation.residual import (
    corrupt_candidates,
    fixed_six_camera_rig,
)


def _augmentation(**changes):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = validate_residual_config(
            compose(config_name="train_triangulation_residual")
        )
    return replace(config.augmentation, **changes)


def _world(frames: int = 16) -> np.ndarray:
    world: np.ndarray = np.zeros((frames, 17, 3), dtype=np.float64)
    world[..., 1] = -5.0
    world[..., 2] = 1.2
    world[:, (11, 12), 2] = 0.9
    return world


def test_fixed_camera_and_clean_corruption_are_deterministic() -> None:
    config = _augmentation(
        error_mode="clean",
        focal_scale_min=1.0,
        focal_scale_max=1.0,
        true_camera_position_jitter_m=0.0,
        true_camera_height_jitter_m=0.0,
    )
    rig = fixed_six_camera_rig((1920, 1080))
    first = corrupt_candidates(
        _world(), rig, np.random.default_rng(9), config, fps=30.0
    )
    second = corrupt_candidates(
        _world(), rig, np.random.default_rng(9), config, fps=30.0
    )
    np.testing.assert_equal(first.observations_px, second.observations_px)
    np.testing.assert_array_equal(first.valid_indices, np.arange(6))
    assert first.family == 0


def test_persistent_schedule_preserves_seconds_across_fps() -> None:
    config = _augmentation(
        persistent_rate_per_view_second=2.0,
        persistent_min_seconds=0.4,
        persistent_max_seconds=1.2,
    )
    slow = sample_persistent_events(150, 6, np.random.default_rng(30), config, fps=15)
    fast = sample_persistent_events(600, 6, np.random.default_rng(30), config, fps=60)
    assert len(slow) == len(fast) > 10
    for left, right in zip(slow, fast, strict=True):
        assert left.start_seconds == right.start_seconds
        assert left.duration_seconds == right.duration_seconds
        assert left.kind == right.kind
        assert left.views == right.views
        assert left.joints == right.joints
