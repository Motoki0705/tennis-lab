from dataclasses import replace
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.data.augmentation.persistent_pose import (
    PersistentKind,
    corrupt_persistent,
    sample_persistent_events,
)


def _config(**changes):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = validate_residual_config(
            compose(config_name="train_triangulation_residual")
        )
    return replace(config.augmentation, **changes)


def test_schedule_is_repeatable_and_uses_plcs_distal_joints() -> None:
    config = _config(persistent_rate_per_view_second=4.0)
    first = sample_persistent_events(1800, 6, np.random.default_rng(48), config, fps=30)
    second = sample_persistent_events(
        1800, 6, np.random.default_rng(48), config, fps=30
    )
    assert first == second
    assert {event.kind for event in first} == {
        PersistentKind.SELF_OFFSET,
        PersistentKind.CONTRALATERAL,
        PersistentKind.FROZEN_JOINT,
    }
    assert {joint for event in first for joint in event.joints} <= {7, 8, 9, 10, 15, 16}


def test_missing_source_is_explicit_and_never_restored() -> None:
    points = np.full((2, 30, 17, 2), [900.0, 500.0])
    visible = np.ones(points.shape[:-1], dtype=bool)
    visible[:, :, (7, 8, 9, 10, 15, 16)] = False
    points[:, :, (7, 8, 9, 10, 15, 16)] = np.nan
    result = corrupt_persistent(
        points,
        visible.astype(float),
        points,
        visible,
        np.tile([1920, 1080], (2, 1)),
        np.random.default_rng(3),
        _config(persistent_rate_per_view_second=0.0),
        fps=30,
        force_event=True,
    )
    assert result.source_missing.any()
    assert np.all(result.scores[result.source_missing] == 0)
