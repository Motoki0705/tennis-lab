from dataclasses import replace
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.data.augmentation.residual import fixed_six_camera_rig
from src.tasks.plcs.data.residual_dataset import sample_residual_window
from src.tasks.plcs.geometry.residual_features import InsufficientGeometryError


def test_sampling_is_deterministic_and_audits_six_candidates() -> None:
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = validate_residual_config(
            compose(config_name="train_triangulation_residual")
        )
    config = replace(
        config,
        data=replace(config.data, min_views=2, max_views=2),
        augmentation=replace(
            config.augmentation,
            error_mode="clean",
            evaluation_views=2,
            focal_scale_min=1.0,
            focal_scale_max=1.0,
            true_camera_position_jitter_m=0.0,
            true_camera_height_jitter_m=0.0,
        ),
    )
    world: np.ndarray = np.zeros((16, 17, 3), np.float32)
    world[..., 1] = -5.0
    world[..., 2] = 1.2
    world[:, (11, 12), 2] = 0.9
    first = sample_residual_window(
        world,
        np.ones(16, bool),
        30.0,
        fixed_six_camera_rig((1280, 720)),
        np.random.default_rng(52),
        config,
        scene_id="fixed",
        split="val",
    )
    second = sample_residual_window(
        world,
        np.ones(16, bool),
        30.0,
        fixed_six_camera_rig((1280, 720)),
        np.random.default_rng(52),
        config,
        scene_id="fixed",
        split="val",
    )
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_equal(first[key], second[key])
    assert first["calibration_attempts"] == 6
    assert first["num_views"] == 2


def test_rejected_subset_reuses_the_full_corruption_draw(monkeypatch) -> None:
    import src.tasks.plcs.data.residual_dataset as module

    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = validate_residual_config(
            compose(config_name="train_triangulation_residual")
        )
    config = replace(
        config,
        data=replace(config.data, min_views=2, max_views=2),
        augmentation=replace(
            config.augmentation,
            error_mode="clean",
            evaluation_views=2,
            focal_scale_min=1.0,
            focal_scale_max=1.0,
            true_camera_position_jitter_m=0.0,
            true_camera_height_jitter_m=0.0,
        ),
    )
    world: np.ndarray = np.zeros((16, 17, 3), np.float32)
    world[..., 1], world[..., 2] = -5.0, 1.2
    world[:, (11, 12), 2] = 0.9
    original_corrupt, original_prepare = (
        module.corrupt_candidates,
        module.prepare_geometry,
    )
    draws: list[object] = []
    attempts = 0

    def corrupt(*args, **kwargs):
        result = original_corrupt(*args, **kwargs)
        draws.append(result)
        return result

    def prepare(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InsufficientGeometryError("reject first subset")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(module, "corrupt_candidates", corrupt)
    monkeypatch.setattr(module, "prepare_geometry", prepare)
    sample = sample_residual_window(
        world,
        np.ones(16, bool),
        30.0,
        fixed_six_camera_rig((1280, 720)),
        np.random.default_rng(52),
        config,
        scene_id="fixed",
        split="val",
    )
    assert len(draws) == 1
    assert sample["geometry_attempts"] == 2
    assert sample["calibration_attempts"] == 6
