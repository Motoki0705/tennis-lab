"""Calibration/subset retries must preserve the complete corruption draw."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from src.tasks.base.triangulation_residual.cameras import fixed_six_camera_rig
from src.tasks.base.triangulation_residual.configuration import validate_config
from src.tasks.base.triangulation_residual.geometry import InsufficientGeometryError
from src.tasks.base.triangulation_residual.sampling_v2 import sample_v2
from src.utils.geometry.planar_camera import PlanarCameraFailure, PlanarCameraFitError


def fixture():
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/blcs/configs"), version_base="1.3"
    ):
        config = validate_config(compose(config_name="train_triangulation_residual_v2"))
    assert config.v2 is not None
    config = replace(
        config,
        data=replace(config.data, min_views=2, max_views=2),
        v2=replace(
            config.v2,
            error_mode="clean",
            evaluation_views=2,
            true_camera_position_jitter_m=0,
            true_camera_height_jitter_m=0,
        ),
        corruption=replace(config.corruption, focal_scale_min=1, focal_scale_max=1),
    )
    world: np.ndarray = np.zeros((16, 1, 3), np.float32)
    world[:, 0] = [0, -6, 1.5]
    world[:, 0, 0] = np.linspace(-1, 1, 16)
    return config, world, np.ones(16, bool), fixed_six_camera_rig((1280, 720))


def test_rejected_geometry_subset_reuses_same_full_rig_draw(monkeypatch):
    import src.tasks.base.triangulation_residual.sampling_v2 as module

    config, world, valid, rig = fixture()
    original_corrupt = module.corrupt_candidates
    original_prepare = module.prepare_geometry
    draws = []
    tried = []

    def corrupt(*args, **kwargs):
        result = original_corrupt(*args, **kwargs)
        draws.append(result)
        return result

    def prepare(*args, **kwargs):
        tried.append(args[4].centers)
        if len(tried) == 1:
            raise InsufficientGeometryError("Deliberately invisible first subset")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(module, "corrupt_candidates", corrupt)
    monkeypatch.setattr(module, "prepare_geometry", prepare)
    sample = sample_v2(
        world,
        valid,
        30,
        rig,
        np.random.default_rng(52),
        config,
        scene_id="fixed",
        split="train",
    )
    assert len(draws) == 1
    assert len(tried) == 2
    assert sample["geometry_attempts"] == 2
    assert sample["calibration_attempts"] == 6
    np.testing.assert_array_equal(sample["target_world"], world)
    np.testing.assert_allclose(sample["init_world"], world, atol=2e-5)


def test_all_calibration_failures_raise_without_camera_fallback(monkeypatch):
    import src.tasks.base.triangulation_residual.sampling_v2 as module

    config, world, valid, rig = fixture()
    failure = PlanarCameraFitError(PlanarCameraFailure.DEGENERATE_POINTS, "collinear")
    calls = []
    fixed_errors = []

    def corrupt(*args, **kwargs):
        calls.append(args[0].copy())
        fixed_errors.append(kwargs["fixed_error"])
        return SimpleNamespace(
            valid_indices=np.array([], np.int64),
            court_fit=SimpleNamespace(failures=(failure,) * 6),
            family=0,
            severity=0.0,
        )

    monkeypatch.setattr(module, "corrupt_candidates", corrupt)
    with pytest.raises(InsufficientGeometryError, match="degenerate_points.*48"):
        sample_v2(
            world,
            valid,
            30,
            rig,
            np.random.default_rng(1),
            config,
            scene_id="fixed",
            split="train",
        )
    assert len(calls) == 8
    assert fixed_errors == [None] + [(0, 0.0)] * 7
    for target in calls:
        np.testing.assert_array_equal(target, world)


def test_loss_ablation_has_exactly_identical_inputs_and_targets():
    config, world, valid, rig = fixture()
    assert config.v2 is not None
    first = sample_v2(
        world,
        valid,
        30,
        rig,
        np.random.default_rng(15),
        config,
        scene_id="fixed",
        split="val",
    )
    legacy = replace(config, v2=replace(config.v2, loss_mode="legacy"))
    second = sample_v2(
        world,
        valid,
        30,
        rig,
        np.random.default_rng(15),
        legacy,
        scene_id="fixed",
        split="val",
    )
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
