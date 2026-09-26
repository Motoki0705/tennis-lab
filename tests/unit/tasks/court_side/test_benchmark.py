"""Benchmark trials know the true sides, perturb as declared, and share the production judgement."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tasks.court_side.benchmark import (
    Perturbation,
    SyntheticScene,
    ThresholdGrid,
    judge,
    local_calibration,
    make_trial,
    select_thresholds,
)
from src.tasks.court_side.hypothesis import CourtSideConfig
from src.utils.geometry.triangulation import PinholeCamera

CONFIG = CourtSideConfig(reprojection_px=20., min_motion_px=5., min_frames=8, max_cost=.5, min_support=.5, min_margin=.1)
CLEAN = Perturbation("clean", missing_rate=0., pixel_sigma_px=0., calibration_scale=0.)


def look_at(name: str, center: list[float]) -> PinholeCamera:
    position = np.asarray(center, np.float64)
    forward = np.array([0, 0, 1.]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[1100., 0, 640], [0, 1100, 360], [0, 0, 1]]), rotation, -rotation @ position)


def scene(frames: int = 120) -> SyntheticScene:
    cameras = (look_at("a", [-9, -18, 4]), look_at("b", [9, -18, 3]), look_at("c", [9, 18, 3]), look_at("d", [-9, 18, 4]))
    t = np.linspace(0, 1, frames)
    xyz = np.stack((3 * np.sin(6 * t), 11 * np.cos(3 * t), 1 + 2 * np.abs(np.sin(5 * t))), -1)
    return SyntheticScene("fixture", cameras, (1280, 720), xyz, np.ones((4, frames), bool), 30.)


def test_local_calibration_half_turns_only_cameras_beyond_the_net() -> None:
    far = look_at("far", [0, 18, 5])
    local, turned = local_calibration(far)
    assert turned and local.center[1] < 0
    assert not local_calibration(look_at("near", [0, -18, 5]))[1]


@pytest.mark.parametrize("cameras", [3, 4])
def test_clean_trials_are_decided_correctly_for_any_reference(cameras: int) -> None:
    rng = np.random.default_rng(1)
    trials = [make_trial(scene(), replace(CLEAN, cameras=cameras), CONFIG, rng) for _ in range(8)]
    assert len({t.expected for t in trials}) > 1  # the random reference changes the expected assignment
    assert judge(trials, CONFIG).correct == 8
    assert all(t.evidence.hypotheses[0].cost < .01 and len(t.evidence.hypotheses) == 2 ** (cameras - 1) for t in trials)


def test_declared_perturbations_degrade_the_evidence() -> None:
    rng = np.random.default_rng(2)
    missing = make_trial(scene(), replace(CLEAN, missing_rate=.9), CONFIG, rng)
    assert missing.evidence.frames < 60
    unsynchronized = make_trial(scene(), replace(CLEAN, sync_offset_frames=8), CONFIG, rng)
    assert unsynchronized.evidence.hypotheses[0].cost > .05
    false = make_trial(scene(), replace(CLEAN, false_rate=.5), CONFIG, rng)
    assert false.evidence.hypotheses[0].support < .9
    with pytest.raises(ValueError, match="needs 5"):
        make_trial(scene(), replace(CLEAN, cameras=5), CONFIG, rng)


def test_threshold_selection_prefers_no_wrong_decision_then_fewer_stops() -> None:
    rng = np.random.default_rng(3)
    trials = [make_trial(scene(), replace(CLEAN, name=name, window_frames=w), CONFIG, rng)
              for name, w in (("short", 6), ("long", 120)) for _ in range(4)]
    grid = ThresholdGrid(max_cost=(.5, .6), min_support=(.4, .5), min_margin=(.05, .1), min_frames=(2, 4, 8, 30))
    selected, table = select_thresholds(trials, grid, CONFIG)
    assert len(table) == 32 and all(row["wrong"] == 0 for row in table)
    # 6-frame windows pass with min_frames<=4; min_frames=2 and the permissive edge have no looser neighbour.
    assert (selected.min_frames, selected.max_cost, selected.min_support, selected.min_margin) == (4, .5, .5, .1)
    assert not any(row["safe"] for row in table if row["config"].min_frames == 2)
    assert judge([t for t in trials if t.condition == "short"], replace(CONFIG, min_frames=8)).stopped == {"insufficient_frames": 4}
