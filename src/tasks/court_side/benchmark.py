"""Synthetic robustness benchmark of the ball side decision and its threshold selection.

Scenes are BLCS synthetic rallies with known physical cameras. Each trial
chooses cameras and a reference, builds their camera-local calibrations
(cameras beyond the net are half-turned locally, as in ``camera_view_v2``),
perturbs calibration and observations, and stores the threshold-free
:class:`BallSideEvidence`. Thresholds are then judged on the stored evidence
by the production :func:`judge_side_evidence`, so the benchmark and the
pipeline share one decision rule.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.tasks.court_side.hypothesis import (
    BallSideEvidence,
    CourtSideConfig,
    CourtSideUndecided,
    collect_side_evidence,
    judge_side_evidence,
)
from src.utils.geometry.triangulation import PinholeCamera

# Pixel quantities are specified at 1920x1080 and scaled by image diagonal, as in the pipeline.
REFERENCE_DIAGONAL = math.hypot(1920, 1080)


@dataclass(frozen=True)
class SyntheticScene:
    """Physical cameras, the true ball and its in-image visibility at the scene frame rate."""

    scene_id: str
    cameras: tuple[PinholeCamera, ...]
    image_size: tuple[int, int]
    ball_xyz: NDArray[np.float64]  # (T,3)
    visible: NDArray[np.bool_]  # (V,T)
    fps: float


@dataclass(frozen=True)
class Perturbation:
    """One benchmark condition.

    ``false_rate`` is the fraction of each camera's window replaced by static
    false detections in segments of ``false_segment_frames``; ``false_shared``
    places every camera's segments on the same frames and the same 3D point
    (a ball held by a ball boy seen by all cameras). ``sync_offset_frames``
    delays one non-reference camera. Calibration noise is Gaussian with the
    nominal standard deviations times ``calibration_scale``.
    """

    name: str
    cameras: int = 3
    window_frames: int = 300
    missing_rate: float = 0.1
    false_rate: float = 0.0
    false_shared: bool = False
    false_segment_frames: tuple[int, int] = (5, 30)
    sync_offset_frames: int = 0
    pixel_sigma_px: float = 2.0
    calibration_scale: float = 1.0
    # Nominal calibration noise gives a median true-hypothesis cost near the 0.10 measured with
    # reviewed balls on Meiji clip_000 (knowledge run-scene-component-meiji-target2-20260925).
    rotation_sigma_deg: float = 0.05
    focal_sigma_ratio: float = 0.004
    center_sigma_m: float = 0.08


@dataclass(frozen=True)
class Trial:
    scene_id: str
    condition: str
    expected: tuple[bool, ...]
    evidence: BallSideEvidence


@dataclass(frozen=True)
class Outcome:
    correct: int = 0
    wrong: int = 0
    stopped: dict[str, int] = field(default_factory=dict)

    @property
    def trials(self) -> int:
        return self.correct + self.wrong + sum(self.stopped.values())


def local_calibration(camera: PinholeCamera) -> tuple[PinholeCamera, bool]:
    """The ``camera_view_v2`` camera-local calibration and whether it is half-turned."""
    turned = bool(camera.center[1] > 0)
    return camera.half_turned(turned), turned


def perturb_camera(camera: PinholeCamera, p: Perturbation, rng: np.random.Generator) -> PinholeCamera:
    scale = p.calibration_scale
    rotation = Rotation.from_rotvec(rng.normal(0, math.radians(p.rotation_sigma_deg) * scale, 3)).as_matrix() @ camera.rotation
    intrinsic = camera.intrinsic.copy()
    intrinsic[[0, 1], [0, 1]] *= 1 + rng.normal(0, p.focal_sigma_ratio * scale)
    center = camera.center + rng.normal(0, p.center_sigma_m * scale, 3)
    return PinholeCamera(camera.camera_id, intrinsic, rotation, -rotation @ center)


def _segments(frames: int, rate: float, lengths: tuple[int, int], rng: np.random.Generator) -> NDArray[np.bool_]:
    mask: NDArray[np.bool_] = np.zeros(frames, bool)
    target = rate * frames
    while mask.sum() < target:
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        start = int(rng.integers(0, max(frames - length, 0) + 1))
        mask[start:start + length] = True
    return mask


def _distractor(rng: np.random.Generator) -> NDArray[np.float64]:
    """A static ball outside the playing area: held behind a baseline or beside the court."""
    if rng.random() < .5:
        return np.array([rng.uniform(-6, 6), rng.choice([-1, 1]) * rng.uniform(12.5, 16), rng.uniform(0, 1.2)])
    return np.array([rng.choice([-1, 1]) * rng.uniform(5.5, 7.5), rng.uniform(-12, 12), rng.uniform(0, 1.2)])


def make_trials(scene: SyntheticScene, p: Perturbation, configs: Sequence[CourtSideConfig],
                rng: np.random.Generator) -> tuple[Trial, ...]:
    """One perturbed observation of ``scene``, scored under each scoring config."""
    views = len(scene.cameras)
    if p.cameras > views:
        raise ValueError(f"Scene {scene.scene_id} has {views} cameras; condition {p.name} needs {p.cameras}")
    chosen = rng.permutation(views)[:p.cameras]
    physical = tuple(scene.cameras[v] for v in chosen)
    width, height = scene.image_size
    pixel_scale = math.hypot(width, height) / REFERENCE_DIAGONAL
    locals_, turned = zip(*(local_calibration(c) for c in physical), strict=True)
    local = tuple(perturb_camera(c, p, rng) for c in locals_)
    reference = 0  # the permutation already randomizes which physical camera is the reference
    expected = tuple(t != turned[reference] for t in turned)

    total = scene.ball_xyz.shape[0]
    offset = p.sync_offset_frames
    length = min(p.window_frames, total - offset)
    if length < 1:
        raise ValueError(f"Scene {scene.scene_id} is shorter than the sync offset")
    start = int(rng.integers(0, total - offset - length + 1))
    frames = np.arange(start, start + length)
    source_frames = np.tile(frames, (p.cameras, 1))
    if offset:
        source_frames[int(rng.integers(1, p.cameras))] += offset  # a delayed stream shows later ball positions
    uv = np.zeros((p.cameras, length, 2))
    visible = np.zeros((p.cameras, length), bool)
    for row, (camera, view) in enumerate(zip(physical, chosen, strict=True)):
        projected, front = camera.project(scene.ball_xyz[source_frames[row]])
        uv[row] = projected + rng.normal(0, p.pixel_sigma_px * pixel_scale, projected.shape)
        visible[row] = scene.visible[view, source_frames[row]] & front
    visible &= rng.random(visible.shape) >= p.missing_rate
    if p.false_rate > 0:
        shared_mask = _segments(length, p.false_rate, p.false_segment_frames, rng) if p.false_shared else None
        shared_point = _distractor(rng)
        for row, camera in enumerate(physical):
            mask = shared_mask if shared_mask is not None else _segments(length, p.false_rate, p.false_segment_frames, rng)
            if p.false_shared or rng.random() < .5:
                point, front = camera.project(shared_point if p.false_shared else _distractor(rng))
                if not front or not (0 <= point[0] < width and 0 <= point[1] < height):
                    continue  # the distractor is outside this camera's image
            else:
                point = rng.uniform([0, 0], [width, height])  # a static ball-like pattern in the image
            uv[row, mask] = point + rng.normal(0, p.pixel_sigma_px * pixel_scale, (int(mask.sum()), 2))
            visible[row, mask] = True
    trials = []
    for config in configs:
        scaled = replace(config, reprojection_px=config.reprojection_px * pixel_scale, min_motion_px=config.min_motion_px * pixel_scale)
        evidence = collect_side_evidence(local, local[reference].camera_id, uv.astype(np.float32), visible, scaled)
        trials.append(Trial(scene.scene_id, p.name, expected, evidence))
    return tuple(trials)


def make_trial(scene: SyntheticScene, p: Perturbation, config: CourtSideConfig, rng: np.random.Generator) -> Trial:
    return make_trials(scene, p, (config,), rng)[0]


def judge(trials: Iterable[Trial], config: CourtSideConfig) -> Outcome:
    correct = wrong = 0
    stopped: dict[str, int] = {}
    for trial in trials:
        try:
            decision = judge_side_evidence(trial.evidence, config)
        except CourtSideUndecided as undecided:
            stopped[undecided.reason] = stopped.get(undecided.reason, 0) + 1
            continue
        if decision.view_half_turns == trial.expected:
            correct += 1
        else:
            wrong += 1
    return Outcome(correct, wrong, dict(sorted(stopped.items())))


@dataclass(frozen=True)
class ThresholdGrid:
    max_cost: tuple[float, ...]
    min_support: tuple[float, ...]
    min_margin: tuple[float, ...]
    min_frames: tuple[int, ...]

    def configs(self, base: CourtSideConfig) -> list[CourtSideConfig]:
        return [replace(base, max_cost=c, min_support=s, min_margin=m, min_frames=f)
                for c, s, m, f in product(self.max_cost, self.min_support, self.min_margin, self.min_frames)]


def _looser_neighbours(config: CourtSideConfig, grid: ThresholdGrid) -> list[CourtSideConfig] | None:
    """One grid step looser in each threshold, or ``None`` when a threshold sits on the permissive edge."""
    def step(values: tuple[float, ...] | tuple[int, ...], current: float, looser: int) -> Any:
        ordered = sorted(values)
        index = ordered.index(current) + looser
        return ordered[index] if 0 <= index < len(ordered) else None

    cost, support = step(grid.max_cost, config.max_cost, 1), step(grid.min_support, config.min_support, -1)
    margin, frames = step(grid.min_margin, config.min_margin, -1), step(grid.min_frames, config.min_frames, -1)
    if cost is None or support is None or margin is None or frames is None:
        return None
    return [replace(config, max_cost=cost), replace(config, min_support=support),
            replace(config, min_margin=margin), replace(config, min_frames=frames)]


def select_thresholds(trials: Sequence[Trial], grid: ThresholdGrid, base: CourtSideConfig,
                      ) -> tuple[CourtSideConfig, list[dict[str, Any]]]:
    """The lowest mean stop rate over conditions among safe thresholds.

    A grid point is *safe* when it and every one-step looser neighbour (higher
    cost, lower support/margin/frames) make no wrong decision in any
    condition, so the choice keeps one grid step of slack and never sits on the
    permissive edge of the grid. Ties prefer higher margin, frames and support
    and lower cost. Without any safe point the selection stops explicitly.
    """
    conditions = sorted({t.condition for t in trials})
    by_condition = {name: [t for t in trials if t.condition == name] for name in conditions}
    table: list[dict[str, Any]] = []
    wrong: dict[CourtSideConfig, int] = {}
    for config in grid.configs(base):
        outcomes = {name: judge(items, config) for name, items in by_condition.items()}
        wrong[config] = sum(o.wrong for o in outcomes.values())
        stop = float(np.mean([sum(o.stopped.values()) / o.trials for o in outcomes.values()]))
        table.append({"config": config, "wrong": wrong[config], "mean_stop_rate": stop})
    for row in table:
        neighbours = _looser_neighbours(row["config"], grid)
        row["safe"] = row["wrong"] == 0 and neighbours is not None and all(wrong[n] == 0 for n in neighbours)
    safe = [row for row in table if row["safe"]]
    if not safe:
        raise ValueError("No threshold in the grid is free of wrong decisions with one step of slack")
    best = min(safe, key=lambda row: (round(row["mean_stop_rate"], 9), -row["config"].min_margin,
                                      -row["config"].min_frames, -row["config"].min_support, row["config"].max_cost))
    return best["config"], table


def outcome_row(outcome: Outcome) -> dict[str, Any]:
    trials = outcome.trials
    return {"trials": trials, "correct": outcome.correct, "wrong": outcome.wrong, "stopped": outcome.stopped,
            "wrong_rate": outcome.wrong / trials, "stop_rate": sum(outcome.stopped.values()) / trials}


def load_blcs_scene(scene_dir: Path) -> SyntheticScene:
    """A ``single_object`` BLCS camera_view_v2 scene with its physical cameras."""
    from src.tasks.blcs.generate_dataset.io.dataset_io import load_scene

    scene = load_scene(scene_dir, court_keypoint_contract="camera_view_v2")
    if scene["num_balls"] != 1:
        raise ValueError(f"{scene_dir} is not a single-ball scene")
    cameras: list[PinholeCamera] = []
    visible: list[NDArray[np.bool_]] = []
    sizes = set()
    for data in scene["cameras"]:
        params = data["params"]
        rotation = np.asarray(params["R"], np.float64)
        intrinsic = np.array([[params["f"], 0, params["cx"]], [0, params["f"], params["cy"]], [0, 0, 1]], np.float64)
        cameras.append(PinholeCamera(data["camera_id"], intrinsic, rotation, -rotation @ np.asarray(params["C"], np.float64)))
        visible.append(np.asarray(data["ball_vis"], bool))
        sizes.add((int(params["w"]), int(params["h"])))
    if len(sizes) != 1:
        raise ValueError(f"{scene_dir} mixes image sizes")
    return SyntheticScene(scene_dir.name, tuple(cameras), sizes.pop(), np.asarray(scene["ball_pos_world"], np.float64),
                          np.stack(visible), float(scene["meta"]["fps_out"]))
