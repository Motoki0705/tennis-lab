"""BLCS physics -> one immutable rally store shared by 2D and 3D training."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from src.tasks.ball_refiner.coordinates.config import parse_section, resolved_config
from src.tasks.base.generate_dataset.parallel_runner import (
    run_parallel_scene_generation,
)
from src.tasks.blcs.generate_dataset.config import build_generator_config
from src.tasks.blcs.generate_dataset.physics_retry import (
    generate_with_bounded_physics_resampling,
)
from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
from src.tasks.blcs.generate_dataset.simulation.cell_manager import CellManager
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import (
    RallyResult,
    RallySimulator,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.configuration import PathRole
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.projection import make_look_at_camera
from src.utils.schema.court import (
    BASELINE_CLEAR,
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    SIDELINE_CLEAR,
)

SCHEMA = "ball_refiner.single_object.v1"


@dataclass(frozen=True)
class CameraSampling:
    views: int
    width: int
    height: int
    z_min: float
    z_max: float
    hfov_min: float
    hfov_max: float
    look_x: float
    look_y: float
    look_z_min: float
    look_z_max: float
    maximum_attempts: int

    def __post_init__(self) -> None:
        if self.views < 2 or self.views % 2 or min(self.width, self.height) < 2:
            raise ValueError("Use an even camera count >=2 and valid image dimensions")
        if not 0 < self.z_min <= self.z_max or not 0 < self.hfov_min <= self.hfov_max < 170:
            raise ValueError("Invalid height/field-of-view ranges")
        if min(self.look_x, self.look_y, self.look_z_min) < 0 or self.look_z_min > self.look_z_max:
            raise ValueError("Invalid look-at ranges")
        if self.maximum_attempts < 1:
            raise ValueError("Camera sampling requires a finite positive attempt budget")


@dataclass(frozen=True)
class GenerationConfig:
    dataset: str
    train_rallies: int
    val_rallies: int
    test_rallies: int
    seed: int
    workers: int
    min_frames: int
    maximum_attempts: int

    def __post_init__(self) -> None:
        if min(self.train_rallies, self.val_rallies, self.test_rallies, self.workers, self.min_frames, self.maximum_attempts) < 1 or self.seed < 0:
            raise ValueError("Require positive generation counts and nonnegative seed")


def generation_config(config: DictConfig) -> tuple[dict[str, Any], Path, GenerationConfig, CameraSampling, GeneratorConfig]:
    raw, resolver = resolved_config(config, {
        "paths", "generation", "camera_sampling", "physics", "rally", "camera",
        "generator", "targeted_velocity", "court_keypoints",
    })
    generation = parse_section(GenerationConfig, raw["generation"])
    camera = parse_section(CameraSampling, raw["camera_sampling"])
    physics = build_generator_config(config)
    if physics.rally.sim_fps % physics.rally.output_fps or physics.rally.output_fps <= 0:
        raise ValueError("BLCS simulation FPS must be divisible by output FPS")
    if not np.isclose(physics.physics.dt, 1 / physics.rally.sim_fps):
        raise ValueError("Physics dt must match simulation FPS")
    raw["paths"] = {name: str(value) for name, value in asdict(resolver.roots).items()}
    return raw, resolver.resolve(PathRole.DATA, generation.dataset), generation, camera, physics


def validate_generation(config: DictConfig) -> None:
    generation_config(config)


def sample_cameras(config: CameraSampling, rng: np.random.Generator) -> tuple[PinholeCamera, ...]:
    result = []
    for index in range(config.views):
        # Same fence planes for every clip, continuous X/Z within those planes.
        center = (rng.uniform(-HALF_DOUBLES_WIDTH - SIDELINE_CLEAR, HALF_DOUBLES_WIDTH + SIDELINE_CLEAR),
                  (-1 if index % 2 == 0 else 1) * (HALF_LENGTH + BASELINE_CLEAR), rng.uniform(config.z_min, config.z_max))
        look = (rng.uniform(-config.look_x, config.look_x), rng.uniform(-config.look_y, config.look_y), rng.uniform(config.look_z_min, config.look_z_max))
        camera = make_look_at_camera(center, look_at=look, image_size=(config.width, config.height), hfov_deg=rng.uniform(config.hfov_min, config.hfov_max))
        intrinsic = np.array([[camera.f, 0, camera.cx], [0, camera.f, camera.cy], [0, 0, 1]], dtype=np.float64)
        rotation = camera.R.numpy().astype(np.float64)
        translation = -rotation @ camera.C.numpy().astype(np.float64)
        result.append(PinholeCamera(f"cam{index}", intrinsic, rotation, translation))
    return tuple(result)


def sample_visible_cameras(config: CameraSampling, xyz: np.ndarray, rng: np.random.Generator) -> tuple[PinholeCamera, ...] | None:
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
            if front.all() and (uv >= 0).all() and (uv[:, 0] < config.width).all() and (uv[:, 1] < config.height).all():
                accepted[index] = camera
        if len(accepted) == config.views:
            return tuple(accepted[index] for index in range(config.views))
    return None


def event_frames(result: RallyResult, stride: int = 1, frames: int | None = None) -> NDArray[np.uint8]:
    """Discard hypothetical later bounces after a shot was actually returned."""
    events: NDArray[np.uint8] = np.zeros(frames if frames is not None else len(result.trajectory[::stride]), dtype=np.uint8)
    for index, shot in enumerate(result.shot_events):
        end = result.shot_events[index + 1].t_start if index + 1 < len(result.shot_events) else len(result.trajectory)
        if shot.t_return >= 0:
            end = min(end, shot.t_return + 1)
        start_frame = int(np.floor(shot.t_start / stride + 0.5))
        if 0 <= start_frame < len(events):
            events[start_frame] |= 1
        for bounce in (shot.t_bounce1, shot.t_bounce2, shot.t_bounce3):
            if shot.t_start <= bounce < end:
                frame = int(np.floor(bounce / stride + 0.5))
                if frame < len(events):
                    events[frame] |= 2
    return events


def _generate_rally(index: int, root: Path, generation: GenerationConfig, camera: CameraSampling, physics: GeneratorConfig, split: tuple[str, ...]) -> dict[str, Any]:
    torch.set_num_threads(1)
    seed = generation.seed + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    attempts = 0
    stride = physics.rally.sim_fps // physics.rally.output_fps
    # Extract event ownership at native physics resolution, before FPS rounding.
    native_rally = replace(physics.rally, output_fps=physics.rally.sim_fps)

    def proposal() -> tuple[RallyResult, np.ndarray, tuple[PinholeCamera, ...]] | None:
        nonlocal attempts
        attempts += 1
        simulator = RallySimulator(physics_config=physics.physics.sample(), rally_config=native_rally,
                                   cell_manager=CellManager(), targeted_velocity_config=physics.targeted_velocity, device="cpu")
        rally = simulator.generate_rally(from_cell=int(rng.integers(9)), from_side="near" if rng.random() < 0.5 else "far")
        xyz = rally.trajectory[::stride].numpy().astype(np.float32)
        outside = (np.abs(xyz[:, 0]) > HALF_DOUBLES_WIDTH + SIDELINE_CLEAR) | (np.abs(xyz[:, 1]) > HALF_LENGTH + BASELINE_CLEAR)
        if outside.any():
            xyz = xyz[:int(np.flatnonzero(outside)[0])]
        if len(xyz) < generation.min_frames:
            return None
        cameras = sample_visible_cameras(camera, xyz, rng)
        if cameras is None:
            return None
        return rally, xyz, cameras

    name = f"rally_{index:06d}"
    result, xyz, cameras = generate_with_bounded_physics_resampling(proposal, scene_id=name, maximum_attempts=generation.maximum_attempts)
    points, visible = [], []
    for view in cameras:
        uv, front = view.project(xyz)
        if not front.all():
            # Keep only projection-finite trajectories: never fabricate GT behind a camera.
            raise ValueError(f"{name}: trajectory passed behind a rear-fence camera")
        points.append(uv.astype(np.float32))
        visible.append(front & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height))
    arrays = {
        "xyz_m": xyz, "uv_px": np.stack(points), "visible": np.stack(visible),
        "events": event_frames(result, stride, len(xyz)), "time_s": np.arange(len(xyz), dtype=np.float64) / physics.rally.output_fps,
        "projection": np.stack([view.matrix for view in cameras]),
        "camera_centers": np.stack([view.center for view in cameras]),
        "intrinsic": np.stack([view.intrinsic for view in cameras]),
        "rotation": np.stack([view.rotation for view in cameras]),
        "translation": np.stack([view.translation for view in cameras]),
    }
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite physical trajectory: {name}")
    path = root / "rallies" / f"{name}.npz"
    with path.with_suffix(".partial").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".partial").replace(path)
    metadata = {
        "id": name, "split": split[index], "seed": seed, "path": f"rallies/{name}.npz",
        "frames": len(xyz), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attempts": attempts, "shots": len(result.shot_events), "event_frames": int(np.count_nonzero(arrays["events"])),
        "end_reason": result.end_reason.value, "visible_fraction": float(np.mean(visible)),
        "shot_metadata_fps": physics.rally.sim_fps,
        "shot_metadata": [{key: getattr(shot, key) for key in ("shot_index", "t_start", "t_return", "t_bounce1", "t_bounce2", "t_bounce3", "shot_type", "return_type")} for shot in result.shot_events],
    }
    write_json_atomic(path.with_suffix(".json"), metadata)
    return metadata


def generate_dataset(config: DictConfig) -> Path:
    raw, root, generation, camera, physics = generation_config(config)
    if root.exists():
        raise FileExistsError(f"Dataset output already exists: {root}")
    (root / "rallies").mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(raw), root / "config.yaml")
    write_json_atomic(root / "state.json", {"status": "generating"})
    splits = np.array(["train"] * generation.train_rallies + ["val"] * generation.val_rallies + ["test"] * generation.test_rallies)
    np.random.default_rng(generation.seed).shuffle(splits)
    start = time.monotonic()
    records = []
    args = (root, generation, camera, physics, tuple(str(s) for s in splits))
    indices = list(range(len(splits)))
    iterator = (map(lambda i: _generate_rally(i, *args), indices) if generation.workers == 1 else
                run_parallel_scene_generation(_generate_rally, indices, *args, num_workers=generation.workers))
    for record in iterator:
        records.append(record)
        if len(records) % 25 == 0 or len(records) == len(indices):
            print(json.dumps({"generated": len(records), "total": len(indices), "seconds": round(time.monotonic() - start, 1)}), flush=True)
    manifest = {
        "schema": SCHEMA, "fps": physics.rally.output_fps, "image_size_wh": [camera.width, camera.height],
        "views": camera.views, "event_bits": {"shot": 1, "bounce": 2}, "records": records,
        "split_policy": "seeded rally-disjoint; every view and temporal window follows its parent rally",
        "selection_policy": "truncate at first fence exit; bounded resampling of short/no-full-rally-visible-camera/full-physics-rejected rallies; attempts in each record",
        "event_policy": "native simulation-frame ownership before nearest output-frame mapping; hypothetical post-return bounces excluded",
        "camera_policy": "fixed within rally; continuous X/Z on both baseline-rear fence planes; clean trajectory fully in frame before occlusion/noise",
        "generation_seconds": time.monotonic() - start,
    }
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "state.json", {"status": "complete", "rallies": len(records)})
    print(json.dumps({"dataset": str(root), "rallies": len(records), "frames": sum(r["frames"] for r in records)}), flush=True)
    return root
