"""Load the versioned generation plan and pin all referenced inputs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

from src.utils.configuration import (
    BoundaryPathField,
    NonHydraPathBoundary,
    PathDirection,
    PathKind,
    PathResolver,
    PathRole,
)

SOURCE_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.synthetic_sources",
    fields=(BoundaryPathField("cameras", PathRole.DATA, PathDirection.INPUT, PathKind.FILE, must_exist=True, many=True),),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GenerationPlan:
    values: dict[str, Any]
    physics: dict[str, Any]
    rally: dict[str, Any]
    targeted: dict[str, Any]
    camera_paths: tuple[Path, ...]
    input_paths: tuple[Path, ...]
    input_hashes: dict[str, str]

    def verify_inputs(self) -> None:
        for path in self.input_paths:
            if sha256(path) != self.input_hashes[str(path)]:
                raise ValueError(f"Generation input changed: {path}")


def load_plan(path: Path, resolver: PathResolver) -> GenerationPlan:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "status", "purpose", "seed", "simulation", "sampling",
        "geometry", "degradation", "counts", "storage", "acceptance_checks",
    }:
        raise ValueError("Unknown or incomplete generation plan")
    if raw["schema_version"] != 1 or raw["status"] != "cpu_generator_v1":
        raise ValueError("Plan is not an executable CPU v1 recipe")
    simulation, sampling, degradation = raw["simulation"], raw["sampling"], raw["degradation"]
    if simulation["device"] != "cpu" or simulation["workers"] not in range(1, 5):
        raise ValueError("CPU generation requires 1..4 workers")
    if simulation["class"] != "src.tasks.blcs.generate_dataset.simulation.rally_simulator.RallySimulator":
        raise ValueError("Unsupported simulator")
    if (simulation["sim_fps"], simulation["native_output_fps"], sampling["fps_numerator"], sampling["fps_denominator"]) != (240, 240, 60000, 1001):
        raise ValueError("This recipe requires native 240 Hz -> 60000/1001 Hz")
    if simulation["physics_dt_seconds"] != "1/240" or sampling["physics_event_mask_radius_frames"] != 5:
        raise ValueError("Physics timestep or event exclusion differs from v1")
    if sampling["max_frames_per_rally"] < 72:
        raise ValueError("Need >=72 frames of capacity for the 64-frame gap")
    if degradation["components_per_camera"] != 3 or degradation["max_components"] != 64:
        raise ValueError("Require K=3 and all (K+1)^3=64 components")
    if degradation["occlusion_changes_presence"] or not degradation["out_of_frame_changes_presence"]:
        raise ValueError("Presence is amodal in-image existence, not visibility")
    if raw["geometry"]["calibration_error_condition"] != "clean_shared_perturbed_camera":
        raise ValueError("Calibration-error generation is not implemented")
    for key in ("source_pixel_sigma_range", "correlation_range"):
        bounds = np.asarray(degradation[key], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid {key}")
    if degradation["source_pixel_sigma_range"][0] <= 0 or not all(-1 < v < 1 for v in degradation["correlation_range"]):
        raise ValueError("Invalid 2D covariance range")
    if not 0 <= degradation["error_ar1"] < 1 or degradation["max_nfev"] < 1:
        raise ValueError("Invalid AR1 or optimizer budget")
    for key in ("observed_weights", "gap_weights"):
        weights = np.asarray(degradation[key], dtype=float)
        if weights.shape != (3,) or not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError(f"Invalid {key}")
    if degradation["mean_bounds_policy"] != "clip_to_source_grid":
        raise ValueError("Unsupported synthetic mean bounding policy")
    if degradation["gap_lengths_frames"] != [1, 4, 8, 16, 32, 64]:
        raise ValueError("The v1 gap schedule must cover 1/4/8/16/32/64 frames")
    if not 0 < degradation["presence_logit_magnitude"] <= 10:
        raise ValueError("Presence must remain interior for full enumeration")
    if any(not np.isfinite(degradation[key]) or degradation[key] < 1 for key in ("gap_sigma_multiplier", "distractor_sigma_multiplier")):
        raise ValueError("Covariance multipliers must be finite and >=1")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("Need a nonnegative integer seed")
    counts = [raw["counts"]["smoke_rallies_per_split"], *raw["counts"]["pilot_rallies"].values()]
    if any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("Rally counts must be positive integers")
    sources = raw["geometry"]["sources"]
    if [source["split"] for source in sources] != ["train", "val", "test"]:
        raise ValueError("Need one ordered camera source per split")
    if [source["clip_id"].split("/")[0] for source in sources] != ["video_002", "video_000", "video_001"]:
        raise ValueError("Camera recording split must match #934")
    if any(len(source["camera_keys"]) != 3 or len(set(source["camera_keys"])) != 3 for source in sources):
        raise ValueError("Need three distinct camera keys")
    paths = SOURCE_BOUNDARY.validate({"cameras": [source["path"] for source in sources]}, resolver=resolver)
    camera_paths = tuple(value.path for value in paths.declared_many("cameras"))
    hashes = {str(path): sha256(path)}
    input_paths = [path, *camera_paths]
    for source, camera_path in zip(sources, camera_paths, strict=True):
        hashes[str(camera_path)] = sha256(camera_path)
        if hashes[str(camera_path)] != source["sha256"]:
            raise ValueError(f"Camera SHA mismatch: {camera_path}")
    bases = {}
    for key in ("physics", "rally", "targeted_velocity"):
        base_path = resolver.resolve(PathRole.PROJECT, simulation[f"{key}_base"])
        hashes[str(base_path)] = sha256(base_path)
        input_paths.append(base_path)
        bases[key] = OmegaConf.load(base_path)
    composed = OmegaConf.create(bases)
    expanded = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(expanded, dict):
        raise TypeError("Simulator bases must be mappings")
    physics, rally, targeted = (dict(expanded[key]) for key in ("physics", "rally", "targeted_velocity"))
    physics["dt"] = 1 / simulation["sim_fps"]
    rally["sim_fps"] = simulation["sim_fps"]
    rally["output_fps"] = simulation["native_output_fps"]
    # Simulate only the prefix we will store, with two interpolation guard samples.
    rally["max_total_frames"] = int(np.ceil(sampling["max_frames_per_rally"] * 240 * 1001 / 60000)) + 2
    for directory in (
        "src/tasks/ball_refiner/refiner_3d/synthetic",
        "src/tasks/blcs/generate_dataset/simulation",
        "src/utils/geometry/probabilistic_triangulation",
    ):
        input_paths.extend(sorted(resolver.resolve(PathRole.PROJECT, directory).glob("*.py")))
    for name in (
        "src/tasks/ball_refiner/refiner_3d/triangulation.py",
        "src/tasks/ball_refiner/refiner_2d/distribution.py",
        "src/tasks/ball_refiner/scripts/generate_synthetic_3d.py",
        "src/utils/geometry/triangulation.py",
        "src/utils/schema/court.py",
    ):
        input_paths.append(resolver.resolve(PathRole.PROJECT, name))
    hashes.update({str(value): sha256(value) for value in input_paths})
    return GenerationPlan(raw, physics, rally, targeted, camera_paths, tuple(input_paths), hashes)
