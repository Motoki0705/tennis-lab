"""Load the versioned generation plan and pin all referenced inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from src.utils.geometry.probabilistic_triangulation.convergence import (
    convergence_config,
)

from .calibration import CalibrationBank, load_calibration

SOURCE_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.synthetic_sources",
    fields=(BoundaryPathField("cameras", PathRole.DATA, PathDirection.INPUT, PathKind.FILE, must_exist=True, many=True),),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CALIBRATION_BOUNDARY = NonHydraPathBoundary(
    name="ball_refiner.synthetic_calibration",
    fields=(
        BoundaryPathField("bank", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
        BoundaryPathField("report", PathRole.PROJECT, PathDirection.INPUT, PathKind.FILE, must_exist=True),
    ),
)


@dataclass(frozen=True)
class GenerationPlan:
    values: dict[str, Any]
    physics: dict[str, Any]
    rally: dict[str, Any]
    targeted: dict[str, Any]
    camera_paths: tuple[Path, ...]
    input_paths: tuple[Path, ...]
    input_hashes: dict[str, str]
    calibration: CalibrationBank

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
    if raw["schema_version"] != 2 or raw["status"] != "cpu_generator_v2":
        raise ValueError("Plan is not an executable CPU v2 recipe")
    simulation, sampling, degradation = raw["simulation"], raw["sampling"], raw["degradation"]
    if simulation["device"] != "cpu" or simulation["workers"] not in range(1, 5):
        raise ValueError("CPU generation requires 1..4 workers")
    if type(simulation["maximum_physics_attempts_per_rally"]) is not int or simulation["maximum_physics_attempts_per_rally"] < 1:
        raise ValueError("Need a finite positive physics proposal budget")
    if simulation["class"] != "src.tasks.blcs.generate_dataset.simulation.rally_simulator.RallySimulator":
        raise ValueError("Unsupported simulator")
    if (simulation["sim_fps"], simulation["native_output_fps"], sampling["fps_numerator"], sampling["fps_denominator"]) != (240, 240, 60000, 1001):
        raise ValueError("This recipe requires native 240 Hz -> 60000/1001 Hz")
    if simulation["physics_dt_seconds"] != "1/240" or sampling["physics_event_mask_radius_frames"] != 5:
        raise ValueError("Physics timestep or event exclusion differs from v1")
    if sampling["max_frames_per_rally"] < 72:
        raise ValueError("Need >=72 frames of capacity for the 64-frame gap")
    if degradation["triangulation"] != "src.utils.geometry.probabilistic_triangulation.convergence.triangulate_converged":
        raise ValueError("This recipe requires convergence-checked A/B integration")
    convergence_config(degradation["boundary_convergence"])
    calibration = degradation["calibration"]
    paths = CALIBRATION_BOUNDARY.validate(
        {key: resolver.resolve(PathRole.PROJECT, calibration[key]) for key in ("bank", "report")}, resolver=resolver,
    )
    bank_path, report_path = (paths.declared(key).path for key in ("bank", "report"))
    bank = load_calibration(bank_path, calibration["bank_sha256"])
    if sha256(report_path) != calibration["report_sha256"]:
        raise ValueError("Calibration report SHA mismatch")
    report = json.loads(report_path.read_text())
    if report["schema"] != "ball_refiner_3d.degradation_calibration.v1" or report["bank_sha256"] != calibration["bank_sha256"] or report["components"] != bank.components or report["status"] != degradation["status"]:
        raise ValueError("Calibration bank/report identity mismatch")
    if degradation["components_per_camera"] != bank.components or degradation["max_components"] != (bank.components + 1) ** 3:
        raise ValueError("Require every calibrated component and camera-subset product")
    if type(calibration["block_frames"]) is not int or not 1 <= calibration["block_frames"] <= 16:
        raise ValueError("Pilot evidence supports bootstrap blocks of 1..16 frames")
    if not -10 <= calibration["out_of_frame_presence_logit"] < 0 or degradation["max_nfev"] < 1:
        raise ValueError("Invalid explicit out-of-frame presence or optimizer budget")
    if degradation["occlusion_changes_presence"] or not degradation["out_of_frame_changes_presence"]:
        raise ValueError("Presence is amodal in-image existence, not visibility")
    if raw["geometry"]["calibration_error_condition"] != "clean_shared_perturbed_camera":
        raise ValueError("Calibration-error generation is not implemented")
    if degradation["mean_bounds_policy"] != "clip_to_source_grid":
        raise ValueError("Unsupported synthetic mean bounding policy")
    if degradation["gap_lengths_frames"] != [1, 4, 8, 16, 32, 64]:
        raise ValueError("The gap schedule must cover 1/4/8/16/32/64 frames")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("Need a nonnegative integer seed")
    counts = [raw["counts"]["smoke_rallies_per_split"], *raw["counts"]["pilot_rallies"].values(), *raw["counts"]["dev_rallies"].values()]
    if any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("Rally counts must be positive integers")
    sources = raw["geometry"]["sources"]
    if [source["split"] for source in sources] != ["train", "val", "test"]:
        raise ValueError("Need one ordered camera source per split")
    if [source["clip_id"].split("/")[0] for source in sources] != ["video_002", "video_000", "video_001"]:
        raise ValueError("Camera recording split must match #934")
    if any(source["camera_keys"] != ["cam_0_params", "cam_1_params", "cam_2_params"] for source in sources):
        raise ValueError("Camera order must match the cam0/cam1/cam2 calibration bank")
    paths = SOURCE_BOUNDARY.validate({"cameras": [source["path"] for source in sources]}, resolver=resolver)
    camera_paths = tuple(value.path for value in paths.declared_many("cameras"))
    hashes = {str(path): sha256(path)}
    input_paths = [path, *camera_paths, bank_path, report_path]
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
    return GenerationPlan(raw, physics, rally, targeted, camera_paths, tuple(input_paths), hashes, bank)
