"""
Freeze B00 court-alignment gates using fit images only.

Usage:
    python -m src.synthetic_data_generation.scripts.calibrate_court_alignment
    python -m src.synthetic_data_generation.scripts.calibrate_court_alignment device=cuda:0

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/calibrate_court_alignment.yaml`.
    - Holdout groups are partitioned before image decode and remain uninferred.
    - The published artifact freezes the only gates allowed in C05 validation.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytorch_lightning
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.synthetic_data_generation.alignment.court_line_acceptance import (
    ALIGNMENT_CALIBRATION_SCHEMA,
    CourtLineEvaluationSettings,
    evaluate_projected_court_lines,
    point_cloud_court_support,
    publish_alignment_artifact,
    transform_stability,
)
from src.synthetic_data_generation.alignment.court_template_fit import (
    CourtLocalRefitSettings,
    fit_court_instance_near_reference,
    load_court_geometry_artifact,
)
from src.synthetic_data_generation.alignment.ground_line_map import (
    GroundLineMapSettings,
    load_ground_line_map_artifact,
)
from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.synthetic_data_generation.alignment.line_evidence_collection import (
    collect_projected_line_evidence,
)
from src.synthetic_data_generation.alignment.line_inference import (
    load_verified_line_detector,
)
from src.synthetic_data_generation.alignment.view_inputs import (
    partition_fit_and_holdout_cameras,
)
from src.synthetic_data_generation.provider.bundle import (
    load_scene_provider_bundle,
    sha256_file,
)
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="calibrate_court_alignment",
)
def main(cfg: DictConfig) -> int:
    """Calibrate fit-only gates and publish them before holdout inference."""
    repo_root = Path(to_absolute_path(".")).resolve()
    provider_path = _path(cfg.provider_bundle)
    line_path = _path(cfg.ground_line_artifact)
    geometry_path = _path(cfg.geometry_artifact)
    if str(sha256_file(geometry_path)) != str(cfg.geometry_file_sha256):
        raise ValueError("Geometry artifact file SHA-256 mismatch.")
    line_manifest, line_arrays = load_ground_line_map_artifact(line_path)
    geometry = load_court_geometry_artifact(geometry_path)
    if line_manifest["artifact_fingerprint"] != str(cfg.ground_line_fingerprint):
        raise ValueError("Ground-line artifact fingerprint mismatch.")
    if geometry["artifact_fingerprint"] != str(cfg.geometry_fingerprint):
        raise ValueError("Geometry artifact fingerprint mismatch.")
    if line_manifest["split"]["holdout_inference_status"] != "not_run":
        raise ValueError("C04 ground-line holdout must remain uninferred.")

    bundle = load_scene_provider_bundle(
        provider_path,
        verify_files=bool(cfg.verify_provider_files),
    )
    holdout_groups = tuple(int(value) for value in cfg.holdout_group_ids)
    fit_cameras, holdout_cameras = partition_fit_and_holdout_cameras(
        bundle.manifest.cameras,
        holdout_group_ids=holdout_groups,
    )
    if [camera.camera_id for camera in fit_cameras] != line_manifest["split"][
        "fit_camera_ids"
    ]:
        raise ValueError("Calibration fit camera ids differ from C04.")
    if [camera.camera_id for camera in holdout_cameras] != line_manifest["split"][
        "holdout_camera_ids"
    ]:
        raise ValueError("Calibration holdout camera ids differ from C04.")

    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    detector = load_verified_line_detector(
        _path(cfg.line_checkpoint),
        checkpoint_sha256=str(cfg.line_checkpoint_sha256),
        backbone_repository=_path(cfg.backbone_repository),
        backbone_checkpoint=_path(cfg.backbone_checkpoint),
        backbone_checkpoint_sha256=str(cfg.backbone_checkpoint_sha256),
        device=str(cfg.device),
        expected_short_side=int(cfg.expected_short_side),
    )
    plane = GroundPlaneEstimate(**line_manifest["ground_plane"]["estimate"])
    bounds = cast(
        tuple[float, float, float, float],
        tuple(float(value) for value in line_manifest["projection"]["bounds_uv"]),
    )
    projection_settings = GroundLineMapSettings(
        **line_manifest["projection"]["settings"]
    )
    collected = collect_projected_line_evidence(
        fit_cameras,
        bundle=bundle,
        detector=detector,
        plane=plane,
        bounds=bounds,
        settings=projection_settings,
    )

    selected = _selected_candidate(geometry)
    court_from_scene = np.asarray(selected["court_from_scene"]).reshape(4, 4)
    scene_from_court = np.asarray(selected["scene_from_court"]).reshape(4, 4)
    evaluation_settings = CourtLineEvaluationSettings(
        **_mapping(cfg.evaluation, name="evaluation")
    )
    aggregate_metrics = evaluate_projected_court_lines(
        collected.points_scene,
        collected.weights,
        court_from_scene=court_from_scene,
        settings=evaluation_settings,
    )
    group_metrics = {
        str(group_id): evaluate_projected_court_lines(
            collected.points_by_group[group_id],
            collected.weights_by_group[group_id],
            court_from_scene=court_from_scene,
            settings=evaluation_settings,
        )
        for group_id in sorted(collected.points_by_group)
    }
    local_refit_settings = CourtLocalRefitSettings(
        **_mapping(cfg.local_refit, name="local_refit")
    )
    stability_records = _fit_subset_stability(
        cfg.stability_subsets,
        evidence_by_group=collected.evidence_by_group,
        reference_candidate=selected,
        reference_scene_from_court=scene_from_court,
        bounds=bounds,
        grid_spacing=projection_settings.grid_spacing,
        plane=plane,
        settings=local_refit_settings,
    )
    point_support = point_cloud_court_support(
        np.load(bundle.point_cloud_path(), allow_pickle=False),
        court_from_scene=court_from_scene,
        settings=evaluation_settings,
    )
    accepted_count = sum(record["accepted"] for record in collected.records)
    fit_metrics = {
        "camera_count": len(fit_cameras),
        "accepted_view_count": accepted_count,
        "accepted_view_fraction": accepted_count / len(fit_cameras),
        "projected_line_pixel_count": int(
            sum(record["projected_line_pixel_count"] for record in collected.records)
        ),
        "aggregate": aggregate_metrics,
        "by_group": group_metrics,
        "view_records": list(collected.records),
        "c04_aggregate_evidence_sha256": line_manifest["files"]["arrays"]["sha256"],
        "c04_nonzero_raster_cells": int(np.count_nonzero(line_arrays["view_count"])),
    }
    fit_gates = _mapping(cfg.fit_gates, name="fit_gates")
    gate_results = _fit_gate_results(
        fit_metrics,
        stability_records,
        point_support,
        fit_gates,
    )
    status = (
        "fit_calibration_passed"
        if all(gate_results.values())
        else "fit_calibration_failed"
    )
    code_files = (
        repo_root / "src/synthetic_data_generation/alignment/court_line_acceptance.py",
        repo_root
        / "src/synthetic_data_generation/alignment/line_evidence_collection.py",
        repo_root / "src/synthetic_data_generation/alignment/line_inference.py",
        repo_root
        / "src/synthetic_data_generation/scripts/calibrate_court_alignment.py",
        repo_root
        / "src/synthetic_data_generation/configs/calibrate_court_alignment.yaml",
    )
    payload = {
        "schema": ALIGNMENT_CALIBRATION_SCHEMA,
        "artifact_id": str(cfg.artifact_id),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provider": {
            "bundle_id": bundle.manifest.bundle_id,
            "bundle_fingerprint": bundle.manifest.bundle_fingerprint,
            "scene_fingerprint": bundle.manifest.scene_fingerprint,
        },
        "geometry": {
            "path": _relative(geometry_path, repo_root),
            "file_sha256": str(cfg.geometry_file_sha256),
            "artifact_fingerprint": geometry["artifact_fingerprint"],
            "selected_candidate_id": selected["candidate_id"],
            "scene_from_court": selected["scene_from_court"],
            "court_from_scene": selected["court_from_scene"],
            "selected_symmetry": "positive-court-y-has-positive-plane-u",
        },
        "split": {
            "fit_group_ids": sorted({camera.group_id for camera in fit_cameras}),
            "holdout_group_ids": list(holdout_groups),
            "fit_camera_ids": [camera.camera_id for camera in fit_cameras],
            "holdout_camera_ids": [camera.camera_id for camera in holdout_cameras],
            "holdout_inference_status": "not_run",
        },
        "detector": {
            "checkpoint": _relative(_path(cfg.line_checkpoint), repo_root),
            "checkpoint_sha256": detector.checkpoint_sha256,
            "backbone_checkpoint_sha256": (detector.backbone_checkpoint_sha256),
            "short_side": detector.predictor.short_side,
            "checkpoint_epoch19_val_dice": float(cfg.checkpoint_val_dice),
            "training_best_val_dice": float(cfg.checkpoint_best_val_dice),
            "localization_error_available": False,
        },
        "evaluation_settings": asdict(evaluation_settings),
        "gates": {
            "fit": fit_gates,
            "holdout_frozen": _mapping(
                cfg.holdout_gates,
                name="holdout_gates",
            ),
            "threshold_basis": (
                "0.25 m is the fit weighted-q95 0.234 m rounded upward; "
                "checkpoint epoch19 val/dice=0.5770 and 256-short-side output "
                "are recorded because no pixel localization error is available"
            ),
        },
        "metrics": fit_metrics,
        "gate_results": gate_results,
        "stability": {
            "subsets": stability_records,
            "local_refit_settings": asdict(local_refit_settings),
            "selection_rule": (
                "local optimization locked to frozen court-0 physical cluster; "
                "transform comparison resolves 180-degree painted-court symmetry"
            ),
        },
        "point_cloud_support": point_support,
        "status": status,
        "provenance": _provenance(
            repo_root,
            code_files=code_files,
            command_module="src.synthetic_data_generation.scripts.calibrate_court_alignment",
            detector=detector,
        ),
    }
    published = publish_alignment_artifact(
        payload,
        output_dir=_path(cfg.output_dir),
    )
    print(published)
    if status != "fit_calibration_passed":
        raise RuntimeError(
            f"Fit calibration gates failed; artifact preserved at {published}."
        )
    return 0


def _fit_subset_stability(
    configured_subsets: Any,
    *,
    evidence_by_group: dict[int, np.ndarray[Any, Any]],
    reference_candidate: dict[str, Any],
    reference_scene_from_court: np.ndarray[Any, Any],
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    plane: GroundPlaneEstimate,
    settings: CourtLocalRefitSettings,
) -> list[dict[str, Any]]:
    records = []
    for index, subset_value in enumerate(configured_subsets):
        group_ids = [int(value) for value in subset_value]
        if not group_ids or not set(group_ids).issubset(evidence_by_group):
            raise ValueError(f"Invalid stability subset groups: {group_ids}.")
        evidence = np.sum(
            [evidence_by_group[group_id] for group_id in group_ids],
            axis=0,
            dtype=np.float32,
        )
        candidate = fit_court_instance_near_reference(
            evidence,
            bounds=bounds,
            grid_spacing=grid_spacing,
            plane=plane,
            reference_center_uv=tuple(reference_candidate["center_uv"]),
            reference_orientation_radians=float(
                reference_candidate["orientation_radians"]
            ),
            reference_scale_scene_per_metre=float(
                reference_candidate["scale_scene_per_metre"]
            ),
            settings=settings,
        )
        metrics = transform_stability(
            reference_scene_from_court,
            np.asarray(candidate.scene_from_court).reshape(4, 4),
        )
        records.append(
            {
                "subset_id": f"subset-{index}",
                "group_ids": group_ids,
                "refit_candidate_id": candidate.candidate_id,
                "refit_center_uv": list(candidate.center_uv),
                **metrics,
            }
        )
    return records


def _fit_gate_results(
    metrics: dict[str, Any],
    stability: list[dict[str, Any]],
    point_support: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, bool]:
    aggregate = metrics["aggregate"]
    return {
        "accepted_view_fraction": metrics["accepted_view_fraction"]
        >= gates["minimum_accepted_view_fraction"],
        "weighted_inlier_fraction": aggregate["weighted_inlier_fraction"]
        >= gates["minimum_weighted_inlier_fraction"],
        "distance_weighted_q95": aggregate["distance_weighted_q95_m"]
        <= gates["maximum_distance_weighted_q95_m"],
        "template_coverage": aggregate["template_coverage_fraction"]
        >= gates["minimum_template_coverage_fraction"],
        "stability_centre": max(item["centre_shift_m"] for item in stability)
        <= gates["maximum_stability_centre_shift_m"],
        "stability_orientation": max(
            item["orientation_difference_deg_mod_180"] for item in stability
        )
        <= gates["maximum_stability_orientation_deg"],
        "stability_scale": max(item["relative_scale_difference"] for item in stability)
        <= gates["maximum_stability_relative_scale_difference"],
        "point_support_count": point_support["support_point_count"]
        >= gates["minimum_point_support_count"],
        "point_support_rms": point_support["residual_rms_m"]
        <= gates["maximum_point_support_rms_m"],
        "point_grid_coverage": point_support["occupied_grid_fraction"]
        >= gates["minimum_point_grid_coverage_fraction"],
    }


def _selected_candidate(geometry: dict[str, Any]) -> dict[str, Any]:
    selected_id = geometry["selection"]["selected_candidate_id"]
    return next(
        candidate
        for candidate in geometry["candidates"]
        if candidate["candidate_id"] == selected_id
    )


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    raw = OmegaConf.to_container(value, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping.")
    return cast(dict[str, Any], raw)


def _provenance(
    repo_root: Path,
    *,
    code_files: tuple[Path, ...],
    command_module: str,
    detector: Any,
) -> dict[str, Any]:
    inventory = [
        {
            "path": _relative(path, repo_root),
            "sha256": str(sha256_file(path)),
        }
        for path in code_files
    ]
    return {
        "seed": 20260725,
        "git_revision": _git(repo_root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repo_root, "status", "--porcelain=v1")),
        "code_files": inventory,
        "code_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest(),
        "command": shlex.join([sys.executable, "-m", command_module, *sys.argv[1:]]),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "pytorch_lightning_version": pytorch_lightning.__version__,
        "opencv_version": cv2.__version__,
        "device": str(detector.predictor.device),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


if __name__ == "__main__":
    cast(Any, main)()
