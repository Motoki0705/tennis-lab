"""
Run the one-shot B00 holdout validation against frozen fit-only gates.

Usage:
    python -m src.synthetic_data_generation.scripts.validate_court_alignment_holdout
    python -m src.synthetic_data_generation.scripts.validate_court_alignment_holdout device=cuda:0

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/validate_court_alignment_holdout.yaml`.
    - Gate values and the court transform come only from the immutable calibration.
    - An accepted scene contract is published only when every frozen gate passes.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytorch_lightning
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.synthetic_data_generation.alignment.court_line_acceptance import (
    ALIGNMENT_VALIDATION_SCHEMA,
    CourtLineEvaluationSettings,
    camera_heights_in_court,
    evaluate_projected_court_lines,
    holdout_gate_results,
    load_alignment_artifact,
    publish_alignment_artifact,
)
from src.synthetic_data_generation.alignment.fit_view_detection import (
    partition_fit_and_holdout_cameras,
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
from src.synthetic_data_generation.provider.bundle import (
    load_scene_provider_bundle,
    sha256_file,
)
from src.synthetic_data_generation.scene_contract import (
    AcceptedAlignment,
    ArtifactRef,
    SceneContract,
    SimilarityTransform,
    load_scene_contract,
    write_scene_contract,
)
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="validate_court_alignment_holdout",
)
def main(cfg: DictConfig) -> int:
    """Evaluate untouched holdout cameras once and publish the result."""
    repo_root = Path(to_absolute_path(".")).resolve()
    calibration_path = _path(cfg.calibration_artifact)
    if str(sha256_file(calibration_path)) != str(cfg.calibration_file_sha256):
        raise ValueError("Calibration artifact file SHA-256 mismatch.")
    calibration = load_alignment_artifact(calibration_path)
    if calibration["artifact_fingerprint"] != str(cfg.calibration_fingerprint):
        raise ValueError("Calibration artifact fingerprint mismatch.")
    if calibration["status"] != "fit_calibration_passed":
        raise ValueError("Holdout validation requires passed fit calibration.")
    if calibration["split"]["holdout_inference_status"] != "not_run":
        raise ValueError("Calibration already touched holdout images.")

    line_path = _path(cfg.ground_line_artifact)
    line_manifest, _ = load_ground_line_map_artifact(line_path)
    if line_manifest["artifact_fingerprint"] != str(cfg.ground_line_fingerprint):
        raise ValueError("Ground-line artifact fingerprint mismatch.")
    bundle = load_scene_provider_bundle(
        _path(cfg.provider_bundle),
        verify_files=bool(cfg.verify_provider_files),
    )
    holdout_groups = tuple(int(value) for value in cfg.holdout_group_ids)
    fit_cameras, holdout_cameras = partition_fit_and_holdout_cameras(
        bundle.manifest.cameras,
        holdout_group_ids=holdout_groups,
    )
    if [camera.camera_id for camera in fit_cameras] != calibration["split"][
        "fit_camera_ids"
    ]:
        raise ValueError("Current fit camera ids differ from frozen calibration.")
    if [camera.camera_id for camera in holdout_cameras] != calibration["split"][
        "holdout_camera_ids"
    ]:
        raise ValueError("Current holdout camera ids differ from frozen calibration.")
    if list(holdout_groups) != calibration["split"]["holdout_group_ids"]:
        raise ValueError("Current holdout groups differ from frozen calibration.")

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
    if detector.checkpoint_sha256 != calibration["detector"]["checkpoint_sha256"]:
        raise ValueError("Validation detector differs from calibration detector.")
    plane = GroundPlaneEstimate(**line_manifest["ground_plane"]["estimate"])
    bounds = cast(
        tuple[float, float, float, float],
        tuple(float(value) for value in line_manifest["projection"]["bounds_uv"]),
    )
    projection_settings = GroundLineMapSettings(
        **line_manifest["projection"]["settings"]
    )
    collected = collect_projected_line_evidence(
        holdout_cameras,
        bundle=bundle,
        detector=detector,
        plane=plane,
        bounds=bounds,
        settings=projection_settings,
    )

    court_from_scene = np.asarray(calibration["geometry"]["court_from_scene"]).reshape(
        4, 4
    )
    scene_from_court = np.asarray(calibration["geometry"]["scene_from_court"]).reshape(
        4, 4
    )
    evaluation_settings = CourtLineEvaluationSettings(
        **calibration["evaluation_settings"]
    )
    aggregate = evaluate_projected_court_lines(
        collected.points_scene,
        collected.weights,
        court_from_scene=court_from_scene,
        settings=evaluation_settings,
    )
    by_group = {
        str(group_id): evaluate_projected_court_lines(
            collected.points_by_group[group_id],
            collected.weights_by_group[group_id],
            court_from_scene=court_from_scene,
            settings=evaluation_settings,
        )
        for group_id in sorted(collected.points_by_group)
    }
    records = []
    for record in collected.records:
        camera_id = str(record["camera_id"])
        records.append(
            {
                **record,
                "fixed_transform_metrics": evaluate_projected_court_lines(
                    collected.points_by_camera[camera_id],
                    collected.weights_by_camera[camera_id],
                    court_from_scene=court_from_scene,
                    settings=evaluation_settings,
                ),
            }
        )
    camera_centres = np.asarray(
        [
            np.asarray(camera.camera_to_scene).reshape(4, 4)[:3, 3]
            for camera in holdout_cameras
        ]
    )
    camera_heights = camera_heights_in_court(
        camera_centres,
        court_from_scene=court_from_scene,
    )
    accepted_by_group = {
        str(group_id): sum(
            record["accepted"] and record["group_id"] == group_id for record in records
        )
        for group_id in holdout_groups
    }
    accepted_count = sum(record["accepted"] for record in records)
    metrics = {
        "camera_count": len(holdout_cameras),
        "accepted_view_count": accepted_count,
        "accepted_view_fraction": accepted_count / len(holdout_cameras),
        "accepted_view_count_by_group": accepted_by_group,
        "projected_line_pixel_count": int(
            sum(record["projected_line_pixel_count"] for record in records)
        ),
        "aggregate": aggregate,
        "by_group": by_group,
        "camera_heights_m": {
            "minimum": float(np.min(camera_heights)),
            "median": float(np.median(camera_heights)),
            "maximum": float(np.max(camera_heights)),
            "positive_fraction": float(np.mean(camera_heights > 0.0)),
        },
    }
    gates = calibration["gates"]["holdout_frozen"]
    gate_results = holdout_gate_results(metrics, gates)
    status = "accepted" if all(gate_results.values()) else "rejected"
    code_files = (
        repo_root / "src/synthetic_data_generation/alignment/court_line_acceptance.py",
        repo_root
        / "src/synthetic_data_generation/alignment/line_evidence_collection.py",
        repo_root / "src/synthetic_data_generation/alignment/line_inference.py",
        repo_root
        / "src/synthetic_data_generation/scripts/validate_court_alignment_holdout.py",
        repo_root
        / "src/synthetic_data_generation/configs/validate_court_alignment_holdout.yaml",
    )
    payload = {
        "schema": ALIGNMENT_VALIDATION_SCHEMA,
        "artifact_id": str(cfg.artifact_id),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provider": {
            "bundle_id": bundle.manifest.bundle_id,
            "bundle_fingerprint": bundle.manifest.bundle_fingerprint,
            "scene_fingerprint": bundle.manifest.scene_fingerprint,
        },
        "geometry": calibration["geometry"],
        "calibration": {
            "path": _relative(calibration_path, repo_root),
            "file_sha256": str(cfg.calibration_file_sha256),
            "artifact_fingerprint": calibration["artifact_fingerprint"],
            "status": calibration["status"],
        },
        "split": {
            "fit_group_ids": calibration["split"]["fit_group_ids"],
            "holdout_group_ids": list(holdout_groups),
            "fit_camera_ids": [camera.camera_id for camera in fit_cameras],
            "holdout_camera_ids": [camera.camera_id for camera in holdout_cameras],
            "holdout_inference_status": "complete",
            "selection_or_tuning_after_holdout": False,
        },
        "detector": {
            "checkpoint_sha256": detector.checkpoint_sha256,
            "backbone_checkpoint_sha256": (detector.backbone_checkpoint_sha256),
            "short_side": detector.predictor.short_side,
        },
        "evaluation_settings": calibration["evaluation_settings"],
        "gates": gates,
        "metrics": metrics,
        "gate_results": gate_results,
        "records": records,
        "status": status,
        "provenance": _provenance(
            repo_root,
            code_files=code_files,
            detector=detector,
        ),
    }
    validation_path = publish_alignment_artifact(
        payload,
        output_dir=_path(cfg.output_dir),
    )
    print(validation_path)
    if status != "accepted":
        raise RuntimeError(
            "Frozen holdout gates rejected the alignment; "
            f"artifact preserved at {validation_path}."
        )
    _publish_scene_contract(
        cfg,
        repo_root=repo_root,
        bundle=bundle,
        calibration=calibration,
        validation_path=validation_path,
        scene_from_court=scene_from_court,
    )
    return 0


def _publish_scene_contract(
    cfg: DictConfig,
    *,
    repo_root: Path,
    bundle: Any,
    calibration: dict[str, Any],
    validation_path: Path,
    scene_from_court: np.ndarray[Any, Any],
) -> None:
    validation_ref = ArtifactRef(
        artifact_id="b00-court-alignment-holdout-v1",
        uri=_relative(validation_path, repo_root),
        sha256=str(sha256_file(validation_path)),
        size_bytes=validation_path.stat().st_size,
    )
    transform = _similarity(scene_from_court)
    alignment = AcceptedAlignment(
        alignment_id=str(cfg.alignment_id),
        accepted=True,
        selected_court_cluster=calibration["geometry"]["selected_candidate_id"],
        selected_symmetry=str(cfg.selected_symmetry),
        fit_camera_ids=tuple(calibration["split"]["fit_camera_ids"]),
        holdout_camera_ids=tuple(calibration["split"]["holdout_camera_ids"]),
        scene_from_court=transform,
        court_from_scene=transform.inverse(),
        manifest=validation_ref,
    )
    contract = SceneContract.create(
        scene_id=str(cfg.scene_id),
        provider_backend=bundle.manifest.provider_backend,
        artifacts=bundle.manifest.source_artifacts,
        cameras=bundle.manifest.cameras,
        alignment=alignment,
    )
    if contract.scene_fingerprint != bundle.manifest.scene_fingerprint:
        raise ValueError("Accepted contract changed the provider scene fingerprint.")
    contract_path = _path(cfg.scene_contract_path)
    write_scene_contract(contract_path, contract)
    load_scene_contract(contract_path)
    print(contract_path)


def _similarity(matrix: np.ndarray[Any, Any]) -> SimilarityTransform:
    scale = float(np.linalg.norm(matrix[:3, 0]))
    rotation = matrix[:3, :3] / scale
    return SimilarityTransform(
        scale=scale,
        rotation=tuple(float(value) for value in rotation.ravel()),
        translation=tuple(float(value) for value in matrix[:3, 3]),
    )


def _provenance(
    repo_root: Path,
    *,
    code_files: tuple[Path, ...],
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
        "command": shlex.join(
            [
                sys.executable,
                "-m",
                "src.synthetic_data_generation.scripts.validate_court_alignment_holdout",
                *sys.argv[1:],
            ]
        ),
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
