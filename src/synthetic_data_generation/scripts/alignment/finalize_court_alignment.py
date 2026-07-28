"""
Validate untouched holdout views and finalize the SceneContract.

Usage:
    python -m src.synthetic_data_generation.scripts.alignment.finalize_court_alignment
    python -m src.synthetic_data_generation.scripts.alignment.finalize_court_alignment device=cuda:0

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/alignment/finalize_court_alignment.yaml`.
    - Gate values and the court transform come only from the immutable calibration.
    - A rejected result is finalized only through an explicit verified user override.
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

from src.synthetic_data_generation.alignment.artifacts.acceptance_decision import (
    ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA,
    USER_OVERRIDE_DECISION,
    AlignmentAcceptanceDecision,
    load_alignment_acceptance_decision,
    publish_alignment_acceptance_decision,
    verify_machine_evidence,
)
from src.synthetic_data_generation.alignment.artifacts.calibration import (
    load_calibration_artifact,
)
from src.synthetic_data_generation.alignment.artifacts.ground_line_map import (
    load_ground_line_map_artifact,
)
from src.synthetic_data_generation.alignment.artifacts.holdout_validation import (
    ALIGNMENT_VALIDATION_SCHEMA,
    load_holdout_validation_artifact,
    publish_holdout_validation_artifact,
)
from src.synthetic_data_generation.alignment.components.evaluation.court_lines import (
    CourtLineEvaluationSettings,
    camera_heights_in_court,
    evaluate_projected_court_lines,
    holdout_gate_results,
)
from src.synthetic_data_generation.alignment.components.evidence.collection import (
    collect_projected_line_evidence,
)
from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
    GroundLineMapSettings,
)
from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneEstimate,
)
from src.synthetic_data_generation.alignment.components.inference.line_detector import (
    load_verified_line_detector,
)
from src.synthetic_data_generation.alignment.components.inputs.view_inputs import (
    partition_fit_and_holdout_cameras,
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
from src.synthetic_data_generation.scripts.alignment.common import (
    StageResult,
    json_artifact_handle,
    print_stage_result,
)
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../../configs/alignment",
    config_name="finalize_court_alignment",
)
def main(cfg: DictConfig) -> int:
    """Run the stage through its shared orchestration entry point."""
    print_stage_result(run(cfg))
    return 0


def run(cfg: DictConfig) -> StageResult:
    """Evaluate untouched holdout cameras and finalize the result."""
    repo_root = Path(to_absolute_path(".")).resolve()
    calibration_path = _path(cfg.calibration_artifact)
    if str(sha256_file(calibration_path)) != str(cfg.calibration_file_sha256):
        raise ValueError("Calibration artifact file SHA-256 mismatch.")
    calibration = load_calibration_artifact(calibration_path)
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
        repo_root
        / "src/synthetic_data_generation/alignment/components/evaluation/court_lines.py",
        repo_root
        / "src/synthetic_data_generation/alignment/components/evidence/collection.py",
        repo_root
        / "src/synthetic_data_generation/alignment/components/inference/line_detector.py",
        repo_root
        / "src/synthetic_data_generation/alignment/artifacts/holdout_validation.py",
        repo_root
        / "src/synthetic_data_generation/scripts/alignment/finalize_court_alignment.py",
        repo_root
        / "src/synthetic_data_generation/configs/alignment/finalize_court_alignment.yaml",
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
    validation_path = publish_holdout_validation_artifact(
        payload,
        output_dir=_path(cfg.output_dir),
    )
    validation = load_holdout_validation_artifact(validation_path)
    validation_handle = json_artifact_handle(validation_path, validation)
    if status == "accepted":
        contract_path = _publish_scene_contract(
            cfg,
            repo_root=repo_root,
            bundle=bundle,
            calibration=calibration,
            validation_path=validation_path,
            scene_from_court=scene_from_court,
        )
        return StageResult(
            stage="finalization",
            status="accepted",
            artifact_paths=(validation_path, contract_path),
            primary_artifact=validation_path,
            fingerprint=validation_handle.fingerprint,
            metadata={
                "artifact": validation_handle.to_dict(),
                "scene_contract": str(contract_path),
            },
        )

    override_cfg = cfg.get("override")
    if isinstance(override_cfg, DictConfig) and bool(
        override_cfg.get("enabled", False)
    ):
        decision_path, contract_path = _publish_user_override(
            cfg,
            override_cfg=override_cfg,
            repo_root=repo_root,
            bundle=bundle,
            calibration=calibration,
            calibration_path=calibration_path,
            validation=validation,
            validation_path=validation_path,
        )
        _, decision_fingerprint = load_alignment_acceptance_decision(decision_path)
        return StageResult(
            stage="finalization",
            status="accepted_by_user_override",
            artifact_paths=(validation_path, decision_path, contract_path),
            primary_artifact=validation_path,
            fingerprint=validation_handle.fingerprint,
            metadata={
                "artifact": validation_handle.to_dict(),
                "decision": {
                    "path": str(decision_path),
                    "fingerprint": decision_fingerprint,
                    "file_sha256": str(sha256_file(decision_path)),
                },
                "scene_contract": str(contract_path),
            },
        )
    return StageResult(
        stage="finalization",
        status="rejected",
        artifact_paths=(validation_path,),
        primary_artifact=validation_path,
        fingerprint=validation_handle.fingerprint,
        metadata={
            "artifact": validation_handle.to_dict(),
            "failed_gates": sorted(
                name for name, passed in gate_results.items() if not passed
            ),
            "scene_contract": None,
        },
    )


def _publish_scene_contract(
    cfg: DictConfig,
    *,
    repo_root: Path,
    bundle: Any,
    calibration: dict[str, Any],
    validation_path: Path,
    scene_from_court: np.ndarray[Any, Any],
) -> Path:
    validation_ref = ArtifactRef(
        artifact_id=str(cfg.artifact_id),
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
    if load_scene_contract(contract_path) != contract:
        raise ValueError("Published scene contract failed strict reload.")
    return contract_path


def _publish_user_override(
    cfg: DictConfig,
    *,
    override_cfg: DictConfig,
    repo_root: Path,
    bundle: Any,
    calibration: dict[str, Any],
    calibration_path: Path,
    validation: dict[str, Any],
    validation_path: Path,
) -> tuple[Path, Path]:
    declared_provider_fingerprint = str(
        override_cfg.get(
            "provider_bundle_fingerprint",
            bundle.manifest.bundle_fingerprint,
        )
    )
    if bundle.manifest.bundle_fingerprint != declared_provider_fingerprint:
        raise ValueError("Override provider bundle fingerprint mismatch.")
    source_path = _verified_path(
        override_cfg.decision.source_path,
        expected_sha256=str(override_cfg.decision.source_sha256),
    )
    code_files = (
        repo_root
        / "src/synthetic_data_generation/alignment/artifacts/acceptance_decision.py",
        repo_root / "src/synthetic_data_generation/scene_contract.py",
        repo_root
        / "src/synthetic_data_generation/scripts/alignment/finalize_court_alignment.py",
        repo_root
        / "src/synthetic_data_generation/configs/alignment/finalize_court_alignment.yaml",
    )
    decision = AlignmentAcceptanceDecision(
        schema=ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA,
        decision_id=str(override_cfg.decision_id),
        created_at_utc=datetime.now(UTC).isoformat(),
        decision=USER_OVERRIDE_DECISION,
        authority=str(override_cfg.decision.authority),
        reason=str(override_cfg.decision.reason),
        provider_bundle_fingerprint=bundle.manifest.bundle_fingerprint,
        selected_court_cluster=str(validation["geometry"]["selected_candidate_id"]),
        selected_symmetry=str(validation["geometry"]["selected_symmetry"]),
        machine_validation_status=str(validation["status"]),
        failed_gates=tuple(str(value) for value in override_cfg.decision.failed_gates),
        decision_source=_artifact_ref(
            str(
                override_cfg.decision.get(
                    "source_artifact_id",
                    "alignment-user-override-source-v1",
                )
            ),
            source_path,
            repo_root=repo_root,
        ),
        calibration=_artifact_ref(
            str(calibration["artifact_id"]),
            calibration_path,
            repo_root=repo_root,
        ),
        holdout_validation=_artifact_ref(
            str(validation["artifact_id"]),
            validation_path,
            repo_root=repo_root,
        ),
        git_revision=_git(repo_root, "rev-parse", "HEAD"),
        git_dirty=bool(_git(repo_root, "status", "--porcelain=v1")),
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "src.synthetic_data_generation.scripts.alignment."
                "finalize_court_alignment",
                *sys.argv[1:],
            ]
        ),
        code_sha256=_code_sha256(code_files, repo_root=repo_root),
    )
    verify_machine_evidence(
        decision,
        calibration=calibration,
        holdout_validation=validation,
    )
    decision_path = publish_alignment_acceptance_decision(
        decision,
        output_dir=_path(override_cfg.decision_output_dir),
    )
    loaded_decision, _ = load_alignment_acceptance_decision(decision_path)
    if loaded_decision != decision:
        raise ValueError("Published acceptance decision failed strict reload.")

    scene_from_court = np.asarray(
        calibration["geometry"]["scene_from_court"],
        dtype=np.float64,
    ).reshape(4, 4)
    transform = _similarity(scene_from_court)
    alignment = AcceptedAlignment(
        alignment_id=str(override_cfg.get("alignment_id", cfg.alignment_id)),
        accepted=True,
        selected_court_cluster=decision.selected_court_cluster,
        selected_symmetry=decision.selected_symmetry,
        fit_camera_ids=tuple(calibration["split"]["fit_camera_ids"]),
        holdout_camera_ids=tuple(calibration["split"]["holdout_camera_ids"]),
        scene_from_court=transform,
        court_from_scene=transform.inverse(),
        manifest=_artifact_ref(
            str(override_cfg.decision_id),
            decision_path,
            repo_root=repo_root,
        ),
    )
    contract = SceneContract.create(
        scene_id=str(cfg.scene_id),
        provider_backend=bundle.manifest.provider_backend,
        artifacts=bundle.manifest.source_artifacts,
        cameras=bundle.manifest.cameras,
        alignment=alignment,
    )
    if contract.scene_fingerprint != bundle.manifest.scene_fingerprint:
        raise ValueError("Scene contract changed the provider scene fingerprint.")
    contract_path = _path(override_cfg.scene_contract_path)
    write_scene_contract(contract_path, contract)
    if load_scene_contract(contract_path) != contract:
        raise ValueError("Published override scene contract failed strict reload.")
    return decision_path, contract_path


def _artifact_ref(
    artifact_id: str,
    path: Path,
    *,
    repo_root: Path,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=_relative(path, repo_root),
        sha256=str(sha256_file(path)),
        size_bytes=path.stat().st_size,
    )


def _verified_path(value: Any, *, expected_sha256: str) -> Path:
    path = _path(value)
    actual = str(sha256_file(path))
    if actual != expected_sha256:
        raise ValueError(
            f"Artifact SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}."
        )
    return path


def _code_sha256(paths: tuple[Path, ...], *, repo_root: Path) -> str:
    inventory = [
        {
            "path": _relative(path, repo_root),
            "sha256": str(sha256_file(path)),
        }
        for path in paths
    ]
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


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
                "src.synthetic_data_generation.scripts.alignment."
                "finalize_court_alignment",
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
