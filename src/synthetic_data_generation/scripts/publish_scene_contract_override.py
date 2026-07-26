"""
Publish a SceneContract from an explicit user alignment override.

Usage:
    python -m src.synthetic_data_generation.scripts.publish_scene_contract_override

Notes:
    - Hydra loads `src/synthetic_data_generation/configs/publish_scene_contract_override.yaml`.
    - The rejected holdout artifact is preserved and referenced, never rewritten.
    - The contract alignment manifest points to the immutable user decision.
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

import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.synthetic_data_generation.alignment.acceptance_decision import (
    ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA,
    USER_OVERRIDE_DECISION,
    AlignmentAcceptanceDecision,
    load_alignment_acceptance_decision,
    publish_alignment_acceptance_decision,
    verify_machine_evidence,
)
from src.synthetic_data_generation.alignment.court_line_acceptance import (
    load_alignment_artifact,
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
    config_name="publish_scene_contract_override",
)
def main(cfg: DictConfig) -> int:
    """Verify the override evidence and publish its decision and contract."""
    repo_root = Path(to_absolute_path(".")).resolve()
    bundle = load_scene_provider_bundle(
        _path(cfg.provider_bundle),
        verify_files=bool(cfg.verify_provider_files),
    )
    if bundle.manifest.bundle_fingerprint != str(cfg.provider_bundle_fingerprint):
        raise ValueError("Provider bundle fingerprint mismatch.")

    calibration_path = _verified_path(
        cfg.calibration_artifact,
        expected_sha256=str(cfg.calibration_file_sha256),
    )
    calibration = load_alignment_artifact(calibration_path)
    if calibration["artifact_fingerprint"] != str(cfg.calibration_fingerprint):
        raise ValueError("Calibration artifact fingerprint mismatch.")

    validation_path = _verified_path(
        cfg.holdout_validation_artifact,
        expected_sha256=str(cfg.holdout_validation_file_sha256),
    )
    validation = load_alignment_artifact(validation_path)
    if validation["artifact_fingerprint"] != str(cfg.holdout_validation_fingerprint):
        raise ValueError("Holdout validation fingerprint mismatch.")

    source_path = _verified_path(
        cfg.decision.source_path,
        expected_sha256=str(cfg.decision.source_sha256),
    )
    code_files = (
        repo_root / "src/synthetic_data_generation/alignment/acceptance_decision.py",
        repo_root / "src/synthetic_data_generation/scene_contract.py",
        repo_root
        / "src/synthetic_data_generation/scripts/publish_scene_contract_override.py",
        repo_root
        / "src/synthetic_data_generation/configs/publish_scene_contract_override.yaml",
    )
    code_sha256 = _code_sha256(code_files, repo_root=repo_root)
    decision = AlignmentAcceptanceDecision(
        schema=ALIGNMENT_ACCEPTANCE_DECISION_SCHEMA,
        decision_id=str(cfg.decision_id),
        created_at_utc=datetime.now(UTC).isoformat(),
        decision=USER_OVERRIDE_DECISION,
        authority=str(cfg.decision.authority),
        reason=str(cfg.decision.reason),
        provider_bundle_fingerprint=bundle.manifest.bundle_fingerprint,
        selected_court_cluster=str(validation["geometry"]["selected_candidate_id"]),
        selected_symmetry=str(validation["geometry"]["selected_symmetry"]),
        machine_validation_status=str(validation["status"]),
        failed_gates=tuple(str(value) for value in cfg.decision.failed_gates),
        decision_source=_artifact_ref(
            "c06-user-override-source-v1",
            source_path,
            repo_root=repo_root,
        ),
        calibration=_artifact_ref(
            "b00-court-alignment-calibration-v2",
            calibration_path,
            repo_root=repo_root,
        ),
        holdout_validation=_artifact_ref(
            "b00-court-alignment-holdout-v1",
            validation_path,
            repo_root=repo_root,
        ),
        git_revision=_git(repo_root, "rev-parse", "HEAD"),
        git_dirty=bool(_git(repo_root, "status", "--porcelain=v1")),
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "src.synthetic_data_generation.scripts.publish_scene_contract_override",
                *sys.argv[1:],
            ]
        ),
        code_sha256=code_sha256,
    )
    verify_machine_evidence(
        decision,
        calibration=calibration,
        holdout_validation=validation,
    )
    decision_path = publish_alignment_acceptance_decision(
        decision,
        output_dir=_path(cfg.decision_output_dir),
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
        alignment_id=str(cfg.alignment_id),
        accepted=True,
        selected_court_cluster=decision.selected_court_cluster,
        selected_symmetry=decision.selected_symmetry,
        fit_camera_ids=tuple(calibration["split"]["fit_camera_ids"]),
        holdout_camera_ids=tuple(calibration["split"]["holdout_camera_ids"]),
        scene_from_court=transform,
        court_from_scene=transform.inverse(),
        manifest=_artifact_ref(
            str(cfg.decision_id),
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
    contract_path = _path(cfg.scene_contract_path)
    write_scene_contract(contract_path, contract)
    if load_scene_contract(contract_path) != contract:
        raise ValueError("Published scene contract failed strict reload.")
    print(decision_path)
    print(contract_path)
    return 0


def _artifact_ref(
    artifact_id: str,
    path: Path,
    *,
    repo_root: Path,
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=path.resolve().relative_to(repo_root).as_posix(),
        sha256=str(sha256_file(path)),
        size_bytes=path.stat().st_size,
    )


def _similarity(matrix: np.ndarray[Any, Any]) -> SimilarityTransform:
    scale = float(np.linalg.norm(matrix[:3, 0]))
    rotation = matrix[:3, :3] / scale
    return SimilarityTransform(
        scale=scale,
        rotation=tuple(float(value) for value in rotation.ravel()),
        translation=tuple(float(value) for value in matrix[:3, 3]),
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
            "path": path.resolve().relative_to(repo_root).as_posix(),
            "sha256": str(sha256_file(path)),
        }
        for path in paths
    ]
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _path(value: Any) -> Path:
    return Path(to_absolute_path(str(value))).resolve()


if __name__ == "__main__":
    cast(Any, main)()
