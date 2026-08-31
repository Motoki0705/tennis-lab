"""Unit tests for publication bundle inventory and content validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.publication.bundle as bundle_module
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.synthetic_data_generation.visualization.publication.bundle import (
    validate_publication_bundle_structure_only,
)


def test_structure_only_validator_accepts_complete_fixture(
    valid_publication_bundle: Path,
) -> None:
    manifest = validate_publication_bundle_structure_only(valid_publication_bundle)

    assert manifest.scene_id == "scene-0"
    assert len(manifest.artifacts) == 10


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_structure_only_validator_rejects_missing_or_extra_media(
    valid_publication_bundle: Path,
    mutation: str,
) -> None:
    if mutation == "missing":
        (valid_publication_bundle / "dataset-court.gif").unlink()
    else:
        (valid_publication_bundle / "foreign-media.bin").write_bytes(b"foreign")

    with pytest.raises(ValueError, match="inventory differs"):
        validate_publication_bundle_structure_only(valid_publication_bundle)


def test_structure_only_validator_rejects_tampered_manifest(
    valid_publication_bundle: Path,
) -> None:
    manifest_path = valid_publication_bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["scene_id"] = "foreign-scene"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Every source owner must bind"):
        validate_publication_bundle_structure_only(valid_publication_bundle)


def test_structure_only_validator_rejects_tampered_media_digest(
    valid_publication_bundle: Path,
) -> None:
    media_path = valid_publication_bundle / "dataset-court.gif"
    data = bytearray(media_path.read_bytes())
    data[-1] ^= 1
    media_path.write_bytes(data)

    with pytest.raises(ValueError, match="content digest changed"):
        validate_publication_bundle_structure_only(valid_publication_bundle)


def test_structure_only_validator_rejects_tampered_sampled_camera_mapping(
    valid_publication_bundle: Path,
) -> None:
    manifest_path = valid_publication_bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    captured = next(
        artifact
        for artifact in payload["artifacts"]
        if artifact["file_name"] == "captured-camera-trajectory.png"
    )
    captured["mapping"][0]["rendered_camera_indices"] = [0]
    captured["mapping"][0]["rendered_camera_ids"] = ["cam-0"]
    captured["mapping"][0]["rendered_camera_count"] = 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic drawing policy"):
        validate_publication_bundle_structure_only(valid_publication_bundle)


def test_publication_matrix_validation_accepts_canonical_valid_numeric_drift() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] += 5.0e-8
    assert np.max(np.abs(matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3))) == (
        pytest.approx(1.0e-7)
    )
    RigidTransform.from_matrix(matrix)

    validated = bundle_module._finite_matrix4(
        matrix.tolist(), name="camera_pose.camera_to_metric_scene"
    )

    np.testing.assert_array_equal(validated, matrix)


def test_manifest_comparison_recomputes_identical_drift_as_six_of_six() -> None:
    angle = 1.1884684684684685
    forward = np.asarray(
        (0.6 * np.sin(angle), 0.8 * np.sin(angle), np.cos(angle)),
        dtype=np.float64,
    )
    right = np.cross(np.asarray((0.0, 1.0, 0.0)), forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward * (1.0 + 5.0e-8)))
    RigidTransform.from_matrix(matrix)
    matrices = np.stack([matrix.copy() for _ in range(6)])
    matrices[:, 0, 3] = np.arange(6, dtype=np.float64)

    metrics = bundle_module._camera_comparison_metrics_from_poses(
        camera_ids=tuple(f"camera-{index}" for index in range(6)),
        blcs_matrices=matrices,
        plcs_matrices=matrices.copy(),
        centre_tolerance_metres=1.0e-6,
        forward_angle_tolerance_degrees=1.0e-6,
    )

    assert metrics["coincident_camera_count"] == 6
    assert metrics["coincident_camera_fraction"] == 1.0
    assert metrics["maximum_forward_angle_difference_degrees"] == 0.0


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([1.0 + 5.1e-7, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        np.diag([1.001, 1.001, 1.001, 1.0]),
    ],
    ids=["just-beyond-canonical-tolerance", "reflection", "scaled-rotation"],
)
def test_publication_matrix_validation_rejects_noncanonical_rotations(
    matrix: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError):
        RigidTransform.from_matrix(matrix)

    with pytest.raises(ValueError):
        bundle_module._finite_matrix4(
            matrix.tolist(), name="camera_pose.camera_to_metric_scene"
        )


@pytest.mark.parametrize(
    "value",
    [
        np.full((4, 4), np.nan, dtype=np.float64).tolist(),
        np.eye(3, dtype=np.float64).tolist(),
        np.asarray(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0 + 2.0e-6),
            ),
            dtype=np.float64,
        ).tolist(),
    ],
    ids=["non-finite", "wrong-shape", "non-homogeneous"],
)
def test_publication_matrix_validation_preserves_canonical_rigid_failures(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        bundle_module._finite_matrix4(value, name="camera_pose.camera_to_metric_scene")
