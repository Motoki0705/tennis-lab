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
    assert len(manifest.artifacts) == 5


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
