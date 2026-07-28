"""Unit tests for the renderer-independent tennis scene contract."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

import src.synthetic_data_generation.scene_contract as scene_contract_module
from src.synthetic_data_generation.scene_contract import (
    AcceptedAlignment,
    ArtifactRef,
    SceneCamera,
    SceneContract,
    SimilarityTransform,
    load_scene_contract,
    write_scene_contract,
)


def _artifact(artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=artifact_id,
        uri=f"artifact://provider/{artifact_id}",
        sha256="a" * 64,
        size_bytes=123,
    )


def _camera(camera_id: str, frame_index: int, group_id: int) -> SceneCamera:
    camera_to_scene = np.eye(4, dtype=np.float64)
    camera_to_scene[0, 3] = float(frame_index)
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="colmap-1",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=frame_index,
        group_id=group_id,
        width=959,
        height=539,
        intrinsics=(
            524.468,
            0.0,
            474.401,
            0.0,
            530.394,
            270.403,
            0.0,
            0.0,
            1.0,
        ),
        camera_to_scene=tuple(float(value) for value in camera_to_scene.ravel()),
    )


def _alignment() -> AcceptedAlignment:
    scene_from_court = SimilarityTransform(
        scale=0.25,
        rotation=(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        translation=(1.0, 2.0, 3.0),
    )
    return AcceptedAlignment(
        alignment_id="alignment-v1",
        accepted=True,
        selected_court_cluster="court-0",
        selected_symmetry="rotate-90",
        fit_camera_ids=("fit-0",),
        holdout_camera_ids=("holdout-0",),
        scene_from_court=scene_from_court,
        court_from_scene=scene_from_court.inverse(),
        manifest=_artifact("alignment-manifest"),
    )


def _contract() -> SceneContract:
    return SceneContract.create(
        scene_id="b00-tennis",
        provider_backend="gsplat-default@2b902ff",
        artifacts=(_artifact("checkpoint"), _artifact("camera-bundle")),
        cameras=(_camera("fit-0", 0, 0), _camera("holdout-0", 64, 2)),
        alignment=_alignment(),
    )


def test_similarity_round_trip_points() -> None:
    transform = _alignment().scene_from_court
    points = np.asarray([[0.0, 0.0, 0.0], [5.0, -2.0, 1.0]])

    recovered = transform.inverse().apply(transform.apply(points))

    np.testing.assert_allclose(recovered, points, atol=1.0e-12)


def test_scene_contract_json_round_trip_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    contract = _contract()
    path = tmp_path / "scene.json"

    write_scene_contract(path, contract)
    loaded = load_scene_contract(path)

    assert loaded == contract
    assert loaded.scene_fingerprint == contract.scene_fingerprint
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_scene_contract(path, contract)


def test_scene_fingerprint_is_order_independent() -> None:
    contract = _contract()

    reordered = SceneContract.create(
        scene_id=contract.scene_id,
        provider_backend=contract.provider_backend,
        artifacts=tuple(reversed(contract.artifacts)),
        cameras=tuple(reversed(contract.cameras)),
        alignment=contract.alignment,
    )

    assert reordered.scene_fingerprint == contract.scene_fingerprint


def test_contract_rejects_unknown_schema() -> None:
    payload = _contract().to_dict()
    payload["schema"] = "tennis_scene_contract_v2"

    with pytest.raises(ValueError, match="Unsupported scene contract schema"):
        SceneContract.from_dict(payload)


def test_contract_rejects_declared_fingerprint_mismatch() -> None:
    payload = _contract().to_dict()
    payload["scene_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="Scene fingerprint mismatch"):
        SceneContract.from_dict(payload)


def test_similarity_rejects_reflection() -> None:
    with pytest.raises(ValueError, match="proper rotation"):
        SimilarityTransform(
            scale=1.0,
            rotation=(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            translation=(0.0, 0.0, 0.0),
        )


def test_alignment_rejects_identity_fallback() -> None:
    identity = SimilarityTransform(
        scale=1.0,
        rotation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation=(0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="identity court alignment"):
        AcceptedAlignment(
            alignment_id="alignment-v1",
            accepted=True,
            selected_court_cluster="court-0",
            selected_symmetry="identity",
            fit_camera_ids=("fit-0",),
            holdout_camera_ids=("holdout-0",),
            scene_from_court=identity,
            court_from_scene=identity,
            manifest=_artifact("alignment-manifest"),
        )


def test_alignment_rejects_inconsistent_inverse() -> None:
    transform = _alignment().scene_from_court
    wrong_inverse = SimilarityTransform(
        scale=4.0,
        rotation=transform.rotation,
        translation=(0.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="are inconsistent"):
        AcceptedAlignment(
            alignment_id="alignment-v1",
            accepted=True,
            selected_court_cluster="court-0",
            selected_symmetry="rotate-90",
            fit_camera_ids=("fit-0",),
            holdout_camera_ids=("holdout-0",),
            scene_from_court=transform,
            court_from_scene=wrong_inverse,
            manifest=_artifact("alignment-manifest"),
        )


def test_scene_contract_has_no_task_or_renderer_backend_imports() -> None:
    tree = ast.parse(inspect.getsource(scene_contract_module))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden = (
        "gsplat",
        "src.tasks",
    )
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in imported_modules
        for prefix in forbidden
    )
